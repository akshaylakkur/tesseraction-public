# Source: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import shutil
import sys
import time

import numpy as np

logger = logging.getLogger("nv_physical_ai")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
DATA_ROOT = pathlib.Path("...")
REVISION = "..."   # dataset commit pinned for reproducibility
CAMERA = "camera_front_wide_120fov"
WASD = ["W", "A", "S", "D"]                               # same order as anna.py


@dataclasses.dataclass(frozen=True)
class PreprocessConfig:
    camera: str = CAMERA
    fps: float = 10.0                 # output frame rate; the camera is 30 fps, so stride 3
    size: tuple[int, int] = (256, 256)  # (H, W)
    speed_on: float = 0.5             # m/s, W / S threshold
    yaw_rate_on: float = 0.09         # rad/s, A / D threshold
    val_fraction: float = 0.1
    revision: str = REVISION

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(dataclasses.asdict(self), sort_keys=True).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------------------------------------

def hf_token() -> str | None:
    """HF_TOKEN from the environment, else from the repo-root .env."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "HF_TOKEN":
                return value.strip().strip("'\"")
    return None


def open_dataset(cache_dir: pathlib.Path = DATA_ROOT / "hf_cache", revision: str = REVISION):
    """The dev-kit interface on the pinned revision; served from the cache once metadata is downloaded."""
    from physical_ai_av import PhysicalAIAVDatasetInterface

    return PhysicalAIAVDatasetInterface(
        revision=revision, token=hf_token(), cache_dir=cache_dir, confirm_download_threshold_gb=float("inf"),
    )


def download(chunks: list[int], camera: str = CAMERA, cache_dir: pathlib.Path = DATA_ROOT / "hf_cache") -> None:
    """Fetch, per chunk, the camera zip, the egomotion zip and the calibration parquets.

    Files go through `download_file` one at a time: `download_chunk_features` hands huggingface_hub>=1.x a
    progress class without `set_postfix_str`, which crashes the xet transfer reporter.
    """
    ds = open_dataset(cache_dir)
    features = [camera, "egomotion", "camera_intrinsics", "sensor_extrinsics", "vehicle_dimensions"]
    for chunk in chunks:
        for feature in features:
            filename = ds.features.get_chunk_feature_filename(chunk, feature)
            logger.info("chunk %d: %s", chunk, filename)
            ds.download_file(filename)


# ---------------------------------------------------------------------------------------------------------
# Per-clip parsing (pure functions are separated out so they can be tested without video)
# ---------------------------------------------------------------------------------------------------------

def select_frames(video_ts: np.ndarray, ego_range: tuple[int, int], stride: int) -> np.ndarray:
    """int64 indices i_0 + k * stride of the video frames whose timestamps lie inside ego_range."""
    valid = np.flatnonzero((video_ts >= ego_range[0]) & (video_ts <= ego_range[1]))
    if len(valid) == 0:
        return valid.astype(np.int64)
    return np.arange(valid[0], valid[-1] + 1, stride, dtype=np.int64)


def relative_motion(pose, timestamps_us: np.ndarray) -> np.ndarray:
    """pose: scipy RigidTransform [K] (rig -> world) -> ego [K-1, 3] = (v_forward, v_left, yaw_rate)."""
    rel = pose[:-1].inv() * pose[1:]                       # P_k^-1 o P_{k+1}, in the rig frame at k
    dt = np.diff(timestamps_us).astype(np.float64) * 1e-6
    yaw = rel.rotation.as_euler("ZYX")[:, 0]               # intrinsic yaw, + = counter-clockwise (left)
    t = rel.translation
    return np.stack([t[:, 0] / dt, t[:, 1] / dt, yaw / dt], axis=1).astype(np.float32)


def ego_to_actions(ego: np.ndarray, speed_on: float, yaw_rate_on: float) -> np.ndarray:
    """ego [N, 3] -> binary WASD actions [N, 4] float32."""
    v, w = ego[:, 0], ego[:, 2]
    return np.stack([v > speed_on, w > yaw_rate_on, v < -speed_on, w < -yaw_rate_on], axis=1).astype(np.float32)


def crop_resize(images: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """[N, h, w, 3] uint8 -> [N, H, W, 3] uint8: center crop to H:W aspect, then BOX (area) resample."""
    from PIL import Image

    H, W = size
    h, w = images.shape[1:3]
    if w * H > h * W:                                      # too wide: crop width
        cw, ch = round(h * W / H), h
    else:                                                  # too tall: crop height
        cw, ch = w, round(w * H / W)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    out = np.empty((len(images), H, W, 3), np.uint8)
    for i, img in enumerate(images):
        crop = Image.fromarray(img[y0:y0 + ch, x0:x0 + cw])
        out[i] = np.asarray(crop.resize((W, H), Image.Resampling.BOX))
    return out


def parse_clip(ds, clip_id: str, cfg: PreprocessConfig, decode_batch: int = 32) -> dict[str, np.ndarray]:
    """One clip -> frames [K, H, W, 3] uint8, timestamps [K] int64 (us), ego [K-1, 3], actions [K-1, 4]."""
    video = ds.get_clip_feature(clip_id, cfg.camera)
    ego_interp = ds.get_clip_feature(clip_id, "egomotion")
    try:
        stride = max(1, round(float(video.fps) / cfg.fps))
        idx = select_frames(video.timestamps, ego_interp.time_range, stride)
        if len(idx) < 2:
            raise ValueError(f"{clip_id}: {len(idx)} frames inside the egomotion range")
        frames = np.concatenate([                          # decode in batches: a 1080p frame is 6 MB
            crop_resize(video.decode_images_from_frame_indices(idx[s:s + decode_batch]), cfg.size)
            for s in range(0, len(idx), decode_batch)
        ])
    finally:
        video.close()
    ts = video.timestamps[idx].astype(np.int64)
    ego = relative_motion(ego_interp(ts).pose, ts)
    if not np.isfinite(ego).all():
        raise ValueError(f"{clip_id}: non-finite ego-motion")
    return {"frames": frames, "timestamps": ts, "ego": ego,
            "actions": ego_to_actions(ego, cfg.speed_on, cfg.yaw_rate_on)}


def split_of(clip_id: str, val_fraction: float) -> str:
    return "val" if int(hashlib.sha1(clip_id.encode()).hexdigest(), 16) % 1000 < 1000 * val_fraction else "train"


# ---------------------------------------------------------------------------------------------------------
# Preprocess: cache -> <out>/clips/<clip_id>/{frames,timestamps,ego,actions}.npy
# ---------------------------------------------------------------------------------------------------------

_WORKER_DS = None


def _worker_init(cache_dir: str, revision: str) -> None:
    global _WORKER_DS
    os.environ["HF_HUB_OFFLINE"] = "1"                     # everything needed is in the cache by now
    _WORKER_DS = open_dataset(pathlib.Path(cache_dir), revision)


def _process_one(clip_id: str, cfg: PreprocessConfig, clips_dir: str) -> tuple[str, int, float, str | None]:
    t0 = time.perf_counter()
    final = pathlib.Path(clips_dir) / clip_id
    tmp = pathlib.Path(clips_dir) / f".{clip_id}.tmp"
    try:
        rec = parse_clip(_WORKER_DS, clip_id, cfg)
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for k, v in rec.items():
            np.save(tmp / f"{k}.npy", v)
        tmp.rename(final)                                  # atomic: a clip dir exists only when complete
        return clip_id, len(rec["frames"]), time.perf_counter() - t0, None
    except Exception as e:                                 # report and keep going; listed in meta.json
        shutil.rmtree(tmp, ignore_errors=True)
        return clip_id, 0, time.perf_counter() - t0, f"{type(e).__name__}: {e}"


def preprocess(chunks: list[int], cfg: PreprocessConfig = PreprocessConfig(), out_dir: pathlib.Path | None = None,
               cache_dir: pathlib.Path = DATA_ROOT / "hf_cache", workers: int = 8,
               limit: int | None = None) -> pathlib.Path:
    """Decode every clip of `chunks` into `out_dir` (default DATA_ROOT/processed/<cfg hash>). Resumable."""
    out_dir = out_dir or DATA_ROOT / "processed" / cfg.hash()
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    ds = open_dataset(cache_dir, cfg.revision)
    index = ds.clip_index[ds.clip_index["chunk"].isin(chunks) & ds.clip_index["clip_is_valid"]]
    clip_ids = sorted(index.index)[:limit]
    todo = [c for c in clip_ids if not (clips_dir / c).is_dir()]
    logger.info("%d clips in chunks %s, %d to process -> %s", len(clip_ids), chunks, len(todo), out_dir)

    failed = {}
    with cf.ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(str(cache_dir), cfg.revision)) as pool:
        futures = [pool.submit(_process_one, c, cfg, str(clips_dir)) for c in todo]
        for n, fut in enumerate(cf.as_completed(futures), 1):
            clip_id, k, secs, err = fut.result()
            if err:
                failed[clip_id] = err
                logger.warning("[%d/%d] %s FAILED: %s", n, len(todo), clip_id, err)
            else:
                logger.info("[%d/%d] %s  %d frames  %.1fs", n, len(todo), clip_id, k, secs)

    done = [c for c in clip_ids if (clips_dir / c).is_dir()]
    meta = {
        "config": dataclasses.asdict(cfg), "config_hash": cfg.hash(), "chunks": chunks, "wasd": WASD,
        "splits": {s: [c for c in done if split_of(c, cfg.val_fraction) == s] for s in ("train", "val")},
        "failed": failed,
        "command": " ".join(sys.argv),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    logger.info("done: %d clips (%d train / %d val), %d failed", len(done),
                len(meta["splits"]["train"]), len(meta["splits"]["val"]), len(failed))
    return out_dir


# ---------------------------------------------------------------------------------------------------------
# Load: windows of T consecutive frames with their T actions
# ---------------------------------------------------------------------------------------------------------

def window_starts(num_frames: int, seq_len: int, stride: int, include_partial: bool = False) -> np.ndarray:
    """Return sequence starts, optionally including a final shorter sequence."""
    limit = num_frames if include_partial else max(num_frames - 1 - seq_len + 1, 0)
    return np.arange(0, limit, stride, dtype=np.int64)


class NVPhysicalAIWindows:
    """Map-style dataset of windows from a preprocessed directory.

    Item: {"frames": uint8 [T, 3, H, W], "actions": float32 [T, 4], "ego": float32 [T, 3]}, where
    actions[t] / ego[t] are the motion from frames[t] to frame t+1. For the trainer's next-frame objective
    use frames[:, :-1] + actions[:, :-1] -> frames[:, 1:]. Frames stay uint8 here (4x less worker -> main
    traffic); `prepare_batch` moves them to the device and scales to [0, 1].
    """

    def __init__(self, root: pathlib.Path | str, split: str = "train", seq_len: int = 32,
                 stride: int | None = None, include_partial: bool = False):
        self.root = pathlib.Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.seq_len = seq_len
        self.include_partial = include_partial
        stride = stride or (1 if split == "train" else seq_len)   # val: non-overlapping windows
        self.clip_ids = self.meta["splits"][split]
        self.clips = []                                    # (frames memmap, actions, ego), opened lazily
        self.index = []
        for ci, clip_id in enumerate(self.clip_ids):
            k = len(np.load(self.root / "clips" / clip_id / "timestamps.npy"))
            self.index.extend(
                (ci, int(s)) for s in window_starts(k, seq_len, stride, include_partial)
            )
        self.index = np.asarray(self.index, dtype=np.int64).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self.index)

    def _clip(self, ci: int):
        if not self.clips:                                 # per process: memmaps are opened after fork/spawn
            d = self.root / "clips"
            self.clips = [(np.load(d / c / "frames.npy", mmap_mode="r"), np.load(d / c / "actions.npy"),
                           np.load(d / c / "ego.npy")) for c in self.clip_ids]
        return self.clips[ci]

    def __getitem__(self, i: int) -> dict:
        import torch

        ci, s = self.index[i]
        frames, actions, ego = self._clip(ci)
        T = self.seq_len
        return {
            "frames": torch.from_numpy(np.ascontiguousarray(frames[s:s + T].transpose(0, 3, 1, 2))),
            "actions": torch.from_numpy(actions[s:s + T]),
            "ego": torch.from_numpy(ego[s:s + T]),
        }

    def __getstate__(self):                                # don't ship open memmaps to DataLoader workers
        state = self.__dict__.copy()
        state["clips"] = []
        return state


def make_loader(root, split: str = "train", batch_size: int = 8, seq_len: int = 32, num_workers: int = 4,
                stride: int | None = None, seed: int = 0):
    import torch
    from torch.utils.data import DataLoader

    dataset = NVPhysicalAIWindows(root, split, seq_len, stride)
    train = split == "train"
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=num_workers,
        persistent_workers=num_workers > 0, generator=torch.Generator().manual_seed(seed),
    )


def prepare_batch(batch: dict, device) -> tuple:
    """-> frames float32 [B, T, 3, H, W] in [0, 1], actions float32 [B, T, 4], ego float32 [B, T, 3]."""
    frames = batch["frames"].to(device, non_blocking=True).float().div_(255.0)
    return frames, batch["actions"].to(device, non_blocking=True), batch["ego"].to(device, non_blocking=True)


# ---------------------------------------------------------------------------------------------------------
# Inspect: the statistics that matter for training on this data
# ---------------------------------------------------------------------------------------------------------

def inspect(root: pathlib.Path) -> dict:
    meta = json.loads((root / "meta.json").read_text())
    report = {"config_hash": meta["config_hash"], "failed": len(meta["failed"])}
    for split, clip_ids in meta["splits"].items():
        acts = np.concatenate([np.load(root / "clips" / c / "actions.npy") for c in clip_ids])
        ego = np.concatenate([np.load(root / "clips" / c / "ego.npy") for c in clip_ids])
        combos = (acts @ (1 << np.arange(4))).astype(int)
        report[split] = {
            "clips": len(clip_ids), "steps": len(acts),
            "key_rate": dict(zip(WASD, acts.mean(0).round(4).tolist())),
            "combo_rate": {"".join(k for b, k in enumerate(WASD) if c >> b & 1) or "-": round(float(n / len(acts)), 4)
                           for c, n in zip(*np.unique(combos, return_counts=True))},
            "speed_mps": np.percentile(ego[:, 0], [1, 50, 99]).round(2).tolist(),
            "yaw_rate_rps": np.percentile(ego[:, 2], [1, 50, 99]).round(3).tolist(),
            "windows_T32": len(NVPhysicalAIWindows(root, split, 32)),
        }
    return report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("download", "preprocess"):
        s = sub.add_parser(name)
        s.add_argument("--chunks", type=int, nargs="+", default=[0])
        s.add_argument("--cache-dir", type=pathlib.Path, default=DATA_ROOT / "hf_cache")
    pre = sub.choices["preprocess"]
    pre.add_argument("--workers", type=int, default=8)
    pre.add_argument("--fps", type=float, default=10.0)
    pre.add_argument("--size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    pre.add_argument("--limit", type=int, default=None)
    pre.add_argument("--out", type=pathlib.Path, default=None)
    ins = sub.add_parser("inspect")
    ins.add_argument("root", type=pathlib.Path, nargs="?", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    if args.cmd == "download":
        download(args.chunks, cache_dir=args.cache_dir)
    elif args.cmd == "preprocess":
        cfg = PreprocessConfig(fps=args.fps, size=tuple(args.size))
        preprocess(args.chunks, cfg, args.out, args.cache_dir, args.workers, args.limit)
    else:
        root = args.root or DATA_ROOT / "processed" / PreprocessConfig().hash()
        print(json.dumps(inspect(root), indent=1))


if __name__ == "__main__":
    main()
