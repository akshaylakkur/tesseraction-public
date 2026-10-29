"""Train Anna with diffusion forcing directly from the five processed chunks."""
from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader

from model.model import Anna
from model.vae import CompressionDecoder, CompressionEncoder
from dataset.dataset import NVPhysicalAIWindows, prepare_batch

LOGGER = logging.getLogger(__name__)

DEFAULT_ROOTS = (Path("..."))


@dataclass(frozen=True)
class TrainConfig:
    data_roots: tuple[Path, ...] = DEFAULT_ROOTS
    vae_checkpoint: Path = REPO_ROOT / "..."
    output_dir: Path = REPO_ROOT / "..."
    seq_len: int = 32
    batch_size: int = 4
    epochs: int = 5
    learning_rate: float = 1e-4
    min_sigma: float = 0.01
    max_sigma: float = 1.0
    num_workers: int = 0
    seed: int = 0
    grad_clip: float = 1.0
    checkpoint_interval: int = 500
    sample_interval: int = 50
    sample_frames: int = 8
    sample_dir: Path = REPO_ROOT / "..."


def device_for_training() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_encoder(path: Path, device: torch.device) -> CompressionEncoder:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    encoder = CompressionEncoder().to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    return encoder.eval().requires_grad_(False)


def load_decoder(path: Path, device: torch.device) -> CompressionDecoder:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    decoder = CompressionDecoder().to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    return decoder.eval().requires_grad_(False)


def sample_sigma(batch_size, steps, config, device):
    low = torch.log(torch.tensor(config.min_sigma, device=device))
    high = torch.log(torch.tensor(config.max_sigma, device=device))
    return torch.rand(batch_size, steps, device=device).mul(high - low).add(low).exp()


def diffusion_forcing_loss(model, latents, actions, config):
    """Predict clean z[t+1] from noisy z[t], action[t], and sigma[t]."""
    if latents.ndim != 5 or actions.ndim != 3:
        raise ValueError("expected latents [B,T+1,C,H,W] and actions [B,T,4]")
    if latents.shape[1] != actions.shape[1] + 1:
        raise ValueError("latents must contain exactly one more frame than actions")
    context, target = latents[:, :-1], latents[:, 1:]
    sigma = sample_sigma(context.shape[0], context.shape[1], config, context.device)
    noisy_context = context + sigma[..., None, None, None] * torch.randn_like(context)
    prediction = model(noisy_context, actions, sigma)
    loss = F.mse_loss(prediction, target)
    return loss


def make_dataset(roots, split, seq_len):
    datasets = [NVPhysicalAIWindows(root, split=split, seq_len=seq_len) for root in roots]
    return ConcatDataset(datasets)


def autoregressive_prediction(model, initial_latent, actions):
    """Roll one latent forward using only prior predictions and fixed actions."""
    history = [initial_latent]
    for step in range(actions.shape[1]):
        context = torch.cat(history, dim=1)
        step_action = actions[:, step:step + 1]
        sigma = torch.zeros(
            actions.shape[0], context.shape[1], device=context.device, dtype=context.dtype
        )
        prediction = model(context, step_action if context.shape[1] == 1 else actions[:, :step + 1], sigma)
        history.append(prediction[:, -1:])
    return torch.cat(history, dim=1)


@torch.no_grad()
def save_validation_sample(model, encoder, decoder, validation_batch, device, path, sample_frames):
    """Save real/predicted frames for one fixed validation action sequence."""
    if device.type == "mps":
        torch.mps.synchronize()
    frames = actions = latents = predicted_latents = predicted_frames = None
    frames, actions, _ = prepare_batch(validation_batch, device)
    frames = frames[:1, :sample_frames]
    actions = actions[:1, :frames.shape[1] - 1]
    try:
        latents = encoder(frames)
        predicted_latents = autoregressive_prediction(model, latents[:, :1], actions)
        predicted_frames = decoder(predicted_latents)
        frame_count = min(frames.shape[1], predicted_frames.shape[1])
        columns = min(8, frame_count)
        indices = torch.linspace(0, frame_count - 1, columns, device=device).long()
        real = frames[0, indices].cpu().permute(0, 2, 3, 1).numpy()
        predicted = predicted_frames[0, indices].clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

        figure, axes = plt.subplots(2, columns, figsize=(2.2 * columns, 4.8))
        if columns == 1:
            axes = axes[:, None]
        for column, (real_frame, predicted_frame, index) in enumerate(zip(real, predicted, indices.tolist())):
            axes[0, column].imshow(real_frame)
            axes[0, column].set_title(f"real t={index}")
            axes[1, column].imshow(predicted_frame)
            axes[1, column].set_title(f"pred t={index}")
            axes[0, column].axis("off")
            axes[1, column].axis("off")
        axes[0, 0].set_ylabel("Validation\nreal")
        axes[1, 0].set_ylabel("Anna\nrollout")
        figure.suptitle("Fixed validation action sequence")
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=120)
        plt.close(figure)
    finally:
        del frames, actions, latents, predicted_latents, predicted_frames
        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()


def save_checkpoint(path, model, optimizer, scheduler, epoch, iteration, config, val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "iteration": iteration,
        "config": {k: [str(x) for x in v] if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v
                   for k, v in asdict(config).items()},
        "val_loss": val_loss,
    }, path)


def train(config: TrainConfig):
    torch.manual_seed(config.seed)
    device = device_for_training()
    encoder = load_encoder(config.vae_checkpoint, device)
    decoder = load_decoder(config.vae_checkpoint, device)
    train_loader = DataLoader(
        make_dataset(config.data_roots, "train", config.seq_len),
        batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        make_dataset(config.data_roots, "val", config.seq_len),
        batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=config.num_workers,
    )
    validation_batch = next(iter(val_loader))
    model = Anna().to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.epochs * len(train_loader)))
    best = float("inf")
    iteration = 0

    def checkpoint(step):
        save_checkpoint(config.output_dir / f"iter_{step:08d}.pt", model, optimizer, scheduler,
                        epoch, step, config, float("nan"))

    for epoch in range(config.epochs):
        model.train()
        train_total = 0.0
        for batch in train_loader:
            frames, actions, _ = prepare_batch(batch, device)
            with torch.no_grad():
                latents = encoder(frames)
            loss = diffusion_forcing_loss(model, latents, actions[:, :-1], config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            iteration += 1
            train_total += loss.item()
            LOGGER.info(
                "iteration=%d/%d epoch=%d/%d loss=%.6f lr=%.8f",
                iteration,
                config.epochs * len(train_loader),
                epoch + 1,
                config.epochs,
                loss.item(),
                scheduler.get_last_lr()[0],
            )
            if iteration % config.checkpoint_interval == 0:
                checkpoint(iteration)
            if iteration % config.sample_interval == 0:
                model.eval()
                save_validation_sample(
                    model,
                    encoder,
                    decoder,
                    validation_batch,
                    device,
                    config.sample_dir / f"iteration_{iteration:08d}.png",
                    config.sample_frames,
                )
                model.train()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                frames, actions, _ = prepare_batch(batch, device)
                latents = encoder(frames)
                val_total += diffusion_forcing_loss(model, latents, actions[:, :-1], config).item()
        train_loss = train_total / max(1, len(train_loader))
        val_loss = val_total / max(1, len(val_loader))
        LOGGER.info("epoch=%d train_loss=%.6f val_loss=%.6f", epoch + 1, train_loss, val_loss)
        save_checkpoint(config.output_dir / "last.pt", model, optimizer, scheduler,
                        epoch, iteration, config, val_loss)
        if val_loss < best:
            best = val_loss
            save_checkpoint(config.output_dir / "best.pt", model, optimizer, scheduler,
                            epoch, iteration, config, val_loss)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(TrainConfig())


if __name__ == "__main__":
    main()
