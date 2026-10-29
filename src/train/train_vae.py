import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision.models import VGG16_Weights, vgg16

sys.path.append("/Users/akshaylakkur/Research/tesseraction/src")
from model.vae import CompressionDecoder, CompressionEncoder
from dataset.dataset import make_loader, prepare_batch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("...")
CHECKPOINT_DIR = Path("...")

BATCH_SIZE = 4
SEQ_LEN = 32            # window length the dataset returns
FRAMES_PER_STEP = 8     # frames sampled from each window per step (saves memory)
VGG_FRAMES = 8          # frames per step that get the perceptual loss

EPOCHS = 5
LR = 3e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
GRAD_CLIP = 1.0

MSE_WEIGHT = 1.0
PERCEPTUAL_WEIGHT = 1.0
LATENT_WEIGHT = 0.1

FRAME_RANGE = "unit"    # "unit" if prepare_batch returns [0, 1], "signed" if [-1, 1]

OVERFIT_ONE_BATCH = False   # set True first, to sanity-check the pipeline
OVERFIT_STEPS = 300
LOG_EVERY = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_unit(x):
    """Map frames to [0, 1], the range VGG's ImageNet normalization expects."""
    return x if FRAME_RANGE == "unit" else (x + 1) / 2


def check_frame_range(frames):
    lo, hi = frames.min().item(), frames.max().item()
    print(f"frame value range: [{lo:.3f}, {hi:.3f}]  (FRAME_RANGE={FRAME_RANGE!r})")
    expected = (0.0, 1.0) if FRAME_RANGE == "unit" else (-1.0, 1.0)
    if lo < expected[0] - 0.05 or hi > expected[1] + 0.05:
        print(f"WARNING: frames fall outside {expected}; check FRAME_RANGE / prepare_batch")


def sample_frames(frames, k, random=True):
    """Pick k frames from each window. frames: (B, T, C, H, W)."""
    T = frames.shape[1]
    if k >= T:
        return frames
    if random:
        idx = torch.randperm(T)[:k]
    else:
        idx = torch.linspace(0, T - 1, k).long()     # evenly spaced, deterministic
    return frames[:, idx.to(frames.device)]


def unit_norm(x, eps=1e-8):
    """Normalize each location's feature vector to length 1 (over channels)."""
    return x / torch.sqrt(x.square().sum(dim=1, keepdim=True) + eps)


def latent_stats_loss(z):
    """Push the latent toward mean 0, variance 1 (keeps its scale stable)."""
    return z.mean().square() + (z.var(unbiased=False) - 1.0).square()


def add_into(totals, parts):
    for key, value in parts.items():
        totals[key] = totals.get(key, 0.0) + value


def fmt(parts):
    return " ".join(f"{key}={value:.5f}" for key, value in parts.items())


# ---------------------------------------------------------------------------
# Perceptual loss
# ---------------------------------------------------------------------------
class VGGPerceptual(nn.Module):
    """Multi-layer VGG16 feature loss.

    Default layers are relu1_2, relu2_2, relu3_3, relu4_3 in
    torchvision's vgg16().features. Inputs must be in [0, 1].
    """

    def __init__(self, layer_ids=(3, 8, 15, 22)):
        super().__init__()
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[: max(layer_ids) + 1].eval()
        self.vgg.requires_grad_(False)
        self.layer_ids = set(layer_ids)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def features(self, x):
        x = (x - self.mean) / self.std
        feats = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_ids:
                feats.append(unit_norm(x))
        return feats

    def forward(self, x, x_out):
        with torch.no_grad():                 # real frames don't need gradients
            target = self.features(x)
        pred = self.features(x_out)
        return sum(F.mse_loss(p, t) for p, t in zip(pred, target)) / len(pred)


# ---------------------------------------------------------------------------
# Losses, schedule, steps
# ---------------------------------------------------------------------------
def compute_losses(encoder, decoder, vgg_loss, frames):
    """frames: (B, T, C, H, W). Returns (total loss tensor, dict of floats)."""
    z = encoder(frames)
    recon = decoder(z)

    mse = F.mse_loss(recon, frames)
    latent = latent_stats_loss(z)

    flat_real = to_unit(frames.flatten(0, 1))
    flat_recon = to_unit(recon.flatten(0, 1))
    n = min(VGG_FRAMES, flat_real.shape[0])
    idx = torch.randperm(flat_real.shape[0])[:n].to(flat_real.device)
    perceptual = vgg_loss(flat_real[idx], flat_recon[idx])

    total = MSE_WEIGHT * mse + PERCEPTUAL_WEIGHT * perceptual + LATENT_WEIGHT * latent
    parts = {
        "total": total.item(),
        "mse": mse.item(),
        "perc": perceptual.item(),
        "latent": latent.item(),
    }
    return total, parts


def make_scheduler(optimizer, total_steps):
    """Linear warmup, then cosine decay from LR to MIN_LR."""
    warmup = min(WARMUP_STEPS, max(1, total_steps // 10))
    floor = MIN_LR / LR

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(encoder, decoder, vgg_loss, optimizer, scheduler, params, frames):
    optimizer.zero_grad(set_to_none=True)
    loss, parts = compute_losses(encoder, decoder, vgg_loss, frames)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
    optimizer.step()
    scheduler.step()
    return parts


@torch.no_grad()
def validate(encoder, decoder, vgg_loss, val_loader, device):
    encoder.eval()
    decoder.eval()
    totals, count = {}, 0
    for batch in val_loader:
        frames, _, _ = prepare_batch(batch, device)
        frames = sample_frames(frames, FRAMES_PER_STEP, random=False)
        _, parts = compute_losses(encoder, decoder, vgg_loss, frames)
        add_into(totals, parts)
        count += 1
    encoder.train()
    decoder.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(path, encoder, decoder, optimizer, scheduler, epoch, val_parts):
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "val": val_parts,
        },
        path,
    )


def overfit_one_batch(encoder, decoder, vgg_loss, optimizer, params, train_loader, device):
    """Train on one fixed batch. MSE should fall steadily toward zero."""
    frames, _, _ = prepare_batch(next(iter(train_loader)), device)
    check_frame_range(frames)
    frames = sample_frames(frames, FRAMES_PER_STEP, random=False)
    scheduler = make_scheduler(optimizer, OVERFIT_STEPS)

    for step in range(1, OVERFIT_STEPS + 1):
        parts = train_step(encoder, decoder, vgg_loss, optimizer, scheduler, params, frames)
        if step == 1 or step % LOG_EVERY == 0:
            print(f"overfit step {step}/{OVERFIT_STEPS} {fmt(parts)}")
    print("If mse is not close to zero by now, something in the pipeline is broken.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    device = select_device()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    encoder = CompressionEncoder().to(device)
    decoder = CompressionDecoder().to(device)
    vgg_loss = VGGPerceptual().to(device)

    train_loader = make_loader(
        DATASET_ROOT, "train", batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_workers=0
    )
    val_loader = make_loader(
        DATASET_ROOT, "val", batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_workers=0
    )

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"device: {device}")
    print(f"train batches per epoch: {len(train_loader)}")
    print(f"validation batches: {len(val_loader)}")
    print(f"encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    if OVERFIT_ONE_BATCH:
        overfit_one_batch(encoder, decoder, vgg_loss, optimizer, params, train_loader, device)
        return

    scheduler = make_scheduler(optimizer, EPOCHS * len(train_loader))
    best_val = float("inf")
    checked_range = False

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        decoder.train()
        totals, count = {}, 0

        for batch_index, batch in enumerate(train_loader, start=1):
            frames, _, _ = prepare_batch(batch, device)
            if not checked_range:
                check_frame_range(frames)
                checked_range = True
            frames = sample_frames(frames, FRAMES_PER_STEP)

            parts = train_step(encoder, decoder, vgg_loss, optimizer, scheduler, params, frames)
            add_into(totals, parts)
            count += 1

            if batch_index % LOG_EVERY == 0 or batch_index == len(train_loader):
                print(
                    f"epoch {epoch} batch {batch_index}/{len(train_loader)} "
                    f"{fmt(parts)} lr={scheduler.get_last_lr()[0]:.2e}"
                )

        train_avg = {key: value / max(count, 1) for key, value in totals.items()}
        val_avg = validate(encoder, decoder, vgg_loss, val_loader, device)
        print(f"epoch {epoch}/{EPOCHS} train: {fmt(train_avg)}")
        print(f"epoch {epoch}/{EPOCHS} val:   {fmt(val_avg)}")

        save_checkpoint(CHECKPOINT_DIR / "last.pt",
                        encoder, decoder, optimizer, scheduler, epoch, val_avg)
        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            save_checkpoint(CHECKPOINT_DIR / "best.pt",
                            encoder, decoder, optimizer, scheduler, epoch, val_avg)
            print(f"new best val total: {best_val:.5f}")


if __name__ == "__main__":
    main()