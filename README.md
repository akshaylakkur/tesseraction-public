# Tesseraction

Tesseraction is a research prototype for a real-time, action-conditioned world
model. It learns to predict the next visual observation from a short history of
observations and keyboard-like driving actions:

```text
video + ego-motion -> processed clips -> latent frames -> action-conditioned rollout
```

The public repository contains the core model, compression, and training mechanisms. It is intentionally a small research snapshot.

## What is in this repository?

```text
src/
├── dataset/dataset.py     Download, preprocess, load, and inspect driving data
├── model/vae.py           Spatial frame encoder and decoder
├── model/model.py         Anna action-conditioned latent dynamics model
└── train/
    ├── train_vae.py       Train the visual compression model
    └── train.py           Train latent next-frame prediction
```

## Model mechanism

### 1. Dataset and actions

The data pipeline is designed around the NVIDIA PhysicalAI Autonomous
Vehicles dataset. For each clip it:

1. Downloads the selected camera, ego-motion, calibration, and vehicle
   features from a pinned dataset revision.
2. Samples the front-facing camera at a configurable rate (10 FPS by default).
3. Center-crops and resizes frames to `256 x 256`.
4. Computes relative motion between consecutive poses.
5. Converts motion into four binary controls in the fixed order
   `[W, A, S, D]`:
   - `W`: forward motion above a speed threshold
   - `A`: positive yaw rate
   - `S`: reverse motion above a speed threshold
   - `D`: negative yaw rate
6. Stores frames, timestamps, ego-motion, and actions per clip as NumPy files.

Preprocessing is resumable and writes a `meta.json` file containing the
preprocessing configuration, a configuration hash, train/validation splits,
failures, and the command used to create the cache. Splits are deterministic
and based on the clip identifier.

Training reads fixed-length windows (32 frames by default). A window contains
frames with the action or ego-motion occurring between adjacent frames. Frames
remain `uint8` in the worker pipeline and are converted to float tensors in
`[0, 1]` only when moved to the training device.

### 2. Visual compression

`CompressionEncoder` reduces each `256 x 256` RGB frame by a factor of eight,
producing an 8-channel `32 x 32` latent representation. The encoder combines:

- three stride-2 convolutional downsampling stages,
- spatial self-attention with 2D rotary positional features,
- a residual projection and group normalization.

`CompressionDecoder` reconstructs RGB frames from the latent grid using
convolutions, residual blocks, and bilinear upsampling. The encoder and decoder
are trained together before training the dynamics model.

The VAE training objective is a weighted combination of:

- pixel mean-squared reconstruction error,
- a VGG16 feature-space perceptual error,
- a latent mean/variance regularizer that stabilizes latent scale.

An optional one-batch overfit test is provided as a pipeline sanity check. A
healthy run should drive reconstruction error down on that fixed batch before
large-scale training is attempted.

### 3. Action-conditioned latent dynamics

`Anna` operates on latent sequences with shape:

```text
latent context: [batch, time, 8, 32, 32]
actions:        [batch, time, 4]
```

It predicts the next clean latent frame. Its four convolutional stages use:

- spatial attention over latent image locations,
- causal temporal attention over tiled spatial tokens,
- rotary positional features for spatial and temporal coordinates,
- action and noise-level modulation,
- convolutional residual processing.

The dynamics trainer uses diffusion forcing: the context latents are perturbed
with a randomly sampled noise scale, and the model learns to predict the clean
next latent conditioned on the action sequence and that scale. The encoder and
decoder are frozen during this stage. At validation time, the model is rolled
out autoregressively from one initial latent, decoded, and compared visually
with the real action-conditioned sequence.

## Setup

The snapshot currently depends on PyTorch, torchvision, NumPy, Pillow,
`huggingface_hub`, the PhysicalAI dataset interface, and related media
packages. Install the dependencies in your own environment, then expose the
source tree:

```bash
export PYTHONPATH="$PWD/src"
```

Before running the scripts, replace the `...` placeholders in:

- `src/dataset/dataset.py` (`DATA_ROOT` and the pinned dataset `REVISION`)
- `src/train/train_vae.py` (`DATASET_ROOT` and `CHECKPOINT_DIR`)
- `src/train/train.py` (`DEFAULT_ROOTS`, `vae_checkpoint`, and output paths)
- `src/model/model.py` (`DATASET_ROOT` and checkpoint paths, if using its demo)

If the dataset requires authentication, provide an `HF_TOKEN` environment
variable. The data loader also supports reading it from a repository-root
`.env`; do not commit that file or the token.

The training scripts select CUDA when available, then Apple Metal (MPS), then
CPU. For local Apple Silicon experiments, verify that the selected device is
MPS and benchmark the actual inference path rather than assuming every
operation is accelerated.

## Typical workflow

### 1. Download data

Download one or more dataset chunks into the configured cache:

```bash
python src/dataset/dataset.py download --chunks 0
```

### 2. Preprocess clips

Decode valid clips, generate actions, and write the resumable processed
dataset. Use `--limit` for a small smoke test:

```bash
python src/dataset/dataset.py preprocess \
  --chunks 0 \
  --workers 8 \
  --fps 10 \
  --size 256 256 \
  --limit 20
```

### 3. Inspect the processed data

The inspection report summarizes clip counts, action frequencies, action
combinations, speed and yaw-rate percentiles, and the number of 32-frame
training windows:

```bash
python src/dataset/dataset.py inspect /path/to/processed/config-hash
```

Use this report to catch class imbalance, implausible motion ranges, failed
clips, or an unexpectedly small validation split before training.

### 4. Train the visual compressor

Set the dataset and checkpoint paths in `train_vae.py`, then run:

```bash
python src/train/train_vae.py
```

The script reports device selection, parameter counts, training and validation
loss components, and writes `last.pt` and the best validation checkpoint. Start
with `OVERFIT_ONE_BATCH = True` to validate the input range and reconstruction
pipeline, then disable it for normal training.

### 5. Train the latent dynamics model

Point `TrainConfig.data_roots` at one or more processed dataset roots and
`TrainConfig.vae_checkpoint` at the best compression checkpoint:

```bash
python src/train/train.py
```

The trainer uses AdamW with cosine learning-rate decay, gradient clipping,
frozen compression networks, train/validation windows, periodic checkpoints,
and fixed validation rollouts. Saved dynamics checkpoints include model,
optimizer, scheduler, epoch/iteration, serialized configuration, and validation
loss.

## Analysis and evaluation

The main analysis loop is deliberately simple and inspectable:

1. **Data analysis:** run `dataset.py inspect` and review action balance and
   motion percentiles.
2. **Compression analysis:** compare reconstruction MSE, perceptual loss, and
   latent statistics on validation windows. Also inspect the fixed-batch
   overfit behavior.
3. **Dynamics analysis:** compare real and autoregressively predicted frames
   from the same initial frame and action sequence. The dynamics trainer saves
   periodic image grids with real frames on top and decoded rollout frames
   below.
4. **Stability analysis:** evaluate longer rollouts, action combinations, and
   held-out clips rather than relying only on one-step loss. Look for blur,
   drift, temporal discontinuities, and loss of controllability.
5. **Runtime analysis:** measure end-to-end frame time and memory on the target
   device. A low validation loss alone does not establish real-time behavior.

The repository does not yet include a formal benchmark harness, experiment
tracker, pretrained checkpoints, or a production inference wrapper. Treat
metrics and generated samples as experiment artifacts and record the dataset
configuration hash, model configuration, device, and command used to produce
them.

## Reproducibility and limitations

- Dataset preprocessing is tied to the configured dataset revision and
  preprocessing configuration hash.
- Random seeds are set in the training entry points, but exact results can
  still vary by device and backend.
- Paths and the dataset revision are placeholders in this public snapshot and
  must be configured locally.
- The current implementation predicts visual latent frames from recorded
  driving data; it does not yet provide a complete interactive application.
- Checkpoints may contain optimizer and scheduler state and can be large.
  Store them outside the source tree or in an artifact store.

## License and dataset

This repository contains research code. Consult the upstream NVIDIA PhysicalAI
dataset terms before downloading, redistributing, or publishing derived data.