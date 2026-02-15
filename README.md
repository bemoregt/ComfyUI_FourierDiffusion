# ComfyUI_FourierDiffusion

A ComfyUI custom node set for image generation using a **Fourier-domain diffusion model** — a denoising diffusion probabilistic model (DDPM) that operates directly on FFT frequency coefficients rather than in spatial pixel space.

---

## Overview

Standard diffusion models add and remove noise in the spatial domain. FourierDiffusion instead transforms feature maps into the frequency domain via `rfft2` at every residual block, processes real and imaginary components separately, then reconstructs via `irfft2`. This allows the model to:

- Treat low-frequency (structure/color) and high-frequency (texture/edge) information independently
- Apply frequency-weighted noise during sampling to bias toward certain spatial scales
- Learn spectral statistics of the training distribution

---

## Architecture

```
Input [B, C, H, W]
       │
       ▼
  InputProj (Conv2d)
       │
       ▼
  ┌─── Encoder ─────────────────────────────┐
  │  Level 0  →  FourierResBlock × N        │
  │              ↓ Downsample               │
  │  Level 1  →  FourierResBlock × N        │
  │              + SpectralAttention        │
  │              ↓ Downsample               │
  │  Level 2  →  FourierResBlock × N        │
  │              + SpectralAttention        │
  └─────────────────────────────────────────┘
       │  skip connections
       ▼
  ┌─── Bottleneck ──────────────────────────┐
  │  FourierResBlock → SpectralAttention    │
  │  → FourierResBlock                      │
  └─────────────────────────────────────────┘
       │
       ▼
  ┌─── Decoder (symmetric) ─────────────────┐
  │  FourierResBlock × N + skip concat      │
  │  ↑ Upsample                             │
  └─────────────────────────────────────────┘
       │
       ▼
  OutputProj (GroupNorm → SiLU → Conv2d)
       │
       ▼
  Predicted noise [B, C, H, W]
```

### FourierResBlock

Each residual block processes features in the frequency domain:

```
x  →  GroupNorm  →  rfft2  →  [real | imag]  →  Conv2d
                                                    │
                               timestep embedding ──┘
                                                    │
                              irfft2  ←  Conv2d  ←─┘
                                │
x ──────────────────────────── + (residual)
```

### SpectralAttention

Multi-head self-attention applied to `rfft2` tokens, preserving imaginary components via residual bypass.

---

## Nodes

All nodes appear under the **FourierDiffusion** category in ComfyUI.

### Fourier Diffusion Loader

Loads a FourierDiffusion checkpoint (`.pt` or `.safetensors`) and returns a model object ready for inference.

| Input | Type | Description |
|-------|------|-------------|
| `ckpt_name` | CHECKPOINT | Checkpoint file from the `checkpoints` folder |
| `model_channels` | INT | Base channel count (default: 64). Used if the checkpoint has no embedded config. |
| `channel_mults` | SELECT | Channel multiplier string per level (default: `1,2,4,8`) |
| `num_res_blocks` | INT | ResBlocks per encoder/decoder level (default: 2) |

| Output | Type |
|--------|------|
| `model` | `FOURIER_DIFFUSION_MODEL` |

> If the checkpoint was saved with `save_checkpoint()` from `utils.py`, the architecture config is embedded and the optional inputs are ignored.

---

### Fourier Diffusion Scheduler

Configures the noise schedule used during sampling and training.

| Input | Type | Description |
|-------|------|-------------|
| `timesteps` | INT | Total diffusion timesteps (default: 1000) |
| `schedule_type` | SELECT | `cosine` (recommended) or `linear` |
| `beta_start` | FLOAT | Linear schedule start β (default: 0.0001) |
| `beta_end` | FLOAT | Linear schedule end β (default: 0.02) |

| Output | Type |
|--------|------|
| `scheduler` | `FOURIER_SCHEDULER` |

> If no scheduler is connected to the Sampler, a default cosine schedule with T=1000 is used automatically.

---

### Fourier Diffusion Sampler

Generates images from a loaded model using DDIM (fast, deterministic) or DDPM (full timestep, stochastic) sampling.

| Input | Type | Description |
|-------|------|-------------|
| `model` | `FOURIER_DIFFUSION_MODEL` | From Fourier Diffusion Loader |
| `width` | INT | Output image width (default: 256) |
| `height` | INT | Output image height (default: 256) |
| `batch_size` | INT | Number of images to generate |
| `sampler` | SELECT | `ddim` or `ddpm` |
| `ddim_steps` | INT | Number of DDIM denoising steps (default: 50) |
| `eta` | FLOAT | DDIM stochasticity: 0.0 = deterministic, 1.0 = DDPM-equivalent |
| `seed` | INT | Random seed for reproducibility |
| `freq_weight` | FLOAT | High-frequency emphasis of initial noise (1.0 = uniform) |
| `scheduler` | `FOURIER_SCHEDULER` | Optional. Connects to Scheduler node. |

| Output | Type |
|--------|------|
| `image` | `IMAGE` |

---

### Fourier Diffusion Inspect (Spectrum)

Visualizes the 2D FFT amplitude and phase spectra of any image. Useful for debugging generated outputs or analyzing the frequency characteristics of your training data.

| Input | Type | Description |
|-------|------|-------------|
| `image` | `IMAGE` | Any ComfyUI image tensor |
| `log_scale` | BOOLEAN | Apply log1p to amplitude for better visibility |
| `colormap` | SELECT | Matplotlib colormap (`inferno`, `magma`, `viridis`, `hot`, `gray`, `jet`) |

| Output | Type |
|--------|------|
| `amplitude_spectrum` | `IMAGE` |
| `phase_spectrum` | `IMAGE` |

---

## Recommended Workflow

```
[Fourier Diffusion Loader]
        │ model
        ▼
[Fourier Diffusion Sampler] ──── [Preview Image]
        │ image
        ▼
[Fourier Diffusion Inspect]
   │ amplitude      │ phase
   ▼                ▼
[Preview Image] [Preview Image]
```

Optionally insert a **Fourier Diffusion Scheduler** node between Loader and Sampler to customize the noise schedule.

---

## Installation

### Automatic (symlink)

If you cloned this repo directly into or alongside your ComfyUI installation:

```bash
ln -s /path/to/ComfyUI_FourierDiffusion \
      /path/to/ComfyUI/custom_nodes/ComfyUI_FourierDiffusion
```

### Manual

Copy or clone the folder into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI_FourierDiffusion
```

Then restart ComfyUI. The four nodes will appear under the **FourierDiffusion** category.

---

## Training Your Own Model

Use the included `train_demo.py` script to train a FourierDiffusion model on your own image dataset.

```bash
# Train on a folder of images
python train_demo.py \
    --data_dir /path/to/your/images \
    --output_dir ./checkpoints \
    --image_size 256 \
    --model_channels 64 \
    --epochs 100 \
    --batch_size 8

# Quick pipeline test with random data (no images needed)
python train_demo.py --output_dir ./checkpoints
```

### Checkpoint Format

Checkpoints saved by `train_demo.py` embed the model config so the Loader node can reconstruct the architecture automatically:

```python
{
    "model_state_dict": { ... },
    "model_config": {
        "in_channels": 3,
        "model_channels": 64,
        "channel_mults": [1, 2, 4, 8],
        "num_res_blocks": 2,
        "attention_levels": [2, 3],
        "dropout": 0.1,
    },
    "epoch": 50,
    "step": 12500,
}
```

`.safetensors` files (e.g., from other tools) are also supported, but require manually matching the architecture parameters in the Loader node.

---

## Requirements

| Package | Version |
|---------|---------|
| `torch` | ≥ 2.0.0 |
| `numpy` | ≥ 1.24.0 |
| `Pillow` | ≥ 9.0.0 |
| `safetensors` | ≥ 0.3.0 |

All are available in a standard ComfyUI virtual environment.

---

## File Structure

```
ComfyUI_FourierDiffusion/
├── __init__.py        # ComfyUI entry point — exports NODE_CLASS_MAPPINGS
├── model.py           # FourierDiffusionUNet, FourierResBlock, SpectralAttention
├── diffusion.py       # FourierDiffusionScheduler, DDPM/DDIM sampling loops
├── nodes.py           # Four ComfyUI node classes
├── utils.py           # Checkpoint I/O, image preprocessing, train_step helper
├── train_demo.py      # Standalone training script
└── requirements.txt
```

---

## License

MIT
