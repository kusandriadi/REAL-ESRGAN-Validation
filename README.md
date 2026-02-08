# Real-ESRGAN (Inference & Evaluation)

Inference and benchmarking pipeline for Real-ESRGAN super-resolution models.
Fork of [ai-forever/Real-ESRGAN](https://github.com/ai-forever/Real-ESRGAN) (simplified Sberbank AI implementation), with custom evaluation tools.

> This is NOT the official [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
> This codebase is **inference-only** -- no training code, no discriminator, no GAN training pipeline.

- [Paper: Real-ESRGAN (Wang et al., 2021)](https://arxiv.org/abs/2107.10833)
- [Original implementation (xinntao)](https://github.com/xinntao/Real-ESRGAN)
- [HuggingFace weights](https://huggingface.co/sberbank-ai/Real-ESRGAN)

## What This Project Does

1. **Super-resolve** low-resolution images using pretrained RRDBNet generator (x2, x4, x8)
2. **Evaluate** image quality via PSNR against ground truth benchmarks (Set5, Set14, BSD100, Urban100)
3. **Monitor** system resources (CPU, RAM, GPU) during inference
4. **Log** all results to files for analysis

## Project Structure

```
Real-ESRGAN/
├── RealESRGAN/                        # Core inference package
│   ├── __init__.py                    # Exports: RealESRGAN class
│   ├── model.py                       # RealESRGAN class (load weights, predict)
│   ├── rrdbnet_arch.py                # Generator: RRDBNet (ResidualDenseBlock, RRDB)
│   ├── arch_utils.py                  # Utilities: init weights, pixel_unshuffle
│   ├── utils.py                       # Image split/stitch for patch-based inference
│   └── process/
│       └── upscale.py                 # Batch upscale + PSNR evaluation per dataset
│
├── metrics/                           # System & image quality metrics
│   ├── computation.py                 # System monitoring orchestrator
│   ├── cpu_and_ram.py                 # CPU/RAM via psutil
│   ├── gpu.py                         # GPU via GPUtil
│   └── psnr.py                        # PSNR via skimage (data_range=255)
│
├── weights/                           # Pretrained .pth model weights
│   ├── RealESRGAN_default_x2.pth      # x2 upscale (~16.7M params)
│   ├── RealESRGAN_default_x4.pth      # x4 upscale (~16.7M params)
│   └── RealESRGAN_default_x8.pth      # x8 upscale (~16.7M params)
│
├── inputs/                            # Test images
│   ├── gt/                            # Ground truth (HR) images
│   │   ├── BSD100/{2,3,4}/            # 100 images, 3 scale factors
│   │   ├── Set5/{2,3,4}/             # 5 images, 3 scale factors
│   │   ├── Set14/{2,3,4}/            # 14 images, 3 scale factors
│   │   └── Urban100/{2,4}/           # 100 images, 2 scale factors
│   ├── lr/                            # Low-resolution input images
│   │   ├── Set5/{2,3,4}/             # Active (no "1" prefix)
│   │   ├── 1BSD100/{2,3,4}/          # Disabled ("1" prefix = skipped)
│   │   ├── 1Set14/{2,3,4}/           # Disabled
│   │   └── 1Urban100/{2,4}/          # Disabled
│   ├── lr_image.png                   # Demo images
│   ├── lr_face.png
│   └── lr_lion.png
│
├── main.py                            # Simple inference (single scale, inputs/ folder)
├── main_computation.py                # Full benchmark (all weights, all datasets, metrics)
├── version.py                         # Print CUDA/PyTorch version info
├── setup.py                           # pip install package setup
├── requirements.txt                   # Dependencies (PyTorch 2.5.1 + CUDA 11.8)
└── LICENSE                            # BSD 3-Clause
```

### "1" Prefix Convention

Files/folders starting with `1` are **skipped** by `main_computation.py`:
- `1BSD100/` in `inputs/lr/` = dataset disabled (not processed)
- `1some_weight.pth` in `weights/` = weight file disabled

Remove the `1` prefix to enable a dataset or weight.

## Generator Architecture: RRDBNet

The only neural network in this codebase. Standard ESRGAN/Real-ESRGAN generator.

```
RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)

Pipeline:
  Input (3ch RGB) ──► [pixel_unshuffle if scale≤2]
    ──► conv_first (3→64)
    ──► 23x RRDB blocks (each = 3x ResidualDenseBlock, each = 5 dense Conv2d)
    ──► conv_body + global skip connection
    ──► Upsample (nearest interp x2 + Conv2d) × N  [N=2 for x4, N=3 for x8]
    ──► conv_hr + conv_last (64→3)
  Output (3ch RGB, upscaled)

Total parameters: ~16.7M
Activation: LeakyReLU(0.2)
Residual scaling: 0.2 (at both RDB and RRDB levels)
Weight init: Kaiming normal, scaled by 0.1
```

### ResidualDenseBlock (RDB)
- 5 Conv2d layers (3x3) with dense connections (each receives concatenated features from all previous)
- Growth channels: 32 per layer → input channels increase: 64, 96, 128, 160, 192
- Final conv maps back to 64 channels + residual * 0.2

### RRDB
- 3 sequential RDB blocks + residual * 0.2

## Inference Pipeline

### `model.predict(lr_image, batch_size=4, patches_size=192, padding=24, pad_size=15)`

```
PIL Image
  ──► numpy array
  ──► reflect-pad (15px all sides)
  ──► split into 192x192 overlapping patches (24px overlap)
  ──► normalize [0,1], to tensor, to GPU
  ──► forward pass in batches of 4 (FP16 autocast, torch.no_grad)
  ──► clamp [0,1]
  ──► stitch patches back together
  ──► unpad, convert to uint8 [0,255]
  ──► PIL Image (SR result)
```

### Weight Loading

Supports three state dict formats:
- `loadnet['params']` (standard Real-ESRGAN format)
- `loadnet['params_ema']` (EMA weights)
- Raw state dict (direct)

Can auto-download from HuggingFace Hub (`sberbank-ai/Real-ESRGAN`) for scales 2, 4, 8.

## Entry Points

### `main.py` -- Simple Inference

```python
# Hardcoded: scale=2, processes all images in inputs/
python main.py
# Output: results/
```

### `main_computation.py` -- Full Benchmark

```python
python main_computation.py
```

Loop logic:
1. Scan `weights/` for all `.pth` not starting with `1`
2. Extract scale from filename (`_x4` → scale=4)
3. For each weight: load model, start system monitoring
4. For each dataset in `inputs/lr/` not starting with `1`:
   - For each scale subfolder → for each image:
     - `model.predict()` → save result
     - Calculate PSNR vs ground truth from `inputs/gt/`
     - Collect CPU/RAM/GPU metrics
5. Write log: avg PSNR, CPU%, RAM%, GPU%, GPU Memory%, total time

Output: `output/{weight_name}/log_{weight_name}.log`

## Metrics

| Metric | Source | Details |
|--------|--------|---------|
| **PSNR** | `metrics/psnr.py` | `skimage.metrics.peak_signal_noise_ratio`, data_range=255 |
| **CPU %** | `metrics/cpu_and_ram.py` | `psutil.cpu_percent()` |
| **RAM %/MB** | `metrics/cpu_and_ram.py` | `psutil.virtual_memory()` |
| **GPU %** | `metrics/gpu.py` | `GPUtil.getGPUs()` |
| **GPU Memory %/MB** | `metrics/gpu.py` | `GPUtil.getGPUs()` |

Note: Only PSNR is computed as an image quality metric. SSIM, NIQE, LPIPS, PI are NOT implemented here.

## Dependencies

```
torch==2.5.1+cu118          # PyTorch with CUDA 11.8
torchvision==0.20.1+cu118
torchaudio==2.5.1+cu118
numpy, opencv-python, Pillow
tqdm, huggingface-hub
psutil, GPUtil              # System monitoring (imported in metrics/)
skimage                     # PSNR calculation
line_profiler, memory_profiler  # Profiling tools
tensorflow, tensorflow_datasets # Not actively used in main pipeline
```

Install: `pip install -r requirements.txt`

## What is NOT in This Project

This is an inference-only fork. The following exist in the **official xinntao/Real-ESRGAN** but are absent here:

| Component | Present? | Notes |
|-----------|:--------:|-------|
| Generator (RRDBNet) | YES | Identical to official |
| Discriminator architectures | NO | No UNetDiscriminatorSN, VGGStyleDiscriminator |
| GAN training model | NO | No training loop, optimizer, scheduler |
| Loss functions | NO | No L1, perceptual, GAN loss |
| YAML config system | NO | No `options/` directory |
| Training script | NO | No `train.py` |
| Dataset/DataLoader | NO | No DIV2K loader, no degradation pipeline |
| basicsr framework | NO | Standalone, no basicsr dependency |
| Face restoration | NO | No GFPGAN/CodeFormer |
| Video processing | NO | Image-only |

## Relationship to DCS Research

This project is the **inference/evaluation tool** for the DCS research at BINUS University:
- **Research project**: `C:\Users\kusan\Documents\kampus\dcs-research\`
- **Training code** (discriminator architectures, experiment configs): in `dcs-research/data/code/` and `dcs-research/data/experiment/`
- The `.pth` weight files here are the **pretrained default** Sberbank AI weights, NOT the custom-trained experiment weights from the DCS research
- The DCS research trains custom discriminators (UNetDiscriminatorSN, ConvNeXtDiscriminator, etc.) using a different codebase (official xinntao Real-ESRGAN with basicsr), then evaluates the resulting generator weights

## Custom Modifications (vs ai-forever fork)

1. **`main_computation.py`** -- Batch evaluation pipeline with multi-weight, multi-dataset loops + system monitoring
2. **`metrics/` module** -- CPU/RAM/GPU monitoring + PSNR calculation (entirely custom)
3. **`RealESRGAN/process/upscale.py`** -- Batch upscaling with GT comparison (custom)
4. **`version.py`** -- Environment info printer (custom)
5. **`requirements.txt`** -- Pinned to CUDA 11.8, added profiling tools
6. **`model.py`** -- Added `weights_only=True` to `torch.load()` (security fix)
7. **Benchmark datasets** -- `inputs/gt/` and `inputs/lr/` populated with Set5, Set14, BSD100, Urban100
