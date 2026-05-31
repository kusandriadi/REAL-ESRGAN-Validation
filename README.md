# REAL-ESRGAN-Validation (Inference & Evaluation)

Inference and benchmarking pipeline for Real-ESRGAN super-resolution models.
Fork of [ai-forever/Real-ESRGAN](https://github.com/ai-forever/Real-ESRGAN) (simplified Sberbank AI implementation), with custom evaluation tools used to validate the trained generators from the DCS research.

> This is NOT the official [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
> This codebase is **inference-only** — no training code, no discriminator, no GAN training pipeline. It loads a trained generator (`.pth`), upscales images, and measures quality.

- [Paper: Real-ESRGAN (Wang et al., 2021)](https://arxiv.org/abs/2107.10833)
- [Original implementation (xinntao)](https://github.com/xinntao/Real-ESRGAN)
- [HuggingFace weights](https://huggingface.co/sberbank-ai/Real-ESRGAN)

---

## Contents

1. [How to run (TL;DR)](#tldr-how-to-run)
2. [Where the `.pth` weights go + naming format](#1-where-the-pth-weights-go-and-naming-format)
3. [Inputs — `lr` / `gt` layout](#2-inputs-what-they-are-and-how-to-name-them)
4. [Outputs — structure & log naming](#3-outputs-structure-and-naming)
5. [The four scripts & how they differ](#4-the-four-scripts-what-each-does-and-how-they-differ)
   - [`basic` vs `custom`](#basic-vs-custom-the-real-difference)
6. [What this project does](#what-this-project-does)
7. [Project structure](#project-structure)
8. [Generator architecture: RRDBNet](#generator-architecture-rrdbnet)
9. [Metrics](#metrics)
10. [Dependencies](#dependencies)
11. [Example: evaluate a custom weight](#example-evaluate-a-custom-trained-weight)
12. [What is NOT included](#what-is-not-in-this-project)
13. [Relationship to the DCS research](#relationship-to-the-dcs-research)

---

## TL;DR: How to Run

All paths in the scripts are **relative**, so you must run from the project root.

```powershell
cd C:\Users\kusan\Documents\code\self\s3\REAL-ESRGAN-Validation
.\.venv\Scripts\Activate.ps1          # activate the virtual env
# or: pip install -r requirements.txt  (needs CUDA 11.8)

python evaluate_basic.py              # full benchmark  -> "basic"  results
python evaluate_custom.py             # full benchmark  -> "custom" results (richer stats)
python simple_upscale.py              # one image + 5 metrics (edit paths inside first)
python main.py                        # minimal upscale demo, no metrics
```

What the batch scripts need on disk:

| Folder | Purpose |
|---|---|
| `weights/*.pth` | trained generator weights to evaluate |
| `inputs/lr/<Dataset>/<scale>/*.png` | low-resolution **input** images |
| `inputs/gt/<Dataset>/<scale>/*.png` | high-resolution **ground truth** (required for metrics) |
| `output/` | created automatically; results + logs land here |

---

## 1. Where the `.pth` weights go and naming format

Put every weight file in `weights/`.

```
weights/
├── RealESRGAN_29_5-DEFAULT_x4.pth     ← will be processed (no "1" prefix)
├── 1RealESRGAN_33_ConvNext_..._x4.pth ← SKIPPED ("1" prefix = disabled)
└── MyModel_x4.pth                     ← any name works if it ends _x4 / _x2 / _x8
```

**Naming rules (enforced by the batch scripts):**

- Must end with `.pth`.
- Must contain the scale token `_x2`, `_x4`, or `_x8` — the scale is parsed from the filename via regex `_x(\d+)`. A file without it is skipped with a warning.
- **A filename starting with `1` is skipped.** This is the "disable / archive" switch — rename to add/remove the leading `1` to turn a weight off/on without deleting it.
- Valid: `experiment_33_x4.pth`, `ConvNeXt_model_x4.pth`, `RealESRGAN_default_x2.pth`
- Invalid: `model.pth` (no scale), `model_4x.pth` (wrong format → skipped).

> **Current state of this repo:** `weights/` holds 23 files, but **22 are prefixed `1`** (archived) and only **`RealESRGAN_29_5-DEFAULT_x4.pth`** is active. So a batch run right now evaluates just that one model. Remove the `1` prefix from any file you also want evaluated. (Weight `.pth` files are git-ignored — they live on disk only, not in the repo.)

### `.pth` content / format

A weight file is a PyTorch state dict for the **RRDBNet** generator. Three layouts are accepted automatically (`model.py`):

| Layout | How it was saved |
|---|---|
| Raw state dict | `torch.save(model.state_dict(), ...)` |
| `{'params': ...}` | official xinntao Real-ESRGAN format |
| `{'params_ema': ...}` | EMA weights from training |

The architecture (`num_feat`, `num_block`, `num_grow_ch`) is **auto-detected from the weight shapes** at load time, so non-standard generators load too — you do **not** have to hand-match `num_block=23`. Loading is `strict=True` after detection, so a genuinely incompatible file still fails loudly.

If a requested weight file is missing **and** `download=True`, the default Sberbank weights for scale 2/4/8 are pulled from HuggingFace Hub (`sberbank-ai/Real-ESRGAN`). The batch scripts call with `download=False`; only `main.py` uses `download=True`.

---

## 2. Inputs: what they are and how to name them

Two parallel trees under `inputs/`, one for the LR input and one for the HR ground truth:

```
inputs/
├── lr/                      ← low-resolution INPUT images
│   └── <Dataset>/           ← dataset name, e.g. Set5, BSD100
│       └── <scale>/         ← scale factor as a plain number: 2, 3, or 4
│           ├── img_001.png
│           └── ...
└── gt/                      ← ground-truth / high-res REFERENCE images
    └── <Dataset>/           ← MUST match the name under lr/
        └── <scale>/         ← MUST match the scale folder under lr/
            ├── img_001.png  ← MUST match the filename under lr/
            └── ...
```

**Critical matching rules:**

- Dataset folder names in `lr/` and `gt/` must be **identical** (case-sensitive).
- The scale subfolder is a **plain number** (`2`, `3`, `4`) — *not* `2x`.
- Each LR image must have a **same-named** GT image at the same dataset/scale path. Metrics for an image are skipped (basic) / that image is dropped (custom) if its GT is missing.
- Accepted formats: `.png`, `.jpg`, `.jpeg`.
- A dataset folder under `lr/` starting with `1` is **skipped** (same disable convention as weights).

Currently active datasets: **BSD100, Set5, Set14, Urban100** (each with `2/ 3/ 4/`, except Urban100 which has `2/ 4/`).

**Add a new dataset (e.g. Manga109, x4):**
```powershell
New-Item -ItemType Directory -Force inputs\lr\Manga109\4
New-Item -ItemType Directory -Force inputs\gt\Manga109\4
# copy LR images into inputs\lr\Manga109\4\  and matching HR into inputs\gt\Manga109\4\
```

**Disable a dataset without deleting it:** rename with a leading `1` (e.g. `Set5` → `1Set5`); rename back to re-enable.

---

## 3. Outputs: structure and naming

### Batch scripts (`evaluate_basic.py` and `evaluate_custom.py`)

Results mirror the input tree under `output/`, one folder per weight:

```
output/
└── <weight_name>/                         ← weight filename without ".pth"
    └── <Dataset>/
        └── <scale>/
            ├── img_001.png                ← super-resolved result (same filename as input)
            ├── img_002.png
            └── log_<weight_name>_<Dataset>_<scale>.log   ← metrics + system usage
```

- `<weight_name>` = the `.pth` filename minus extension (e.g. `RealESRGAN_29_5-DEFAULT_x4`).
- One **log per scale folder**, named `log_<weight>_<dataset>_<scale>.log`, written *inside* that scale folder.

### Demo scripts

| Script | Output |
|---|---|
| `main.py` | `results/<image>_result_<weight>_<scale>.png` (no metrics) |
| `simple_upscale.py` | `output/simple_upscale/<name>_upscaled.png` + a `..._metrics.txt` next to it |

### Log file contents

**basic** (`evaluate_basic.py` → `RealESRGAN/process/upscale.py`) — averages only, Indonesian system labels:
```
Model Parameters:
Total parameters: 16,697,987
Trainable parameters: 0
Model size (calculated): 63.70 MB
Weight file size: 127.85 MB

Image Quality Metrics:
PSNR (HB): 23.74
SSIM (HB): 0.6617
PI   (LB): 2.7450
NIQE (LB): 3.6848
LPIPS (LB): 0.2718

Rata-rata penggunaan sistem selama proses:
CPU: 2.73%
RAM: 73.74% (24046.24 MB)
GPU: 29.83%
Memori GPU: 95.00% (7778.58 MB)
Waktu total: 6 minutes 44 seconds
```

**custom** (`evaluate_custom.py`) — mean ± std with min/max, per-image timing, English labels:
```
Model Parameters:
... (same block) ...

Image Quality Metrics (Average):
Images processed: 100
PSNR (HB): 22.33 +/- 2.59 dB (min: 16.85, max: 28.84)
SSIM (HB): 0.5686 +/- 0.1273 (min: 0.2242, max: 0.8351)
PI (LB): 2.6261 +/- 0.7378 (min: 1.6727, max: 6.2234)
NIQE (LB): 3.3761 +/- 0.8383 (min: 2.3258, max: 8.2941)
LPIPS (LB): 0.3242 +/- 0.0812 (min: 0.1620, max: 0.6743)

System Performance (Average):
CPU: 5.59%
RAM: 76.83% (25053.29 MB)
GPU: 19.49%
GPU Memory: 95.93% (7855.00 MB)
Total processing time: 6 minutes 43 seconds
Average time per image: 4.04 seconds
```

---

## 4. The four scripts: what each does and how they differ

| Script | Scope | Metrics | Stats reported | Notes |
|---|---|---|---|---|
| **`evaluate_basic.py`** (`basic`) | Batch: every active weight × every active dataset | PSNR, SSIM, PI, NIQE, LPIPS | **mean only** | Delegates to `RealESRGAN/process/upscale.py`. Indonesian log labels. **These results feed the dissertation.** |
| **`evaluate_custom.py`** (`custom`) | Batch: same | same 5 | **mean ± std, min/max** + images processed + avg time/image | Self-contained (does not use `upscale.py`). English labels, per-model/per-dataset `try/except`, correct per-scale separation. |
| **`simple_upscale.py`** | **Single image** | same 5 (if GT given) | per-image values | Hardcoded example paths in `main()` — **edit them first** (the defaults point to an example that may not exist). Also exposes `upscale_and_evaluate()` to import. |
| **`main.py`** | Folder of images (flat `inputs/`) | none | — | Minimal legacy demo. Hardcoded `scale=2`, weight `RealESRGAN_x2.pth`, `download=True`. Saves to `results/`. Not part of the benchmark. |

### `basic` vs `custom`: the real difference

Both run the **same models on the same datasets** and both compute metrics **per image, then average** (they are *not* "aggregate vs per-image"). The differences are in *reporting and bookkeeping*:

- **basic** reports only the mean; **custom** reports mean ± std (min/max), image count, and average time per image.
- **custom** resets its accumulators per scale folder, so each scale log is scale-specific. **basic** accumulates one shared list across all scales of a dataset and writes that same cumulative average into every per-scale log of that dataset — a known quirk to be aware of when reading `basic` logs.
- Because mean-of-per-image differs slightly from how each pipeline buckets values, per-dataset numbers can differ a little (e.g. BSD100/4x: basic 23.74 vs custom 22.33), while the global averages across all datasets/scales converge (e.g. model 29_DEFAULT ≈ 23.54 in both).

> In the DCS research, the **`basic`** outputs are the ones reported in the dissertation and papers; `custom` is the secondary, more detailed pass.

---

## What This Project Does

1. **Super-resolve** low-resolution images with a pretrained RRDBNet generator (x2, x4, x8).
2. **Evaluate** quality against ground truth using **PSNR, SSIM, PI, NIQE, LPIPS** (all via [`pyiqa`](https://github.com/chaofengc/IQA-PyTorch)).
3. **Monitor** system resources (CPU, RAM, GPU, GPU memory) during inference.
4. **Log** per-dataset/per-scale results to files for later analysis.

## Project Structure

```
REAL-ESRGAN-Validation/
├── RealESRGAN/                    # Core inference package
│   ├── __init__.py                # Exports: RealESRGAN class
│   ├── model.py                   # RealESRGAN: load_weights (auto-detect arch), predict
│   ├── rrdbnet_arch.py            # Generator: RRDBNet (RRDB / ResidualDenseBlock)
│   ├── arch_utils.py              # make_layer, pixel_unshuffle, init helpers
│   ├── utils.py                   # patch split / stitch for tiled inference
│   └── process/
│       └── upscale.py             # "basic" batch upscale + metrics (used by evaluate_basic.py)
│
├── metrics/                       # Image-quality (pyiqa) + system monitoring
│   ├── psnr.py  ssim.py  niqe.py  lpips.py  perceptual_index.py   # pyiqa metrics
│   ├── computation.py             # monitor_system_metrics(), calculate_average_metrics()
│   ├── cpu_and_ram.py             # CPU / RAM sampling
│   └── gpu.py                     # GPU / GPU-memory sampling
│
├── weights/                       # *.pth generator weights (git-ignored; on disk only)
├── inputs/
│   ├── lr/<Dataset>/<scale>/*.png # low-res inputs
│   └── gt/<Dataset>/<scale>/*.png # ground truth (same names as lr/)
├── output/                        # generated results + logs (git-ignored)
│
├── evaluate_basic.py              # batch "basic"  (mean-only logs)
├── evaluate_custom.py             # batch "custom" (mean±std/min-max logs)
├── simple_upscale.py              # single-image upscale + metrics
├── main.py                        # minimal demo (no metrics)
├── version.py                     # prints CUDA / PyTorch / Python versions
├── setup.py  requirements.txt     # packaging + deps (PyTorch 2.5.1 + CUDA 11.8)
└── LICENSE                        # BSD 3-Clause
```

### "1" prefix convention

Anything starting with `1` is **skipped** by the batch scripts:
- `1SomeWeight_x4.pth` in `weights/` → weight disabled.
- `1BSD100/` in `inputs/lr/` → dataset disabled.

Rename to remove the `1` to enable it again.

## Generator Architecture: RRDBNet

The only neural network in this codebase — the standard ESRGAN/Real-ESRGAN generator.

```
RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)

  Input (3ch RGB) ──► [pixel_unshuffle if scale==2]
    ──► conv_first
    ──► 23× RRDB (each = 3× ResidualDenseBlock, each = 5 dense Conv2d)
    ──► conv_body + global skip
    ──► Upsample × N   (N=2 for x4, N=3 for x8)
    ──► conv_hr ──► conv_last
  Output (upscaled RGB)

~16.7M params · LeakyReLU(0.2) · residual scaling 0.2
```

Defaults above are the standard config; actual values are auto-detected per weight file at load.

### Inference pipeline — `model.predict(lr, batch_size=4, patches_size=192, padding=24, pad_size=15)`

```
PIL ─► numpy ─► reflect-pad 15px ─► split into 192×192 patches (24px overlap)
   ─► normalize [0,1] ─► batched forward (FP16 autocast, no_grad)
   ─► clamp [0,1] ─► stitch patches ─► unpad ─► uint8 ─► PIL (SR image)
```

## Metrics

| Metric | Backend | Direction |
|---|---|---|
| PSNR | `pyiqa` `psnr` | higher better |
| SSIM | `pyiqa` `ssim` | higher better |
| PI (Perceptual Index) | `pyiqa` `pi` | lower better |
| NIQE | `pyiqa` `niqe` | lower better |
| LPIPS | `pyiqa` `lpips` | lower better |
| CPU / RAM / GPU / GPU-mem | `metrics/cpu_and_ram.py`, `metrics/gpu.py` | — |

All image-quality metrics take tensors of shape `(1, C, H, W)` in `[0, 1]`. PSNR/SSIM/LPIPS compare result vs ground truth; PI/NIQE are no-reference (computed on the result only).

## Dependencies

```
torch==2.5.1+cu118  torchvision==0.20.1+cu118  torchaudio==2.5.1+cu118   # CUDA 11.8
numpy  opencv-python  Pillow  tqdm  huggingface-hub
pyiqa==0.1.13                          # PSNR/SSIM/PI/NIQE/LPIPS
line_profiler  memory_profiler         # profiling
tensorflow  tensorflow_datasets        # present in deps; not used by the main pipeline
```

Install: `pip install -r requirements.txt` (needs CUDA 11.8 — change the torch wheels if your CUDA differs).

## Example: evaluate a custom trained weight

```powershell
# 1. Copy a trained generator and name it with the scale token
Copy-Item ..\dcs-research\data\experiment\Eksperimen-33...\net_g.pth weights\experiment_33_x4.pth

# 2. Make sure it is NOT prefixed "1" (so it gets picked up), and datasets are enabled
# 3. Run a benchmark
python evaluate_basic.py              # or evaluate_custom.py for richer stats

# 4. Read a result log (per dataset / per scale)
Get-Content output\experiment_33_x4\BSD100\4\log_experiment_33_x4_BSD100_4.log
```

## What is NOT in this project

Inference-only fork. Present in the **official xinntao/Real-ESRGAN** but absent here:

| Component | Present? |
|---|:--:|
| Generator (RRDBNet) | YES |
| Discriminators (UNetSN, VGGStyle, …) | NO |
| GAN training loop / optimizer / scheduler | NO |
| Loss functions (L1, perceptual, GAN) | NO |
| YAML config / `options/` | NO |
| Training script, DIV2K loader, degradation | NO |
| basicsr dependency | NO |
| Face restoration / video | NO |

## Relationship to the DCS research

This is the **inference/validation tool** for the DCS research at BINUS University.

- Research repo: `C:\Users\kusan\Documents\code\self\s3\dcs-research`
- Training code & experiment configs (the custom discriminators — PatchGAN, ConvNeXt, NextSRGAN, the lightweight variants, etc.) live in `dcs-research/data/code/` and `dcs-research/data/experiment/`. Training uses a separate codebase (official xinntao Real-ESRGAN + basicsr).
- The `.pth` files in `weights/` here are the **trained generators from those experiments** (e.g. `RealESRGAN_33_ConvNextDiscriminator_*`, `..._30_PatchGAN_*`), evaluated by the scripts above.
- Aggregated metric tables and analysis derived from these runs are kept in `dcs-research/data/analysis/` (the `basic` pass is the one cited in the dissertation).
