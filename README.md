# Real-ESRGAN (Inference & Evaluation)

Inference and benchmarking pipeline for Real-ESRGAN super-resolution models.
Fork of [ai-forever/Real-ESRGAN](https://github.com/ai-forever/Real-ESRGAN) (simplified Sberbank AI implementation), with custom evaluation tools.

> This is NOT the official [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
> This codebase is **inference-only** -- no training code, no discriminator, no GAN training pipeline.

- [Paper: Real-ESRGAN (Wang et al., 2021)](https://arxiv.org/abs/2107.10833)
- [Original implementation (xinntao)](https://github.com/xinntao/Real-ESRGAN)
- [HuggingFace weights](https://huggingface.co/sberbank-ai/Real-ESRGAN)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full benchmark (semua weights, semua dataset aktif)
python main_computation.py

# 3. Atau jalankan simple inference saja
python main.py
```

## How to Run (Step by Step)

### Langkah 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Butuh CUDA 11.8. Jika CUDA berbeda, ganti versi torch di `requirements.txt`.

### Langkah 2: Siapkan Weight File (.pth)

Taruh file `.pth` di folder `weights/`. Aturan penamaan:

```
weights/
├── NamaModel_x4.pth          ← Scale harus ada di nama file: _x2, _x4, atau _x8
├── RealESRGAN_default_x4.pth ← Contoh yang sudah ada
├── MyCustomModel_x4.pth      ← Contoh custom weight
└── 1disabled_x4.pth          ← Prefix "1" = dilewati (tidak diproses)
```

**Aturan nama file weight:**
- Harus berakhiran `.pth`
- Harus mengandung `_x2`, `_x4`, atau `_x8` (script extract scale dari sini via regex `_x(\d+)`)
- Nama yang diawali `1` akan di-skip oleh `main_computation.py`
- Contoh valid: `experiment_33_x4.pth`, `ConvNeXt_model_x4.pth`, `RealESRGAN_default_x2.pth`
- Contoh invalid: `model.pth` (tidak ada scale), `model_4x.pth` (format salah)

### Langkah 3: Format File .pth

File `.pth` adalah PyTorch state dict untuk arsitektur **RRDBNet** (702-704 keys). Script mendukung 3 format:

**Format 1: Raw OrderedDict (langsung state_dict)** -- yang dipakai di project ini
```python
# File .pth langsung berisi state_dict
torch.save(model.state_dict(), 'weights/my_model_x4.pth')

# Keys dimulai dengan:
# conv_first.weight         torch.Size([64, 3, 3, 3])   ← x4/x8
# conv_first.weight         torch.Size([64, 12, 3, 3])  ← x2 (pixel_unshuffle 3ch→12ch)
# body.0.rdb1.conv1.weight  torch.Size([32, 64, 3, 3])
# ...
# conv_last.bias            torch.Size([3])
```

**Format 2: Wrapped dalam key `params`** -- format official xinntao Real-ESRGAN
```python
torch.save({'params': model.state_dict()}, 'weights/my_model_x4.pth')
```

**Format 3: Wrapped dalam key `params_ema`** -- EMA weights dari training
```python
torch.save({'params_ema': model.state_dict()}, 'weights/my_model_x4.pth')
```

**Detail state_dict keys per scale:**

| Scale | `conv_first.weight` shape | Upsample layers | Total keys |
|:-----:|:-------------------------:|:---------------:|:----------:|
| x2    | `[64, 12, 3, 3]`         | conv_up1, conv_up2 | 702 |
| x4    | `[64, 3, 3, 3]`          | conv_up1, conv_up2 | 702 |
| x8    | `[64, 3, 3, 3]`          | conv_up1, conv_up2, conv_up3 | 704 |

Perbedaan x2: input 3 channel di-pixel_unshuffle jadi 12 channel sebelum conv_first.
Perbedaan x8: ada 3 upsample layer (bukan 2), sehingga 2 key lebih banyak.

**Arsitektur harus match:** `RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)`. Jika weight dari arsitektur berbeda (misal num_block=6), load akan gagal (`strict=True`).

### Langkah 4: Siapkan Dataset

Dataset ditaruh di 2 tempat dengan **nama folder dan nama file yang harus sama persis**:

```
inputs/
├── lr/                              ← Low-resolution images (INPUT)
│   └── NamaDataset/                 ← Nama folder = nama dataset
│       └── 4/                       ← Subfolder = scale factor
│           ├── img_001.png
│           ├── img_002.png
│           └── ...
│
└── gt/                              ← Ground truth / high-res images (REFERENSI)
    └── NamaDataset/                 ← HARUS SAMA dengan nama di lr/
        └── 4/                       ← HARUS SAMA scale factor
            ├── img_001.png          ← HARUS SAMA nama file
            ├── img_002.png
            └── ...
```

**Aturan kritis:**
- Nama folder dataset di `lr/` dan `gt/` **harus identik** (case-sensitive)
- Nama file gambar di `lr/` dan `gt/` **harus identik** (misal `img_001.png`)
- Subfolder angka (`2/`, `3/`, `4/`) menunjukkan scale factor
- Format gambar: `.png`, `.jpg`, `.jpeg`
- Folder di `lr/` yang diawali `1` akan di-skip (mekanisme disable)

**Contoh yang sudah ada:**

```
inputs/lr/Set5/4/img_001.png    ←→    inputs/gt/Set5/4/img_001.png
inputs/lr/Set5/4/img_002.png    ←→    inputs/gt/Set5/4/img_002.png
```

**Cara menambahkan dataset baru:**
```bash
# Contoh: tambahkan dataset "Manga109" untuk scale x4
mkdir -p inputs/lr/Manga109/4
mkdir -p inputs/gt/Manga109/4

# Copy LR images ke inputs/lr/Manga109/4/
# Copy HR (ground truth) images ke inputs/gt/Manga109/4/
# Pastikan nama file sama di kedua folder
```

**Cara disable dataset tanpa menghapus:**
```bash
# Rename folder dengan prefix "1"
mv inputs/lr/Set5 inputs/lr/1Set5    # Set5 sekarang di-skip
mv inputs/lr/1Set5 inputs/lr/Set5    # Aktifkan kembali
```

### Langkah 5: Jalankan Script

**Script utama: `main_computation.py`** (untuk benchmark lengkap)

```bash
python main_computation.py
```

Apa yang terjadi:
1. Print info CUDA/PyTorch
2. Scan `weights/` → ambil semua `.pth` yang tidak berawalan `1`
3. Untuk setiap weight file:
   - Extract scale dari nama file (regex `_x(\d+)`)
   - Load model ke GPU
   - Proses setiap dataset aktif di `inputs/lr/`
   - Untuk setiap gambar: super-resolve → simpan hasil → hitung PSNR vs ground truth
   - Monitor CPU/RAM/GPU selama proses
4. Output:
   - Gambar hasil: `output/{weight_name}/{dataset}/{scale}/img_NNN.png`
   - Log file: `output/{weight_name}/log_{weight_name}.log`

Contoh log output:
```
Hitung Metriks Gambar:
PSNR: 28.45

Rata-rata penggunaan sistem selama proses:
CPU: 15.23%
RAM: 45.67% (7234.56 MB)
GPU: 78.90%
Memori GPU: 34.56% (2345.67 MB)
Waktu total: 123.45 detik
```

**Script sederhana: `main.py`** (untuk coba cepat)

```bash
python main.py
```

Hardcoded scale=2 dan weight `weights/RealESRGAN_x2.pth`. Memproses semua gambar langsung di folder `inputs/` (bukan subfolder dataset). Output ke `results/`.

### Contoh: Evaluasi Custom Weight dari DCS Research

```bash
# 1. Copy weight hasil training dari experiment (misal Eksperimen-33)
cp /path/to/experiment/net_g_100000.pth weights/experiment_33_x4.pth

# 2. Pastikan dataset aktif (hapus prefix "1" jika perlu)
mv inputs/lr/1BSD100 inputs/lr/BSD100
mv inputs/lr/1Set14 inputs/lr/Set14
mv inputs/lr/1Urban100 inputs/lr/Urban100

# 3. Jalankan benchmark
python main_computation.py

# 4. Lihat hasil
cat output/experiment_33_x4/log_experiment_33_x4.log
```

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
