import os

import numpy as np
import time
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from metrics.computation import monitor_system_metrics, calculate_average_metrics
from metrics.perceptual_index import calculate_pi
from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim
from metrics.niqe import calculate_niqe
from metrics.lpips import calculate_lpips

_DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def upscale(log_name, dataset_path, output_folder, dataset, model, total_params=None, trainable_params=None, model_size_mb=None, weight_file_size_mb=None):
    # Mulai monitoring sistem dan metrics kosong
    start_time = time.time()
    monitor_start_time, collect_metrics, metrics = monitor_system_metrics()
    psnr_values = []
    ssim_values = []
    pi_values   = []
    niqe_values = []
    lpips_values = []

    # Count total images first
    total_images = 0
    scale_folders = []
    for scale_folder in os.listdir(dataset_path):
        scale_path = os.path.join(dataset_path, scale_folder)
        if not os.path.isdir(scale_path):
            continue
        image_files = [f for f in os.listdir(scale_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        total_images += len(image_files)
        scale_folders.append((scale_folder, scale_path, image_files))
    
    # Progress bar will be handled by calling function

    for scale_folder, scale_path, image_files in scale_folders:
        # Determine output path based on folder structure
        result_dir = os.path.join(output_folder, scale_folder)
        os.makedirs(result_dir, exist_ok=True)
        
        for image_file in image_files:
            image_name, _ = os.path.splitext(image_file)
            image_path = os.path.join(scale_path, image_file)
            image = Image.open(image_path).convert('RGB')

            # Load ground truth image from inputs/gt
            gt_path = os.path.join("inputs/gt", dataset, scale_folder, image_file)
            if not os.path.exists(gt_path):
                print(f"Ground truth file not found: {gt_path}")
                continue

            ground_truth_image = Image.open(gt_path).convert('RGB')

            # Start monitoring before prediction
            collect_metrics()
            result_image = model.predict(image)

            result_path = os.path.join(result_dir, f"{image_name}.png")
            result_image.save(result_path)
            collect_metrics()

            # ensure same size as GT
            if result_image.size != ground_truth_image.size:
                result_image = result_image.resize(ground_truth_image.size, Image.BICUBIC)

            # convert PIL → tensor 1×C×H×W in [0,1]
            gt_t = to_tensor(ground_truth_image).unsqueeze(0).to(_DEVICE)
            sr_t = to_tensor(result_image).unsqueeze(0).to(_DEVICE)

            psnr_values.append(calculate_psnr(gt_t, sr_t))
            ssim_values.append(calculate_ssim(gt_t, sr_t))
            pi_values.append(calculate_pi(sr_t))
            niqe_values.append(calculate_niqe(sr_t))
            lpips_values.append(calculate_lpips(gt_t, sr_t))
    
    # Save log files for each scale after processing all images
    for scale_folder, scale_path, image_files in scale_folders:
        result_dir = os.path.join(output_folder, scale_folder)
        with open(os.path.join(result_dir, f"{log_name}_{scale_folder}.log"), 'w', encoding='utf-8') as log_file:
            # Your code to write to the log file
            avg_metrics, avg_pi, avg_psnr, avg_ssim, avg_niqe, avg_lpips = calculate_metrics(
                log_file,
                metrics,
                pi_values,
                psnr_values,
                ssim_values,
                niqe_values,
                lpips_values,
                start_time,
                total_params,
                trainable_params,
                model_size_mb,
                weight_file_size_mb
            )

def calculate_metrics(log_file, metrics, pi_values, psnr_values, ssim_values, niqe_values, lpips_values, start_time, total_params=None, trainable_params=None, model_size_mb=None, weight_file_size_mb=None):
    # Hitung rata-rata metrik image quality
    avg_psnr = np.mean(psnr_values) if psnr_values else float('nan')
    avg_ssim = np.mean(ssim_values) if ssim_values else float('nan')
    avg_pi   = np.mean(pi_values)   if pi_values   else float('nan')
    avg_niqe = np.mean(niqe_values) if niqe_values else float('nan')
    avg_lpips = np.mean(lpips_values) if lpips_values else float('nan')

    # Hitung rata-rata pemakaian sistem
    avg_metrics = calculate_average_metrics(metrics)

    # Total waktu sejak start_time
    total_time = time.time() - start_time

    # Tulis ke log file
    if total_params is not None:
        log_file.write("Model Parameters:\n")
        log_file.write(f"Total parameters: {total_params:,}\n")
        log_file.write(f"Trainable parameters: {trainable_params:,}\n")
        log_file.write(f"Model size (calculated): {model_size_mb:.2f} MB\n")
        if weight_file_size_mb is not None:
            log_file.write(f"Weight file size: {weight_file_size_mb:.2f} MB\n")
        log_file.write("\n")
    
    log_file.write("Image Quality Metrics:\n")
    log_file.write(f"PSNR (HB): {avg_psnr:.2f}\n")
    log_file.write(f"SSIM (HB): {avg_ssim:.4f}\n")
    log_file.write(f"PI   (LB): {avg_pi:.4f}\n")
    log_file.write(f"NIQE (LB): {avg_niqe:.4f}\n")
    log_file.write(f"LPIPS (LB): {avg_lpips:.4f}\n\n")
    log_file.write("Rata-rata penggunaan sistem selama proses:\n")
    log_file.write(f"CPU: {avg_metrics['avg_cpu']:.2f}%\n")
    log_file.write(f"RAM: {avg_metrics['avg_ram_percent']:.2f}% ({avg_metrics['avg_ram_used_mb']:.2f} MB)\n")
    log_file.write(f"GPU: {avg_metrics['avg_gpu_percent']:.2f}%\n")
    log_file.write(f"Memori GPU: {avg_metrics['avg_gpu_memory_percent']:.2f}% ({avg_metrics['avg_gpu_memory_used_mb']:.2f} MB)\n")
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    log_file.write(f"Waktu total: {minutes} minutes {seconds} seconds\n")

    return avg_metrics, avg_pi, avg_psnr, avg_ssim, avg_niqe, avg_lpips