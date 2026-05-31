import os
import re
import time
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

import version
from RealESRGAN import RealESRGAN
from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim
from metrics.perceptual_index import calculate_pi
from metrics.niqe import calculate_niqe
from metrics.lpips import calculate_lpips
from metrics.computation import monitor_system_metrics, calculate_average_metrics


def process_single_image(model, input_path, output_path, gt_path, device):
    """
    Process single image: upscale and calculate metrics
    
    Returns:
        dict: Dictionary containing calculated metrics
    """
    # Load input image
    image = Image.open(input_path).convert('RGB')
    
    # Generate high-resolution result
    result_image = model.predict(image)
    
    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_image.save(output_path)
    
    metrics = {}
    
    # Calculate metrics if ground truth exists
    if os.path.exists(gt_path):
        ground_truth = Image.open(gt_path).convert('RGB')
        
        # Ensure same size as GT
        if result_image.size != ground_truth.size:
            result_image = result_image.resize(ground_truth.size, Image.BICUBIC)
        
        # Convert to tensors
        gt_tensor = to_tensor(ground_truth).unsqueeze(0).to(device)
        sr_tensor = to_tensor(result_image).unsqueeze(0).to(device)
        
        # Calculate all metrics
        with torch.no_grad():
            metrics['psnr'] = calculate_psnr(gt_tensor, sr_tensor)
            metrics['ssim'] = calculate_ssim(gt_tensor, sr_tensor)
            metrics['pi'] = calculate_pi(sr_tensor)
            metrics['niqe'] = calculate_niqe(sr_tensor)
            metrics['lpips'] = calculate_lpips(gt_tensor, sr_tensor)
    
    return metrics


def process_dataset(model, dataset_path, output_root, dataset_name, device, log_name, 
                   total_params=None, trainable_params=None, model_size_mb=None, weight_file_size_mb=None):
    """
    Process entire dataset and calculate average metrics
    """
    start_time = time.time()
    monitor_start_time, collect_metrics, system_metrics = monitor_system_metrics()
    
    # Create separate metrics for each scale folder
    scales_metrics = {}
    
    # Count total images first for overall progress
    total_images = 0
    scale_folders = []
    for scale_folder in os.listdir(dataset_path):
        scale_path = os.path.join(dataset_path, scale_folder)
        if not os.path.isdir(scale_path):
            continue
        scale_folders.append((scale_folder, scale_path))
        image_count = len([f for f in os.listdir(scale_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        total_images += image_count
    
    print(f"  Total images to process: {total_images}")
    
    # Create progress bar for dataset
    dataset_pbar = tqdm(total=total_images, desc=f"  {dataset_name}", unit="img", leave=False,
                       bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for scale_folder, scale_path in scale_folders:
        # Initialize metrics for this scale
        scale_metrics = {
            'psnr': [],
            'ssim': [],
            'pi': [],
            'niqe': [],
            'lpips': []
        }
        
        # Create output directory matching input structure
        result_dir = os.path.join(output_root, scale_folder)
        os.makedirs(result_dir, exist_ok=True)
        
        processed_count = 0
        
        # Get image files for this scale
        image_files = [f for f in os.listdir(scale_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in image_files:
            image_name, ext = os.path.splitext(image_file)
            input_path = os.path.join(scale_path, image_file)
            output_path = os.path.join(result_dir, f"{image_name}.png")
            gt_path = os.path.join("inputs/gt", dataset_name, scale_folder, image_file)
            
            dataset_pbar.set_postfix_str(f"Scale {scale_folder}: {image_file}")
            
            # Monitor system before processing
            collect_metrics()
            
            # Process image and get metrics
            metrics = process_single_image(model, input_path, output_path, gt_path, device)
            
            # Monitor system after processing
            collect_metrics()
            
            # Collect metrics if available
            if metrics:
                for key in scale_metrics:
                    if key in metrics:
                        scale_metrics[key].append(metrics[key])
            
            processed_count += 1
            dataset_pbar.update(1)
        
        # Store metrics for this scale
        scales_metrics[scale_folder] = {
            'metrics': scale_metrics,
            'count': processed_count
        }
        
        # Save log file for this scale folder
        log_path = os.path.join(result_dir, f"{log_name}_{scale_folder}.log")
        save_metrics_log(
            log_path, scale_metrics, system_metrics, start_time,
            total_params, trainable_params, model_size_mb, weight_file_size_mb,
            processed_count
        )
    
    dataset_pbar.close()
    return scales_metrics


def save_metrics_log(log_path, metrics_dict, system_metrics, start_time,
                    total_params=None, trainable_params=None, model_size_mb=None, 
                    weight_file_size_mb=None, processed_count=0):
    """
    Save comprehensive metrics log to file
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, 'w', encoding='utf-8') as log_file:
        # Model parameters info
        if total_params is not None:
            log_file.write("Model Parameters:\n")
            log_file.write(f"Total parameters: {total_params:,}\n")
            log_file.write(f"Trainable parameters: {trainable_params:,}\n")
            log_file.write(f"Model size (calculated): {model_size_mb:.2f} MB\n")
            if weight_file_size_mb is not None:
                log_file.write(f"Weight file size: {weight_file_size_mb:.2f} MB\n")
            log_file.write("\n")
        
        # Image quality metrics
        log_file.write("Image Quality Metrics (Average):\n")
        log_file.write(f"Images processed: {processed_count}\n")
        
        for metric_name, values in metrics_dict.items():
            if values:
                avg_value = np.mean(values)
                std_value = np.std(values)
                min_value = np.min(values)
                max_value = np.max(values)
                
                metric_label = {
                    'psnr': 'PSNR (HB)',
                    'ssim': 'SSIM (HB)', 
                    'pi': 'PI (LB)',
                    'niqe': 'NIQE (LB)',
                    'lpips': 'LPIPS (LB)'
                }.get(metric_name, metric_name.upper())
                
                if metric_name == 'psnr':
                    log_file.write(f"{metric_label}: {avg_value:.2f} +/- {std_value:.2f} dB (min: {min_value:.2f}, max: {max_value:.2f})\n")
                else:
                    log_file.write(f"{metric_label}: {avg_value:.4f} +/- {std_value:.4f} (min: {min_value:.4f}, max: {max_value:.4f})\n")
            else:
                log_file.write(f"{metric_name.upper()}: No data available\n")
        
        # System metrics
        avg_system_metrics = calculate_average_metrics(system_metrics)
        log_file.write(f"\nSystem Performance (Average):\n")
        log_file.write(f"CPU: {avg_system_metrics['avg_cpu']:.2f}%\n")
        log_file.write(f"RAM: {avg_system_metrics['avg_ram_percent']:.2f}% ({avg_system_metrics['avg_ram_used_mb']:.2f} MB)\n")
        log_file.write(f"GPU: {avg_system_metrics['avg_gpu_percent']:.2f}%\n")
        log_file.write(f"GPU Memory: {avg_system_metrics['avg_gpu_memory_percent']:.2f}% ({avg_system_metrics['avg_gpu_memory_used_mb']:.2f} MB)\n")
        
        # Timing info
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        log_file.write(f"Total processing time: {minutes} minutes {seconds} seconds\n")
        
        if processed_count > 0:
            avg_time_per_image = total_time / processed_count
            log_file.write(f"Average time per image: {avg_time_per_image:.2f} seconds\n")


def main() -> int:
    version.printVersion()
    start_time = time.time()
    
    print("RealESRGAN Evaluation - custom (mean +/- std, min/max)")
    print("=" * 60)
    
    # Get available weight files (exclude files starting with '1')
    weight_files = [
        f for f in os.listdir('weights')
        if f.endswith('.pth') and not f.startswith('1')
    ]
    
    if not weight_files:
        print("No weight files found in 'weights' directory")
        return 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Found {len(weight_files)} weight files to process\n")
    
    # Create progress bar for models
    model_pbar = tqdm(weight_files, desc="Models", unit="model",
                     bar_format='{desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]')
    
    for weight_file in model_pbar:
        # Extract scale from weight filename
        match = re.search(r'_x(\d+)', weight_file)
        if not match:
            model_pbar.set_postfix_str(f"❌ Skipping {weight_file} - cannot determine scale")
            continue
        
        scale = int(match.group(1))
        weight_name = weight_file.split('.')[0]
        output_root = os.path.join("output", weight_name)
        
        model_pbar.set_postfix_str(f"{weight_file} (Scale: {scale}x)")
        
        try:
            # Initialize model
            model = RealESRGAN(device, scale=scale)
            model.load_weights(f'weights/{weight_file}', download=False)
            
            # Get model info
            total_params = sum(p.numel() for p in model.model.parameters())
            model.model.eval()
            for param in model.model.parameters():
                param.requires_grad = False
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            
            weight_file_size_mb = os.path.getsize(f'weights/{weight_file}') / 1024 / 1024
            model_size_mb = total_params * 4 / 1024 / 1024
            
        except Exception as e:
            model_pbar.set_postfix_str(f"❌ Failed to load {weight_file}: {e}")
            continue
        
        # Process each dataset
        dataset_count = 0
        for dataset in os.listdir("inputs/lr"):
            if dataset.startswith("1") or dataset.endswith(('.png', '.jpg', '.jpeg')):
                continue  # Skip files that start with '1' or individual image files
            dataset_path = os.path.join("inputs/lr", dataset)
            if not os.path.isdir(dataset_path):
                continue
            
            dataset_count += 1
            dataset_output = os.path.join(output_root, dataset)
            log_name = f"log_{weight_name}_{dataset}"
            
            try:
                scales_results = process_dataset(
                    model, dataset_path, dataset_output, dataset, device, log_name,
                    total_params, trainable_params, model_size_mb, weight_file_size_mb
                )
                
                # Update model progress bar with completion info
                total_images = sum(scale_data['count'] for scale_data in scales_results.values())
                model_pbar.set_postfix_str(f"DONE {dataset}: {total_images} images")
                
            except Exception as e:
                model_pbar.set_postfix_str(f"ERROR in {dataset}: {str(e)[:30]}...")
                continue
        
        if dataset_count == 0:
            model_pbar.set_postfix_str("WARNING: No datasets found")
    
    model_pbar.close()
    
    # Total execution time
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"\nAll processing completed!")
    print(f"Total execution time: {minutes} minutes {seconds} seconds")
    
    return 0


if __name__ == '__main__':
    main()