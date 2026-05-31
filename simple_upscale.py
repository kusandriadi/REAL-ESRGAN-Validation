import os
import time
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor

from RealESRGAN import RealESRGAN
from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim
from metrics.perceptual_index import calculate_pi
from metrics.niqe import calculate_niqe
from metrics.lpips import calculate_lpips


def upscale_and_evaluate(input_image_path, output_image_path, ground_truth_path, weight_path, scale=4):
    """
    Generate high-resolution image from low-resolution input and calculate metrics
    
    Args:
        input_image_path: Path to low-resolution input image
        output_image_path: Path to save the upscaled result
        ground_truth_path: Path to ground truth high-resolution image (for metrics)
        weight_path: Path to model weights
        scale: Upscaling factor (default: 4)
    
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = RealESRGAN(device, scale=scale)
    model.load_weights(weight_path, download=False)
    
    # Set model to evaluation mode
    model.model.eval()
    for param in model.model.parameters():
        param.requires_grad = False
    
    print(f"Model loaded from: {weight_path}")
    
    # Load input image
    start_time = time.time()
    input_image = Image.open(input_image_path).convert('RGB')
    print(f"Input image size: {input_image.size}")
    
    # Generate high-resolution image
    print("Generating high-resolution image...")
    upscale_start = time.time()
    result_image = model.predict(input_image)
    upscale_time = time.time() - upscale_start
    print(f"Upscaling completed in {upscale_time:.2f} seconds")
    print(f"Output image size: {result_image.size}")
    
    # Save result
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    result_image.save(output_image_path)
    print(f"Result saved to: {output_image_path}")
    
    # Calculate metrics if ground truth is provided
    metrics = {}
    if ground_truth_path and os.path.exists(ground_truth_path):
        print("Calculating metrics...")
        ground_truth = Image.open(ground_truth_path).convert('RGB')
        
        # Resize result to match ground truth if needed
        if result_image.size != ground_truth.size:
            result_image = result_image.resize(ground_truth.size, Image.BICUBIC)
            print(f"Resized result to match ground truth: {ground_truth.size}")
        
        # Convert to tensors
        gt_tensor = to_tensor(ground_truth).unsqueeze(0).to(device)
        sr_tensor = to_tensor(result_image).unsqueeze(0).to(device)
        
        # Calculate metrics
        with torch.no_grad():
            metrics['psnr'] = calculate_psnr(gt_tensor, sr_tensor)
            metrics['ssim'] = calculate_ssim(gt_tensor, sr_tensor)
            metrics['pi'] = calculate_pi(sr_tensor)
            metrics['niqe'] = calculate_niqe(sr_tensor)
            metrics['lpips'] = calculate_lpips(gt_tensor, sr_tensor)
        
        # Print metrics
        print("\n=== Image Quality Metrics ===")
        print(f"PSNR (Higher Better): {metrics['psnr']:.2f} dB")
        print(f"SSIM (Higher Better): {metrics['ssim']:.4f}")
        print(f"PI (Lower Better): {metrics['pi']:.4f}")
        print(f"NIQE (Lower Better): {metrics['niqe']:.4f}")
        print(f"LPIPS (Lower Better): {metrics['lpips']:.4f}")
    else:
        print("No ground truth provided, skipping metrics calculation")
    
    total_time = time.time() - start_time
    metrics['upscale_time'] = upscale_time
    metrics['total_time'] = total_time
    
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    
    return metrics


def main():
    """Example usage"""
    # Configuration
    input_path = "inputs/lr/Set5/2x/baby.png"  # Change this to your input image
    output_path = "output/simple_upscale/baby_upscaled.png"
    gt_path = "inputs/gt/Set5/2x/baby.png"  # Ground truth for metrics
    weight_path = "weights/RealESRGAN_x4plus.pth"  # Change this to your model weights
    
    # Ensure paths exist
    if not os.path.exists(input_path):
        print(f"Input image not found: {input_path}")
        print("Please check the path and try again")
        return
    
    if not os.path.exists(weight_path):
        print(f"Weight file not found: {weight_path}")
        print("Please check the path and try again")
        return
    
    # Run upscaling and evaluation
    metrics = upscale_and_evaluate(
        input_image_path=input_path,
        output_image_path=output_path,
        ground_truth_path=gt_path,
        weight_path=weight_path,
        scale=4
    )
    
    # Save metrics to file
    metrics_path = output_path.replace('.png', '_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Image Upscaling Results\n")
        f.write("=" * 30 + "\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"Output: {output_path}\n")
        f.write(f"Ground Truth: {gt_path}\n")
        f.write(f"Model: {weight_path}\n\n")
        
        if 'psnr' in metrics:
            f.write("Quality Metrics:\n")
            f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
            f.write(f"SSIM: {metrics['ssim']:.4f}\n")
            f.write(f"PI: {metrics['pi']:.4f}\n")
            f.write(f"NIQE: {metrics['niqe']:.4f}\n")
            f.write(f"LPIPS: {metrics['lpips']:.4f}\n\n")
        
        f.write("Performance:\n")
        f.write(f"Upscaling Time: {metrics['upscale_time']:.2f} seconds\n")
        f.write(f"Total Time: {metrics['total_time']:.2f} seconds\n")
    
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()