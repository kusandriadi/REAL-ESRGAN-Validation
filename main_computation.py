import os
import re

import time
import torch
from tqdm import tqdm

import version
from RealESRGAN import RealESRGAN
from RealESRGAN.process.upscale import upscale


def main() -> int:
    version.printVersion()
    start_time = time.time()

    # Loop melalui semua file weight
    weight_files = [
        f for f in os.listdir('weights')
        if f.endswith('.pth') and not f.startswith('1')
    ]

    print(f"RealESRGAN Image Upscaling with Progress Bars")
    print(f"Found {len(weight_files)} weight files to process")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create progress bar for models
    model_pbar = tqdm(weight_files, desc="Models", unit="model",
                     bar_format='{desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]')

    for weight_file in model_pbar:
        # Ekstrak scale dari nama file weight
        match = re.search(r'_x(\d+)', weight_file)
        if not match:
            model_pbar.set_postfix_str(f"ERROR: Skipping {weight_file} - cannot determine scale")
            continue

        scale       = int(match.group(1))
        weight_name = weight_file.split('.')[0]
        output_root = os.path.join("output", weight_name)
        os.makedirs(output_root, exist_ok=True)

        model_pbar.set_postfix_str(f"{weight_file} (Scale: {scale}x)")

        model  = RealESRGAN(device, scale=scale)
        model.load_weights(f'weights/{weight_file}', download=False)
        
        # Count parameters after loading weights
        total_params = sum(p.numel() for p in model.model.parameters())
        
        # For inference models, we can set requires_grad=False to show actual trainable vs frozen
        # Set model to eval mode and freeze parameters for inference
        model.model.eval()
        for param in model.model.parameters():
            param.requires_grad = False
            
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        # Get actual file size of the weight file
        weight_file_size_mb = os.path.getsize(f'weights/{weight_file}') / 1024 / 1024
        model_size_mb = total_params * 4 / 1024 / 1024

        # Count total datasets for this model
        datasets = [d for d in os.listdir("inputs/lr") 
                   if not d.startswith("1") and os.path.isdir(os.path.join("inputs/lr", d))]
        
        # Loop melalui setiap dataset LR
        for i, dataset in enumerate(datasets, 1):
            dataset_path = os.path.join("inputs/lr", dataset)
            
            # Update progress bar with current dataset
            model_pbar.set_postfix_str(f"Dataset {i}/{len(datasets)}: {dataset}")

            # Buat folder output per-dataset
            dataset_out = os.path.join(output_root, dataset)
            os.makedirs(dataset_out, exist_ok=True)

            # Path log di dalam folder dataset
            log_name = f"log_{weight_name}_{dataset}"

            # Proses semua gambar dengan fungsi upscale
            upscale(
                log_name,
                dataset_path,
                dataset_out,
                dataset,
                model,
                total_params,
                trainable_params,
                model_size_mb,
                weight_file_size_mb
            )
    
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
