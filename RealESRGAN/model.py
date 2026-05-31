import os

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_url, hf_hub_download

from .rrdbnet_arch import RRDBNet
from .utils import pad_reflect, split_image_into_overlapping_patches, stich_together, \
    unpad_image

HF_MODELS = {
    2: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x2.pth',
    ),
    4: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x4.pth',
    ),
    8: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x8.pth',
    ),
}


class RealESRGAN:
    def __init__(self, device, scale=4):
        self.device = device
        self.scale = scale
        self.model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=scale
        )
        
    def load_weights(self, model_path, download=True):
        if not os.path.exists(model_path) and download:
            assert self.scale in [2,4,8], 'You can download models only with scales: 2, 4, 8'
            config = HF_MODELS[self.scale]
            cache_dir = os.path.dirname(model_path)
            local_filename = os.path.basename(model_path)
            config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
            hf_hub_download(config_file_url, cache_dir=cache_dir, force_filename=local_filename)
            print('Weights downloaded to:', os.path.join(cache_dir, local_filename))
        
        loadnet = torch.load(model_path, weights_only=True)
        
        # Auto-detect architecture from weight file
        if 'params' in loadnet:
            state_dict = loadnet['params']
        elif 'params_ema' in loadnet:
            state_dict = loadnet['params_ema']
        else:
            state_dict = loadnet
            
        # Detect architecture from weight shapes
        num_feat, num_block, num_grow_ch = self._detect_architecture(state_dict)
        
        # Recreate model with detected architecture
        self.model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=num_feat,
            num_block=num_block, num_grow_ch=num_grow_ch, scale=self.scale
        )
        
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)
        
    def _detect_architecture(self, state_dict):
        # Default values
        num_feat = 64
        num_block = 23  
        num_grow_ch = 32
        
        # Debug: print first 10 keys to see structure
        print("Weight file keys (first 10):")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"  {key}: {state_dict[key].shape}")
        
        # Count RRDB blocks by looking for 'body.' layers
        block_keys = [k for k in state_dict.keys() if k.startswith('body.') and '.RDB' in k]
        print(f"Found {len(block_keys)} block keys")
        
        if block_keys:
            block_indices = set()
            for key in block_keys:
                parts = key.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    block_indices.add(int(parts[1]))
            if block_indices:
                num_block = max(block_indices) + 1
                print(f"Detected num_block: {num_block}")
        
        # Detect num_feat from conv_first weight shape
        if 'conv_first.weight' in state_dict:
            num_feat = state_dict['conv_first.weight'].shape[0]
            print(f"Detected num_feat from conv_first: {num_feat}")
            
        # Detect num_grow_ch from first RDB layer
        for key in state_dict.keys():
            if '.RDB1.conv1.0.weight' in key:
                num_grow_ch = state_dict[key].shape[0]
                print(f"Detected num_grow_ch: {num_grow_ch}")
                break
                
        print(f"Final architecture: num_feat={num_feat}, num_block={num_block}, num_grow_ch={num_grow_ch}")
        return num_feat, num_block, num_grow_ch
        
    @torch.cuda.amp.autocast(enabled=True)
    def predict(self, lr_image, batch_size=4, patches_size=192,
                padding=24, pad_size=15):
        scale = self.scale
        device = self.device
        lr_image = np.array(lr_image)
        lr_image = pad_reflect(lr_image, pad_size)

        patches, p_shape = split_image_into_overlapping_patches(
            lr_image, patch_size=patches_size, padding_size=padding
        )
        img = torch.FloatTensor(patches/255).permute((0,3,1,2)).to(device).detach()

        with torch.no_grad():
            res = self.model(img[0:batch_size])
            for i in range(batch_size, img.shape[0], batch_size):
                res = torch.cat((res, self.model(img[i:i+batch_size])), 0)

        sr_image = res.permute((0,2,3,1)).clamp_(0, 1).cpu()
        np_sr_image = sr_image.numpy()

        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
        np_sr_image = stich_together(
            np_sr_image, padded_image_shape=padded_size_scaled, 
            target_shape=scaled_image_shape, padding_size=padding * scale
        )
        sr_img = (np_sr_image*255).astype(np.uint8)
        sr_img = unpad_image(sr_img, pad_size*scale)
        sr_img = Image.fromarray(sr_img)

        return sr_img