import pyiqa
import torch

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_LPIPS_METRIC = pyiqa.create_metric('lpips', device=_DEVICE)

def calculate_lpips(ground_truth_image: torch.Tensor, result_image: torch.Tensor) -> float:
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity) between two images.
    
    Args:
        ground_truth_image: torch.Tensor shape (1, C, H, W), values in [0,1]
        result_image: torch.Tensor shape (1, C, H, W), values in [0,1]
    
    Returns:
        float: LPIPS score (lower is better, range typically 0-1)
    """
    return _LPIPS_METRIC(ground_truth_image, result_image).item()