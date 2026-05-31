import pyiqa
import torch

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_NIQE_METRIC = pyiqa.create_metric('niqe', device=_DEVICE)

def calculate_niqe(result_image: torch.Tensor) -> float:
    """
    Calculate NIQE (Natural Image Quality Evaluator) score for a single image.
    
    Args:
        result_image: torch.Tensor shape (1, C, H, W), values in [0,1]
    
    Returns:
        float: NIQE score (lower is better)
    """
    return _NIQE_METRIC(result_image).item()