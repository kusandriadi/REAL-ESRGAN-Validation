import pyiqa
import torch

_DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_PI_METRIC = pyiqa.create_metric('pi', device=_DEVICE)

def calculate_pi(result_image: torch.Tensor) -> float:
    """
    img: torch.Tensor shape (1, C, H, W), values in [0,1]
    """
    return _PI_METRIC(result_image).item()