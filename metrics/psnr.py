import pyiqa
import torch

_DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_PSNR_METRIC = pyiqa.create_metric('psnr', device=_DEVICE)

def calculate_psnr(ground_truth_image: torch.Tensor,
                   result_image: torch.Tensor) -> float:
    """
    img_true, img_test: torch.Tensor shape (1, C, H, W), values in [0,1]
    """
    return _PSNR_METRIC(ground_truth_image, result_image).item()