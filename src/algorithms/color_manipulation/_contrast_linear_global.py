import torch
from torch import Tensor 

def contrast_linear_global(rgb_image : Tensor, contrast_factor : float) -> Tensor:
    """
    - rgb_image : Tensor RGB image (H,W,3) with each channel represented as a float in [0,1]
    - contrast_factor : float (adviced values : between 0.5 and 2.0)
    
    contrast_factor < 1 -> decrease contrast
    contrast_factor > 1 -> increase contrast

    Multiplies deviations from the global mean of the rgb_image by contrast_factor and keeps mean fixed.
    """
    mean = rgb_image.mean()
    return torch.clamp((rgb_image - mean) * contrast_factor + mean, 0.0, 1.0)