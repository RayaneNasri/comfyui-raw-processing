import torch
import numpy as np
import cv2

# Decorators    
from algorithms._utils import validate_image_input

@validate_image_input
def median_filter(img: torch.Tensor, 
                  ksize: int
                ) -> torch.Tensor:
    """
    Simple wrapper for open-cv function medianBlur

    Parameters
    ----------
    img : torch.Tensor
        Input image
    ksize : int
        Aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ...

    Returns
    -------
    img_wb : torch.Tensor
        Filtred Image
    """
    # ksize validation -- 
    if not isinstance(ksize, int):
        raise TypeError("ksize must be an integer")
    if ksize < 2 or ksize % 2 == 0:
        raise ValueError("ksize must be odd and greater than 1, for example: 3, 5, 7 ...")
    
    # Wrapper start -- 
    src = torch.clip(img * 255, min=0, max=255)

    # This line:
    #   - Moves the image to the CPU
    #   - Converts to numpy
    #   - Converts from ComfyUI's RGB space to Open-CV's BGR space
    src = cv2.cvtColor(src.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)

    out = cv2.medianBlur(src, ksize)
    return torch.from_numpy( cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(float) / 255. )