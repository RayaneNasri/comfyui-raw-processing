import torch
import numpy as np
import cv2
import warnings

# Decorators
from algorithms._utils import validate_image_input


@validate_image_input
def nl_means(
    img: torch.Tensor,
    h: float = 3,
    hColor: float = 3,
    templateWindowSize: int = 7,
    searchWindowSize: int = 21,
) -> torch.Tensor:
    """
    Simple wrapper for open-cv function fastNlMeansDenoisingColored

    Parameters
    ----------
    img : torch.Tensor
        Image to denoise
    h : float (default = 3)
        Parameter regulating filter strength for luminance component
    hColor : float (default = 3)
        The same as h but for color components
    templateWindowSize : (default = 7)
        Size in pixels of the template patch that is used to compute weights. Should be odd
    searchWindowSize : (default = 21)
        Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd

    Returns
    -------
    img_wb : torch.Tensor
        Denoised image
    """
    if not isinstance(h, (int, float)):
        raise TypeError("h must be a number (float or int)")
    if not isinstance(hColor, (int, float)):
        raise TypeError("hColor must be a number (float or int)")
    if h < 0.0:
        raise ValueError("h must be positive")
    if hColor < 0.0:
        raise ValueError("hColor must be positive")

    img_h, img_w, _ = img.shape
    if not isinstance(searchWindowSize, int):
        raise TypeError("searchWindowSize must be an integer")
    if searchWindowSize < 1:
        raise ValueError("searchWindowSize must be positive")
    if searchWindowSize > min(img_h, img_w) / 2:
        max_val = int(min(img_h, img_w) / 2)
        searchWindowSize = max_val - 1 if (max_val % 2 == 0) else max_val
        warnings.warn(
            f"searchWindowSize too large. The value is set to {searchWindowSize}"
        )
    if searchWindowSize % 2 == 0:
        searchWindowSize -= 1
        warnings.warn(
            f"searchWindowSize must be an odd number. The value is set to {searchWindowSize}"
        )

    if not isinstance(templateWindowSize, int):
        raise TypeError("templateWindowSize must be an integer")
    if templateWindowSize < 1:
        raise ValueError("templateWindowSize must be positive")
    if templateWindowSize % 2 == 0:
        templateWindowSize -= 1
        warnings.warn(
            f"templateWindowSize must be an odd number. The value is set to {templateWindowSize}"
        )

    if searchWindowSize < templateWindowSize:
        searchWindowSize = templateWindowSize
        warnings.warn(
            f"searchWindowSize cannot be strictly smaller than templateWindowSize. The value is set to {searchWindowSize}"
        )

    # Wrapper start --
    src = torch.clip(img * 255, min=0, max=255)

    # This line:
    #   - Moves the image to the CPU
    #   - Converts to numpy
    #   - Converts from ComfyUI's RGB space to Open-CV's BGR space
    src = cv2.cvtColor(src.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)

    out = cv2.fastNlMeansDenoisingColored(
        src,
        h=h,
        hColor=hColor,
        templateWindowSize=templateWindowSize,
        searchWindowSize=searchWindowSize,
    )
    return torch.from_numpy(cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(float) / 255.0)
