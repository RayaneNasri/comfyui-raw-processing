import torch
import numpy as np
import math as m
import cv2

from algorithms._utils import validate_image_input


@validate_image_input
def bilateral_filter(
    img: torch.Tensor,
    d: int,
    sigmaColor: float,
    sigmaSpace: float,
    strBorderType: str = "BORDER_DEFAULT",
) -> torch.Tensor:
    """
    Simple wrapper for open-cv function bilateralFilter.

    Args:
        img (torch.Tensor): Input image.
        d (int): Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
        sigmaColor (float): Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color.
        sigmaSpace (float): Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough.
        strBorderType (str): Pixel extrapolation method, see the modes available in the open-cv documentation. BORDER_WRAP is not supported.

    Returns:
        torch.Tensor: Filtered image.
    """

    # d validation --
    if not isinstance(d, int):
        raise TypeError(f"d must be an integer, but found {type(d)}")

    # sigmaColor validation --
    if not isinstance(sigmaColor, (int, float)):
        raise TypeError(
            f"sigmaColor must be either int or float, but found {type(sigmaColor)}"
        )
    if m.isinf(sigmaColor) or m.isnan(sigmaColor):
        raise ValueError("sigmaColor cannot be NaN or infinite")
    if sigmaColor < 0:
        raise ValueError("sigmaColor can't be negative")

    # sigmaSpace validation --
    if not isinstance(sigmaSpace, (int, float)):
        raise TypeError(
            f"sigmaSpace must be either int or float, but found {type(sigmaSpace)}"
        )
    if m.isinf(sigmaSpace) or m.isnan(sigmaSpace):
        raise ValueError("sigmaSpace cannot be NaN or infinite")
    if sigmaSpace < 0:
        raise ValueError("sigmaSpace can't be negative")

    # borderType validation --
    if not isinstance(strBorderType, str):
        raise TypeError(f"borderType must be an str, but found {type(strBorderType)}")

    borderTypes: dict[str, int] = {
        "BORDER_CONSTANT": cv2.BORDER_CONSTANT,
        "BORDER_REPLICATE": cv2.BORDER_REPLICATE,
        "BORDER_REFLECT": cv2.BORDER_REFLECT,
        "BORDER_REFLECT_101": cv2.BORDER_REFLECT_101,
        "BORDER_TRANSPARENT": cv2.BORDER_TRANSPARENT,
        "BORDER_DEFAULT": cv2.BORDER_DEFAULT,
        "BORDER_ISOLATED": cv2.BORDER_ISOLATED,
    }

    try:
        borderType = borderTypes[strBorderType]
    except KeyError:
        raise ValueError(
            "strBorderType must have one of these values : BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_REFLECT_101, BORDER_TRANSPARENT, BORDER_DEFAULT or BORDER_ISOLATED"
        )

    # Wrapper start --
    src = torch.clip(img * 255, min=0, max=255)

    # This line:
    #   - Moves the image to the CPU
    #   - Converts to numpy
    #   - Converts from ComfyUI's RGB space to Open-CV's BGR space
    src = cv2.cvtColor(src.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
    out = cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, borderType=borderType)

    return torch.from_numpy(
        cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    )
