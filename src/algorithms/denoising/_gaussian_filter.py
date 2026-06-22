import torch
import numpy as np
import math as m
import cv2

from algorithms._utils import validate_image_input


@validate_image_input
def gaussian_filter(
    img: torch.Tensor,
    ksize: tuple[int, int],
    sigmaX: float,
    sigmaY: float = 0,
    strBorderType: str = "BORDER_DEFAULT",
    strHint: str = "ALGO_HINT_DEFAULT",
) -> torch.Tensor:
    """
    Simple wrapper for open-cv function GaussianBlur

    Parameters
    ----------
    img : torch.Tensor
        Input image
    ksize : tuple
        Gaussian kernel size
        ksize.width and ksize.height can differ but they both must be positive and odd
        Or, they can be zero's and then they are computed from sigma
    sigmaX : float
        Gaussian kernel standard deviation in X direction.
    sigmaY : float
        Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height, respectively (see getGaussianKernel for details); to fully control the result regardless of possible future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.
    strBorderType : str
        Pixel extrapolation method, see the modes available in the open-cv documentation. BORDER_WRAP is not supported.
    strHint : str
        Implementation modfication flags. See AlgorithmHint in the open-cv documentation

    Returns
    -------
    img_wb : torch.Tensor
        Filtred Image
    """
    # ksize validation --
    if not isinstance(ksize, tuple):
        raise TypeError("kernel must be a tuple")

    kheight, kwidth = ksize
    if not isinstance(kheight, int):
        raise TypeError("kernel height must be integer")
    if not isinstance(kwidth, int):
        raise TypeError("kernel width must be integer")

    if ((kheight < 1) and ((kheight < 0) or (kwidth != 0))) or (
        (kheight >= 1) and ((kheight % 2 == 0) or (kwidth % 2 == 0) or (kwidth < 1))
    ):
        raise ValueError(
            "ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero's and then they are computed from sigma"
        )

    # sigmaX validation --
    if not isinstance(sigmaX, (int, float)):
        raise TypeError(f"sigmaX must be either int or float, but found {type(sigmaX)}")
    if m.isinf(sigmaX) or m.isnan(sigmaX):
        raise ValueError("sigmaX cannot be NaN or infinite")
    if sigmaX < 0:
        raise ValueError(
            "the gaussian kernel standard deviation in X direction can't be negative"
        )

    # sigmaY validation --
    if not isinstance(sigmaY, (int, float)):
        raise TypeError(f"sigmaY must be either int or float, but found {type(sigmaY)}")
    if m.isinf(sigmaY) or m.isnan(sigmaY):
        raise ValueError("sigmaY cannot be NaN or infinite")
    if sigmaY < 0:
        raise ValueError(
            "the gaussian kernel standard deviation in Y direction can't be negative"
        )

    # borderType validation --
    if not isinstance(strBorderType, str):
        raise TypeError("borderType must be an str")

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

    # strHint validation --
    if not isinstance(strHint, str):
        raise TypeError("strHint must be an str")

    hintTypes: dict[str, int] = {
        "ALGO_HINT_DEFAULT": cv2.ALGO_HINT_DEFAULT,
        "ALGO_HINT_ACCURATE": cv2.ALGO_HINT_ACCURATE,
        "ALGO_HINT_APPROX": cv2.ALGO_HINT_APPROX,
    }

    try:
        hint = hintTypes[strHint]
    except KeyError:
        raise ValueError(
            "strHint must have one of these values : ALGO_HINT_DEFAULT, ALGO_HINT_ACCURATE or ALGO_HINT_APPROX"
        )

    # Wrapper start --
    src = torch.clip(img * 255, min=0, max=255)

    # This line:
    #   - Moves the image to the CPU
    #   - Converts to numpy
    #   - Converts from ComfyUI's RGB space to Open-CV's BGR space
    src = cv2.cvtColor(src.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)

    out = cv2.GaussianBlur(
        src, ksize, sigmaX, sigmaY=sigmaY, borderType=borderType, hint=hint
    )

    return torch.from_numpy(cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(float) / 255.0)
