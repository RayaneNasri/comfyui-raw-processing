import torch
import numpy as np
import cv2

# Decorators
from algorithms._utils import validate_image_input


@validate_image_input
def avg_filter(
    img: torch.Tensor, ksize: tuple[int, int], strBorderType: str = "BORDER_DEFAULT"
) -> torch.Tensor:
    """
    Simple wrapper for open-cv function blur.

    Args:
        img (torch.Tensor): Input image.
        ksize (tuple): Blurring kernel size.
        strBorderType (str): Border mode used to extrapolate pixels outside of the image, see the modes available in the open-cv documentation.

    Returns:
        torch.Tensor: Filtered image.
    """
    # ksize validation --
    if not isinstance(ksize, tuple):
        raise TypeError("kernel must be a tuple")

    kheight, kwidth = ksize
    if not isinstance(kheight, int):
        raise TypeError("kernel height must be integer")
    if kheight < 1:
        raise ValueError("kernel height must be positive")

    if not isinstance(kwidth, int):
        raise TypeError("kernel width must be integer")
    if kwidth < 1:
        raise ValueError("kernel width must be positive")

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
            "strBorderType must have one of these values BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_REFLECT_101, BORDER_TRANSPARENT, BORDER_DEFAULT or BORDER_ISOLATED"
        )

    # Wrapper start --
    src = torch.clip(img * 255, min=0, max=255)

    # This line:
    #   - Moves the image to the CPU
    #   - Converts to numpy
    #   - Converts from ComfyUI's RGB space to Open-CV's BGR space
    src = cv2.cvtColor(src.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)

    out = cv2.blur(src, ksize, borderType=borderType)
    return torch.from_numpy(
        cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    )
