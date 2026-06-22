import torch
import numpy as np

from typing import Callable


def validate_image_input(func: Callable) -> Callable:
    """
    Decorator that validates the input image tensor before executing the wrapped function.

    This decorator ensures that the first argument (`img`) passed to the function
    meets all the standard requirements for an RGB image tensor in the context of the project.
    Specifically, it checks that:
    - The input is a valid `torch.Tensor`.
    - The tensor has exactly 3 dimensions (Height, Width, Channels).
    - The spatial dimensions are not empty (H * W > 0).
    - The channel dimension contains exactly 3 channels (RGB).
    - All pixel values are normalized within the [0.0, 1.0] range.

    Parameters
    ----------
        func (Callable): The function to be decorated. Its first argument must be
            the image tensor (`img`).

    Returns
    -------
        Callable: The wrapped function containing the validation logic.

    Raises
    ------
        TypeError: If the input `img` is not an instance of `torch.Tensor`.
        ValueError: If `img` does not have exactly 3 dimensions.
        ValueError: If `img` is empty (Height or Width is 0).
        ValueError: If `img` does not have exactly 3 channels.
        ValueError: If any value within the `img` tensor is outside the [0.0, 1.0] range.
    """

    def wrapper(img, *args, **kwargs):
        if not isinstance(img, torch.Tensor):
            raise TypeError("img must be a torch Tensor")
        if img.ndim != 3:
            raise ValueError("img must be 3 dim tensor : H, W, C")

        img_h, img_w, img_c = img.shape
        if img_h * img_w == 0:
            raise ValueError("img is empty")
        if img_c != 3:
            raise ValueError(f"img shape must be (H, W, 3), but found (H, W, {img_c})")

        if not ((0.0 <= img).all() and (img <= 1.0).all()):
            raise ValueError("all img entries must be between 0 and 1")

        if not (img.dtype == torch.float32):
            raise TypeError("dtype of torch tensor must be torch.float32")
        return func(img, *args, **kwargs)

    return wrapper

def _to_uint8_numpy(image: torch.Tensor) -> np.ndarray:
    """Converts a float tensor image in [0, 1] to a uint8 numpy array.

    Args:
        image (torch.Tensor): A tensor of shape [H, W, 3] with float values in the range [0, 1]

    Returns:
        np.ndarray: A numpy array of shape [H, W, 3] with uint8 values in the range [0, 255]
    """
    
    img = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    img_uint8 = (img * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return img_uint8