import torch
import numpy as np


# FIXME: Tensor size too large with pyTorch
def white_patch_ref(
    img: torch.Tensor,
    percentil: float,
) -> torch.Tensor:
    """
    Apply white balance to an image using the White Patch Reference algorithm.

    This algorithm assumes that the brightest patch or pixels in the image should 
    be white. It normalizes the color channels based on the maximum value (or a 
    specified percentile) found in each individual channel.

    Args:
        img (torch.Tensor): The input image tensor to be white-balanced.
        percentil (float): A value between [0.0, 1.0] indicating the percentile 
            to consider as the channel maximum. A value of 1.0 uses the absolute 
            maximum value of each channel.

    Returns:
        torch.Tensor: The white-balanced image tensor, clipped to the range [0.0, 1.0].

    Raises:
        ValueError: If `percentil` is not between 0.0 and 1.0.
    """
    if (percentil < 0.0) or (percentil > 1.0):
        raise ValueError(
            f"The percentil must be between 0 and, 1 but found {percentil}"
        )

    reshaped_image = img.reshape(-1, 3)

    if abs(percentil - 1.0) < 10 ** (-5):
        parameters = torch.max(reshaped_image, dim=0)[0]
    else:
        try:
            parameters = torch.quantile(reshaped_image, percentil, dim=0)

        except RuntimeError:
            np_reshaped_image = img.reshape(-1, 3).numpy()
            parameters = torch.tensor(
                np.percentile(np_reshaped_image, percentil * 100, axis=0)
            )

    img_wb = img.clone()

    img_wb[:, :, 0] /= parameters[0] if parameters[0] > 0.0 else 1.0
    img_wb[:, :, 1] /= parameters[1] if parameters[1] > 0.0 else 1.0
    img_wb[:, :, 2] /= parameters[2] if parameters[2] > 0.0 else 1.0

    return torch.clip(img_wb, min=0.0, max=1.0)
