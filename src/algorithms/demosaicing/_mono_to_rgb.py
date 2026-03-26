import torch


def mono_to_rgb(
    normalized_image: torch.Tensor, bayer_pattern: torch.Tensor
) -> torch.Tensor:
    """
    Convert a single-channel Bayer image into a sparse RGB image.

    This function maps the pixels of a single-channel Bayer image to their 
    corresponding color channels in a 3D RGB tensor, leaving the missing color 
    channels for each pixel as zeros.

    Args:
        normalized_image (torch.Tensor [H, W]): The 2D single-channel Bayer image 
            containing normalized sensor values.
        bayer_pattern (torch.Tensor [H, W]): A 2D tensor of the same spatial dimensions 
            containing the Bayer channel indices (typically {0, 1, 2, 3}, where 1 and 3 
            both represent the Green channel).

    Returns:
        torch.Tensor [H, W, 3]: A sparse RGB image where each pixel has a non-zero 
            value only in the channel corresponding to its Bayer pattern index.
    """
    height, width = normalized_image.shape
    rgb_image = torch.zeros((height, width, 3), dtype=normalized_image.dtype)

    channel_indices = torch.clone(bayer_pattern)
    channel_indices[channel_indices == 3] = 1

    rows = torch.arange(height)[:, None]
    cols = torch.arange(width)
    rgb_image[rows, cols, channel_indices] = normalized_image
    return rgb_image
