import torch


def mono_to_rgb(
    normalized_image: torch.Tensor, bayer_pattern: torch.Tensor
) -> torch.Tensor:
    """
    Convert a single-channel Bayer image into a sparse RGB image.
    """
    height, width = normalized_image.shape
    rgb_image = torch.zeros((height, width, 3), dtype=normalized_image.dtype)

    channel_indices = torch.clone(bayer_pattern)
    channel_indices[channel_indices == 3] = 1

    rows = torch.arange(height)[:, None]
    cols = torch.arange(width)
    rgb_image[rows, cols, channel_indices] = normalized_image
    return rgb_image
