import torch
import rawpy
import numpy as np


def mono_to_rgb(
    normalized_image: torch.Tensor, bayer_pattern: torch.Tensor
) -> torch.Tensor:
    """
    the docstring ...
    """
    height, width = normalized_image.shape
    # 2. Initialize the output array (H, W, 3) with zeros
    rgb_image = torch.zeros((height, width, 3), dtype=normalized_image.dtype)
    # 3. Handle the 4-color Bayer case (RGGB usually has indices 0, 1, 2, 3)
    # We map index 3 (Green2) to index 1 (Green) so it fits in a 3-channel RGB image.
    # We use a copy to avoid modifying the original input array.
    channel_indices = torch.clone(bayer_pattern)
    channel_indices[channel_indices == 3] = 1
    # 4. Use Advanced Indexing (Vectorization)
    # We create coordinate grids to assign all values at once.
    # rows[:, None] creates a column vector (H, 1)
    # np.arange(width) creates a row vector (1, W)
    # NumPy broadcasts these against 'channel_indices' (H, W) to select exact targets.
    rows = torch.arange(height)[:, None]
    cols = torch.arange(width)
    # This single line replaces the nested for-loops
    rgb_image[rows, cols, channel_indices] = normalized_image
    return rgb_image


def read_raw(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reads a RAW file and returns the Bayer mosaic as a normalized PyTorch tensor.

    Args:
        path (str): Path to the RAW file.

    Returns:
        - bayer_img: torch.Tensor [H, W] normalized to [0, 1] (float32)
        - pattern: torch.Tensor [H, W] showing the CFA layout (int32)
        - wb_gains: torch.Tensor [4] containing the camera white balance multipliers
    """
    with rawpy.imread(path) as raw:
        img = raw.raw_image.copy().astype(np.float32)
        bayer = raw.raw_colors.copy()

        black_levels = raw.black_level_per_channel
        white_level = raw.white_level

        # Normalization Step --
        for i in range(len(black_levels)):
            mask = bayer == i
            bl = black_levels[i]
            img[mask] = (img[mask] - bl) / (white_level - bl)
        img = np.clip(img, 0.0, 1.0)
        return (
            torch.from_numpy(img),
            torch.from_numpy(bayer).int(),
            torch.tensor(raw.camera_whitebalance, dtype=torch.float32),
        )
