import numpy as np  

def mono_to_rgb(normalized_image: np.ndarray, bayer_pattern: np.ndarray)-> np.ndarray:
    """
    the docstring ...
    """
    height, width = normalized_image.shape
    # 2. Initialize the output array (H, W, 3) with zeros
    rgb_image = np.zeros((height, width, 3), dtype=normalized_image.dtype)
    # 3. Handle the 4-color Bayer case (RGGB usually has indices 0, 1, 2, 3)
    # We map index 3 (Green2) to index 1 (Green) so it fits in a 3-channel RGB image.
    # We use a copy to avoid modifying the original input array.
    channel_indices = bayer_pattern.copy()
    channel_indices[channel_indices == 3] = 1
    # 4. Use Advanced Indexing (Vectorization)
    # We create coordinate grids to assign all values at once.
    # rows[:, None] creates a column vector (H, 1)
    # np.arange(width) creates a row vector (1, W)
    # NumPy broadcasts these against 'channel_indices' (H, W) to select exact targets.
    rows = np.arange(height)[:, None]
    cols = np.arange(width)
    # This single line replaces the nested for-loops
    rgb_image[rows, cols, channel_indices] = normalized_image
    return rgb_image

def bilinear_demosaicing(rgb_image: np.ndarray)-> np.ndarray:
    """
    the docstring ...
    """
    pass