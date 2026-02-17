import numpy as np
import rawpy

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

def read_raw_as_ndarray(image_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads a RAW file and returns the Bayer mosaic as a normalized NumPy array.
    
    Returns:
        - bayer_img: ndarray [H, W] normalized to [0, 1]
        - pattern: ndarray [2, 2] showing the CFA layout (e.g., RGGB)
        - wb_gains: ndarray [4] containing the camera white balance multipliers
    """
    with rawpy.imread(image_path) as raw:
        bayer_img = raw.raw_image.copy().astype(np.float32)
        black = raw.black_level_per_channel[0] 
        white = raw.white_level
        bayer_img = (bayer_img - black) / (white - black)
        bayer_img = np.clip(bayer_img, 0, 1)
        pattern = raw.raw_pattern
        wb_gains = np.array(raw.camera_whitebalance[:4], dtype=np.float32)
        
        return bayer_img, pattern, wb_gains