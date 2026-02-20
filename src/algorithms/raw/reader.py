import numpy as np
import rawpy
import torch


def read_raw_sensor_data(
    path: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Read RAW sensor data and metadata.

    Args:
        path (str): Path to RAW file.

    Returns:
        tuple: A tuple containing:
            raw_img: torch.Tensor [H, W], raw sensor values (float32)
            bayer_pattern: torch.Tensor [H, W], CFA channel indices in {0,1,2,3} (int32)
            black_levels: torch.Tensor [4], per-channel black levels (float32)
            white_level: torch.Tensor [1], sensor white level (float32)
            wb_gains: torch.Tensor [4], camera white-balance multipliers (float32)
    """
    with rawpy.imread(path) as raw:
        raw_img = torch.from_numpy(raw.raw_image.copy().astype(np.float32))
        bayer_pattern = torch.from_numpy(raw.raw_colors.copy()).int()

        black_levels = torch.tensor(raw.black_level_per_channel, dtype=torch.float32)
        white_level = torch.tensor([raw.white_level], dtype=torch.float32)
        wb_gains = torch.tensor(raw.camera_whitebalance, dtype=torch.float32)

    return raw_img, bayer_pattern, black_levels, white_level, wb_gains
