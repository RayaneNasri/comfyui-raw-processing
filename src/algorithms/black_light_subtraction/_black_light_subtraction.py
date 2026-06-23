import torch


def linearize_raw(
    raw_img: torch.Tensor,
    bayer_pattern: torch.Tensor,
    black_levels: torch.Tensor,
    white_level: float | torch.Tensor,
) -> torch.Tensor:
    """
    Apply black-level subtraction and linear normalization per CFA channel.

    Args:
        raw_img (torch.Tensor [H, W]): Raw sensor values.
        bayer_pattern (torch.Tensor [H, W]): CFA channel indices {0,1,2,3}.
        black_levels (torch.Tensor [4]): Black level per channel.
        white_level (float or torch.Tensor [1]): Global white level.

    Returns:
        linear_img (torch.Tensor [H, W]): Linearized image in [0, 1].
    """
    raw_img = raw_img.float()
    bayer_pattern = bayer_pattern.long()
    black_levels = black_levels.float().view(-1)

    if isinstance(white_level, torch.Tensor):
        white_level = float(white_level.squeeze().item())

    black_map = black_levels[bayer_pattern]
    denom = white_level - black_map
    linear_img = (raw_img - black_map) / denom

    return linear_img.clamp(0.0, 1.0)
