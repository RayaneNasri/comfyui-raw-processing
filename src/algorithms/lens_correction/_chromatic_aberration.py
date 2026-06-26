import torch
import torch.nn.functional as F


def correct_chromatic_aberration(
    image: torch.Tensor,
    red_scale: float,
    blue_scale: float,
) -> torch.Tensor:
    """
    Correct lateral chromatic aberration by scaling R and B channels around the image centre to align them with the G channel.

    Lateral chromatic aberration (CA) appears because the lens refracts different wavelengths by slightly different amounts, causing colour fringing at high-contrast edges. The fix is a small per-channel magnification/minification implemented via an affine grid_sample, which is GPU-compatible and differentiable.

    Convention:
        red_scale > 1.0  → R channel is magnified (zoomed in).
        red_scale < 1.0  → R channel is minified (zoomed out).
        G channel is always left unchanged.

    Args:
        image (torch.Tensor): Linear RGB image tensor of shape (H, W, 3) in the range [0, 1].
        red_scale (float): Isotropic scale factor for the R channel. 1.0 indicates no change.
        blue_scale (float): Isotropic scale factor for the B channel. 1.0 indicates no change.

    Returns:
        torch.Tensor: Corrected image of shape (H, W, 3), clamped to [0, 1].
    """
    result = image.clone()

    for ch_idx, scale in ((0, red_scale), (2, blue_scale)):
        if abs(scale - 1.0) < 1e-6:
            continue

        channel = image[..., ch_idx].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        # invert the scale, grid_sample maps output coords to input coords,
        # so s = 1/scale magnifies the channel when scale>1.
        s = 1.0 / scale
        theta = torch.tensor(
            [[s, 0.0, 0.0], [0.0, s, 0.0]],
            dtype=torch.float32,
        ).unsqueeze(0)

        grid = F.affine_grid(theta, channel.shape, align_corners=False)  # type: ignore
        scaled = F.grid_sample(
            channel, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        result[..., ch_idx] = scaled.squeeze(0).squeeze(0)

    return torch.clamp(result, 0.0, 1.0)
