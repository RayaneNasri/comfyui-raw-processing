import torch


def correct_vignetting(
    image: torch.Tensor,
    alpha: float,
    beta: float,
    gain_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply vignetting correction to a linear RGB image.

    If a pre-computed gain map is provided it is used directly. Otherwise a radial polynomial model is evaluated: gain(r²) = 1 + alpha * r² + beta * r⁴, where r² is the squared distance from the image centre, normalised so that the farthest corner has r² = 1. Positive alpha/beta brighten the edges relative to the centre, correcting the typical lens fall-off.

    Args:
        image (torch.Tensor): Linear RGB image tensor of shape (H, W, 3) in the range [0, 1].
        alpha (float): Quadratic gain coefficient. 0 indicates no correction.
        beta (float): Quartic gain coefficient. 0 indicates no correction.
        gain_map (torch.Tensor | None, optional): Per-pixel multiplicative gain map of shape (H, W, 3). When provided, alpha and beta are ignored.

    Returns:
        torch.Tensor: Corrected image of shape (H, W, 3), clamped to [0, 1].
    """
    if gain_map is not None:
        return torch.clamp(image * gain_map, 0.0, 1.0)

    if alpha == 0.0 and beta == 0.0:
        return image

    H, W, _ = image.shape
    y = torch.linspace(-1.0, 1.0, H, device=image.device)
    x = torch.linspace(-1.0, 1.0, W, device=image.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r2 = xx**2 + yy**2
    r2_norm = r2 / r2.max().clamp(min=1e-6)  # 1.0 at the farthest corner

    gain = 1.0 + alpha * r2_norm + beta * r2_norm**2
    return torch.clamp(image * gain.unsqueeze(-1), 0.0, 1.0)
