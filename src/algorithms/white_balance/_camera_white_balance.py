import torch


def raw_wb_gains_to_rgb(wb_gains: torch.Tensor) -> torch.Tensor:
    """
    Convert RAW white-balance gains to RGB gains.

    Args:
        wb_gains (torch.Tensor): Camera white-balance gains from RAW metadata.
            Supported layouts are length 3 ``(R, G, B)`` and length 4
            ``(R, G1, B, G2)``.

    Returns:
        torch.Tensor: Tensor of shape ``(3,)`` containing ``(R, G, B)`` gains.

    Raises:
        ValueError: If ``wb_gains`` contains fewer than 3 values.
    """
    gains = wb_gains.reshape(-1)
    if gains.numel() < 3:
        raise ValueError(
            f"wb_gains must contain at least 3 values, got {gains.numel()}"
        )

    if gains.numel() == 3:
        return gains[:3]

    red_gain = gains[0]
    green_1 = gains[1]
    blue_gain = gains[2]
    green_2 = gains[3]

    g1_positive = bool((green_1 > 0).item())
    g2_positive = bool((green_2 > 0).item())

    if g1_positive and g2_positive:
        green_gain = 0.5 * (green_1 + green_2)
    elif g1_positive:
        green_gain = green_1
    elif g2_positive:
        green_gain = green_2
    else:
        green_gain = green_1

    return torch.stack((red_gain, green_gain, blue_gain))


def camera_white_balance(
    img: torch.Tensor,
    wb_gains: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """
    Apply camera white balance to an RGB image.

    Args:
        img (torch.Tensor): Input image tensor with shape ``(H, W, 3)``.
        wb_gains (torch.Tensor): Camera white-balance gains from RAW metadata.
            Accepted as length 3 ``(R, G, B)`` or length 4 ``(R, G1, B, G2)``.
        strength (float): Blend factor in ``[0, +inf)``. ``0.0`` keeps the
            original image, ``1.0`` applies the full camera white balance.

    Returns:
        torch.Tensor: White-balanced image tensor with shape ``(H, W, 3)``,
        clamped to ``[0, 1]``.

    Raises:
        ValueError: If ``img`` does not have shape ``(H, W, 3)``.
        ValueError: If ``strength`` is negative.
    """
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {tuple(img.shape)}")

    if strength < 0.0:
        raise ValueError(f"strength must be >= 0.0, got {strength}")

    gains = wb_gains.to(dtype=img.dtype, device=img.device)
    gains_rgb = raw_wb_gains_to_rgb(gains)

    eps = torch.tensor(1e-6, dtype=img.dtype, device=img.device)
    normalized_gains = gains_rgb / torch.clamp(gains_rgb[1], min=eps)

    scales = 1.0 + (normalized_gains - 1.0) * strength
    scales = scales.view(1, 1, 3)

    out = img.clone()
    out = out * scales

    return torch.clamp(out, min=0.0, max=1.0)
