import torch


def iec_gamma_correction(img: torch.Tensor) -> torch.Tensor:
    """
    Apply IEC gamma correction to convert linear RGB values to sRGB.

    Args:
        img (torch.Tensor): Input image tensor with linear RGB values in the range [0, 1].

    Returns:
        torch.Tensor: Gamma-corrected image tensor in sRGB color space, clamped to [0, 1].
    """
    img = img.clamp_(0.0, 1.0)
    linear = img * 12.92
    gamma = 1.055 * torch.pow(img.clamp(min=1e-10), 1.0 / 2.4) - 0.055

    return torch.where(img <= 0.0031308, linear, gamma).clamp_(0.0, 1.0)
