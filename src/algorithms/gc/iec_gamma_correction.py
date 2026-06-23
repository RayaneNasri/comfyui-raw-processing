import torch


def iec_gamma_correction(img: torch.Tensor) -> torch.Tensor:
    img = img.clamp_(0.0, 1.0)
    linear = img * 12.92
    gamma = 1.055 * torch.pow(img.clamp(min=1e-10), 1.0 / 2.4) - 0.055

    return torch.where(img <= 0.0031308, linear, gamma).clamp_(0.0, 1.0)
