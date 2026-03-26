import torch


def gamma_correction(
    img: torch.Tensor, gamma: float = 2.2, alpha: float = 1.0
) -> torch.Tensor:
    """
    Perform gamma correction on an image.

    This function applies a non-linear power-law transformation (gamma correction)
    to the input image, followed by an optional scaling factor and clipping to [0.0, 1.0].

    Args:
        img (torch.Tensor): The input image tensor to be corrected.
        gamma (float, optional): The gamma value used for correction. The image is
            raised to the power of 1/gamma. Must be strictly positive. Defaults to 2.2.
        alpha (float, optional): A multiplier scaling factor applied after the power
            transformation. Must be non-negative. Defaults to 1.0.

    Returns:
        torch.Tensor: The gamma-corrected image tensor, clipped to the range [0.0, 1.0].

    Raises:
        TypeError: If `img` is not a `torch.Tensor`.
        ValueError: If `gamma` is less than or equal to 0.0.
        ValueError: If `alpha` is less than 0.0.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError("img must be a torch Tensor")
    if gamma <= 0.0:
        raise ValueError("gamma must be strictly positive")
    if alpha < 0.0:
        raise ValueError("alpha must be positive")

    eps = 10 ** (-6)
    return torch.clip(alpha * (img ** (1 / (gamma + eps))), min=0.0, max=1.0)
