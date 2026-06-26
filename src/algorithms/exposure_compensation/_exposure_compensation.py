from torch import Tensor


def exposure_compensation(rgb_image: Tensor, ev_compensation: float) -> Tensor:
    """
    Apply global exposure compensation to an image.

    This function adjusts the exposure of an image by applying a gain factor
    calculated from the specified Exposure Value (EV) compensation. An EV of +1.0
    doubles the brightness, while an EV of -1.0 halves it.

    Args:
        rgb_image (Tensor): The input image tensor (typically linear RGB).
        ev_compensation (float): The exposure compensation value in stops (EV). Positive values brighten the image, and negative values darken it.

    Returns:
        Tensor: The exposure-adjusted image tensor, scaled by 2.0^ev_compensation.
    """
    gain = 2.0**ev_compensation
    result = rgb_image * gain

    return result
