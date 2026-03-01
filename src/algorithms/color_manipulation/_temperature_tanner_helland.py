import torch
from torch import Tensor
import math


def kelvin_to_rgb_tanner_helland(temperature_K : float) -> tuple[float, float, float]:
    """
    temperature_K : temperature in Kelvin, between 1000 K and 40000 K
        (the interesting photographic range, which is 1500 K to 15000 K)
        The white point occurs at 6500-6600 K.
    """
    
    temperature_K = temperature_K / 100

    # Red channel

    if temperature_K < 66:
        coeff_r = 255.0

    else:
        coeff_r = 329.698727446 * ((temperature_K - 60) ** -0.1332047592)
        if coeff_r < 0:
            coeff_r = 0.0
        if coeff_r > 255:
            coeff_r = 255.0

    # Green channel

    if temperature_K < 66:
        coeff_g = 99.4708025861 * math.log(temperature_K) - 161.1195681661
        if coeff_g < 0:
            coeff_g = 0.0
        if coeff_g > 255:
            coeff_g = 255.0
    else:
        coeff_g = 288.1221695283 * ((temperature_K - 60) ** -0.0755148492)
        if coeff_g < 0:
            coeff_g = 0.0
        if coeff_g > 255:
            coeff_g = 255.0

    # Blue channel

    if temperature_K >= 66:
        coeff_b = 255.0
    elif temperature_K <= 19:
        coeff_b = 0.0
    else:
        coeff_b = 138.5177312231 * math.log(temperature_K - 10) - 305.0447927307
        if coeff_b < 0:
            coeff_b = 0.0
        if coeff_b > 255:
            coeff_b = 255.0

    return coeff_r, coeff_g, coeff_b 



def temperature_tanner_helland(rgb_image : Tensor, temperature_K : float) -> Tensor:
    
    # # Rescales color intensities from normalized float range [0,1] to float range [0,255]
    # rgb_image_255 = rgb_image * 255.0
    pass