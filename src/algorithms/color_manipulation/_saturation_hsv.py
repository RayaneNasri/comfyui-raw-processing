import torch
from torch import Tensor

from algorithms.tools._lut_tools import rgb_to_hsv, hsv_to_rgb

def saturation_hsv(rgb_image : Tensor, adjustement_value : float):
    hsv_image = rgb_to_hsv(rgb_image)
    hsv_image[:,:,1] *= adjustement_value
    hsv_image[:,:,1] = hsv_image[:,:,1].clamp(0, 1)
    res = hsv_to_rgb(hsv_image)
    return res