import torch

from torch import Tensor


def exposure_compensation(rgb_image: Tensor, ev_compensation: float) -> Tensor:
    gain = 2.0**ev_compensation
    result = rgb_image * gain

    return result
