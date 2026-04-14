import torch
from torch import Tensor

from algorithms.color_manipulation._saturation_hsv import saturation_hsv

def test_dtype_preserved():
    img = torch.rand(3, 3, 3, dtype=torch.float32)
    out = saturation_hsv(img, 1.4)
    assert out.dtype == img.dtype