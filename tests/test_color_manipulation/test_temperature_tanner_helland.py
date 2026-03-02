import torch
import pytest 

from algorithms.color_manipulation._temperature_tanner_helland import temperature_tanner_helland


def test_dtype_preserved():
    img = torch.rand(3, 3, 3, dtype=torch.float32)
    out = temperature_tanner_helland(img, 6600.0)
    assert out.dtype == img.dtype
