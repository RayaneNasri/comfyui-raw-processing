import torch
import pytest 

from algorithms.color_manipulation._temperature_simple import temperature_simple

def test_adjustement_zero_returns_same_image():
    img = torch.rand(3, 3, 3)
    out = temperature_simple(img, 0.0)
    assert torch.allclose(out, img)

def test_dtype_preserved():
    img = torch.rand(3, 3, 3, dtype=torch.float32)
    out = temperature_simple(img, 10.0)
    assert out.dtype == img.dtype