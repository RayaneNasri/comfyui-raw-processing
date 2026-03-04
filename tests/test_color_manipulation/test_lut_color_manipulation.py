import torch
import pytest 

from algorithms.color_manipulation._lut_color_manipulation import load_cube_lut
from algorithms.color_manipulation._lut_color_manipulation import apply_lut_trilinear_interpolation

def test_dtype_preserved():
    img = torch.rand(3, 3, 3, dtype=torch.float32)
    lut = load_cube_lut("/mnt/c/Users/charl/git/Blues.cube") # TODO : change path
    out = apply_lut_trilinear_interpolation(img, lut)
    assert out.dtype == img.dtype