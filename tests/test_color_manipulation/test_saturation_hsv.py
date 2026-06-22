import torch
from torch import Tensor

from algorithms.color_manipulation._saturation_hsv import saturation_hsv

def test_dtype_preserved():
    img = torch.rand(3, 3, 3, dtype=torch.float32)
    out = saturation_hsv(img, 1.4)
    assert out.dtype == img.dtype

def test_saturation_factor_one():
    img = torch.tensor([
        [[1.0, 0.0, 0.0],
         [0.2, 0.4, 0.6]]
    ])

    result = saturation_hsv(img, 1.0)

    assert torch.allclose(result, img, atol=1e-6)

def test_saturation_zero_produces_grayscale():
    img = torch.tensor([[[1.0, 0.0, 0.0]]])

    result = saturation_hsv(img, 0.0)

    r, g, b = result[0, 0]

    assert torch.allclose(r, g, atol=1e-6)
    assert torch.allclose(g, b, atol=1e-6)

def test_gray_pixel_is_unchanged():
    img = torch.tensor([[[0.5, 0.5, 0.5]]])

    result = saturation_hsv(img, 5.0)

    assert torch.allclose(result, img, atol=1e-6)

def test_saturation_increase():
    img = torch.tensor([[[0.8, 0.6, 0.6]]])

    before_spread = img.max() - img.min()

    result = saturation_hsv(img, 2.0)

    after_spread = result.max() - result.min()

    assert after_spread > before_spread

def test_saturation_decrease():
    img = torch.tensor([[[1.0, 0.0, 0.0]]])

    before_spread = img.max() - img.min()

    result = saturation_hsv(img, 0.5)

    after_spread = result.max() - result.min()

    assert after_spread < before_spread

def test_saturation_clamped():
    img = torch.tensor([[[1.0, 0.0, 0.0]]])

    result = saturation_hsv(img, 5.0)

    assert torch.all(result >= 0)
    assert torch.all(result <= 1)

def test_shape_preserved():
    img = torch.rand((32, 64, 3))

    result = saturation_hsv(img, 2.0)

    assert result.shape == img.shape

def test_output_range():
    img = torch.rand((32, 64, 3))

    result = saturation_hsv(img, 5.0)

    assert torch.all(result >= 0)
    assert torch.all(result <= 1)

def test_multiple_pixels():
    img = torch.tensor([
        [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0],
         [0.5, 0.5, 0.5]]
    ])

    result = saturation_hsv(img, 0.5)

    assert result.shape == img.shape
    assert torch.all(result >= 0)
    assert torch.all(result <= 1)

def test_pure_red_unchanged_for_factor_one():
    img = torch.tensor([[[1.0, 0.0, 0.0]]])

    result = saturation_hsv(img, 1.0)

    expected = torch.tensor([[[1.0, 0.0, 0.0]]])

    assert torch.allclose(result, expected, atol=1e-6)