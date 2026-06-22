import torch
import pytest 

from algorithms.color_manipulation._temperature_simple import temperature_simple

def test_dtype_preserved():
    img = torch.rand(3, 3, 3, dtype=torch.float32)
    out = temperature_simple(img, 10.0)
    assert out.dtype == img.dtype

def test_temperature_zero():
    img = torch.tensor([[[0.2, 0.5, 0.8]]])

    result = temperature_simple(img, 0)

    expected = img
    assert torch.allclose(result, expected)


def test_temperature_positive():
    img = torch.tensor([[[0.2, 0.5, 0.8]]])

    # 25.5 / 255 = 0.1
    result = temperature_simple(img, 25.5)

    expected = torch.tensor([[[0.3, 0.5, 0.7]]])
    assert torch.allclose(result, expected)


def test_temperature_maximum():
    img = torch.tensor([[[0.2, 0.5, 0.8]]])

    # 255 / 255 = 1.0
    result = temperature_simple(img, 255)

    expected = torch.tensor([[[1.0, 0.5, 0.0]]])
    assert torch.allclose(result, expected)


def test_clamp_upper_bound_red():
    img = torch.tensor([[[0.95, 0.5, 0.4]]])

    # 51 / 255 = 0.2
    result = temperature_simple(img, 51)

    expected = torch.tensor([[[1.0, 0.5, 0.2]]])
    assert torch.allclose(result, expected)


def test_clamp_lower_bound_blue():
    img = torch.tensor([[[0.4, 0.5, 0.1]]])

    # 51 / 255 = 0.2
    result = temperature_simple(img, 51)

    expected = torch.tensor([[[0.6, 0.5, 0.0]]])
    assert torch.allclose(result, expected)


def test_green_channel_is_unchanged():
    img = torch.tensor([
        [[0.2, 0.1, 0.8],
         [0.3, 0.7, 0.6]]
    ])

    result = temperature_simple(img, 64)

    assert torch.allclose(result[..., 1], img[..., 1])


def test_multiple_pixels():
    img = torch.tensor([
        [[0.2, 0.5, 0.8],
         [0.4, 0.6, 0.3]]
    ])

    # 25.5 / 255 = 0.1
    result = temperature_simple(img, 25.5)

    expected = torch.tensor([
        [[0.3, 0.5, 0.7],
         [0.5, 0.6, 0.2]]
    ])

    assert torch.allclose(result, expected)


def test_output_values_stay_in_range():
    img = torch.rand((20, 20, 3))

    result = temperature_simple(img, 255)

    assert torch.all(result >= 0)
    assert torch.all(result <= 1)