import torch

from algorithms.color_manipulation._temperature_tanner_helland import (
    temperature_tanner_helland,
)


def test_dtype_preserved():
    img = torch.rand(3, 3, 3, dtype=torch.float32)
    out = temperature_tanner_helland(img, 6600.0)
    assert out.dtype == img.dtype


def test_shape_is_preserved():
    img = torch.rand((32, 64, 3))

    result = temperature_tanner_helland(img, 6500)

    assert result.shape == img.shape


def test_values_are_clamped():
    img = torch.rand((32, 64, 3))

    result = temperature_tanner_helland(img, 1000)

    assert torch.all(result >= 0)
    assert torch.all(result <= 1)


def test_white_point_preserves_gray():
    img = torch.full((10, 10, 3), 0.5)

    result = temperature_tanner_helland(img, 6500)

    assert torch.allclose(result[..., 0], result[..., 1], atol=1e-2)

    assert torch.allclose(result[..., 1], result[..., 2], atol=1e-2)


def test_warm_temperature_increases_red_relative_to_blue():
    img = torch.full((1, 1, 3), 0.5)

    result = temperature_tanner_helland(img, 2000)

    red = result[..., 0]
    blue = result[..., 2]

    assert red > blue


def test_cold_temperature_increases_blue_relative_to_red():
    img = torch.full((1, 1, 3), 0.5)

    result = temperature_tanner_helland(img, 15000)

    red = result[..., 0]
    blue = result[..., 2]

    assert blue > red


def test_warm_and_cold_temperatures_give_different_results():
    img = torch.full((10, 10, 3), 0.5)

    warm = temperature_tanner_helland(img, 2000)
    cold = temperature_tanner_helland(img, 15000)

    assert not torch.allclose(warm, cold)


def test_black_image_remains_black():
    img = torch.zeros((10, 10, 3))

    result = temperature_tanner_helland(img, 6500)

    assert torch.allclose(result, img)


def test_white_image_stays_valid():
    img = torch.ones((10, 10, 3))

    result = temperature_tanner_helland(img, 2000)

    assert torch.all(result >= 0)
    assert torch.all(result <= 1)


def test_extreme_temperatures_are_valid():
    img = torch.rand((20, 20, 3))

    cold = temperature_tanner_helland(img, 40000)
    warm = temperature_tanner_helland(img, 1000)

    assert torch.all(cold >= 0)
    assert torch.all(cold <= 1)

    assert torch.all(warm >= 0)
    assert torch.all(warm <= 1)


def test_tanner_helland_color_ordering():
    img = torch.full((1, 1, 3), 0.5)

    warm = temperature_tanner_helland(img, 2000)
    neutral = temperature_tanner_helland(img, 6500)
    cold = temperature_tanner_helland(img, 15000)

    assert warm[0, 0, 0] > warm[0, 0, 1] > warm[0, 0, 2]

    assert abs(neutral[0, 0, 0] - neutral[0, 0, 1]) < 1e-2
    assert abs(neutral[0, 0, 1] - neutral[0, 0, 2]) < 1e-2

    assert cold[0, 0, 2] > cold[0, 0, 1] > cold[0, 0, 0]
