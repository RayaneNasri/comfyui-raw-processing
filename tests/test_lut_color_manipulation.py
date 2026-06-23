import torch

from algorithms.color_manipulation._lut_color_manipulation import apply_lut_grid_sample


def test_dtype_preserved():
    img = torch.rand(3, 3, 3, dtype=torch.float32)
    lut = torch.ones((2, 2, 2, 3), dtype=torch.float32)
    out = apply_lut_grid_sample(img, lut)
    assert out.dtype == img.dtype


def make_identity_lut(size: int):
    values = torch.linspace(0.0, 1.0, size)

    r, g, b = torch.meshgrid(values, values, values, indexing="ij")

    return torch.stack((r, g, b), dim=-1)


def make_invert_lut(size: int):
    values = torch.linspace(0.0, 1.0, size)

    r, g, b = torch.meshgrid(values, values, values, indexing="ij")

    return torch.stack((1.0 - r, 1.0 - g, 1.0 - b), dim=-1)


def test_shape_preserved():
    image = torch.rand((32, 64, 3))
    lut = torch.rand((17, 17, 17, 3))

    result = apply_lut_grid_sample(image, lut)

    assert result.shape == image.shape


def test_identity_lut():
    image = torch.rand((16, 16, 3))

    lut = make_identity_lut(17)

    result = apply_lut_grid_sample(image, lut)

    assert torch.allclose(result, image, atol=1e-3)


def test_constant_red_lut():
    image = torch.rand((10, 10, 3))

    lut = torch.zeros((17, 17, 17, 3))
    lut[..., 0] = 1.0

    result = apply_lut_grid_sample(image, lut)

    expected = torch.zeros_like(image)
    expected[..., 0] = 1.0

    assert torch.allclose(result, expected, atol=1e-4)


def test_black_image_identity():
    image = torch.zeros((8, 8, 3))

    lut = make_identity_lut(17)

    result = apply_lut_grid_sample(image, lut)

    assert torch.allclose(result, image, atol=1e-4)


def test_white_image_identity():
    image = torch.ones((8, 8, 3))

    lut = make_identity_lut(17)

    result = apply_lut_grid_sample(image, lut)

    assert torch.allclose(result, image, atol=1e-4)


def test_invert_lut():
    image = torch.rand((10, 10, 3))

    lut = make_invert_lut(17)

    result = apply_lut_grid_sample(image, lut)

    expected = 1.0 - image

    assert torch.allclose(result, expected, atol=5e-2)


def test_output_range():
    image = torch.rand((20, 20, 3))
    lut = torch.rand((17, 17, 17, 3))

    result = apply_lut_grid_sample(image, lut)

    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)


def test_equal_pixels_produce_equal_results():
    image = torch.tensor([[[0.2, 0.4, 0.8], [0.2, 0.4, 0.8]]])

    lut = torch.rand((17, 17, 17, 3))

    result = apply_lut_grid_sample(image, lut)

    assert torch.allclose(result[0, 0], result[0, 1])


def test_identity_lut_corners():
    lut = make_identity_lut(17)

    image = torch.tensor(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0]],
            [[1.0, 1.0, 1.0]],
        ]
    )

    result = apply_lut_grid_sample(image, lut)

    assert torch.allclose(result, image, atol=1e-4)


def test_channel_order():
    image = torch.tensor([[[1.0, 0.0, 0.0]]])

    lut = torch.zeros((17, 17, 17, 3))

    values = torch.linspace(0.0, 1.0, 17)

    r, g, b = torch.meshgrid(values, values, values, indexing="ij")

    lut[..., 0] = b
    lut[..., 1] = g
    lut[..., 2] = r

    result = apply_lut_grid_sample(image, lut)

    expected = torch.tensor([[[0.0, 0.0, 1.0]]])

    assert torch.allclose(result, expected, atol=1e-2)
