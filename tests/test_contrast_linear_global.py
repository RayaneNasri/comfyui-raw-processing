import torch

from algorithms.color_manipulation._contrast_linear_global import contrast_linear_global


def test_shape_preserved():
    image = torch.rand((32, 64, 3))

    result = contrast_linear_global(image, 1.5)

    assert result.shape == image.shape


def test_factor_one_identity():
    image = torch.rand((32, 64, 3))

    result = contrast_linear_global(image, 1.0)

    assert torch.allclose(result, image, atol=1e-6)


def test_global_mean_preserved():
    image = torch.rand((64, 64, 3))

    mean_before = image.mean()

    result = contrast_linear_global(image, 1.8)

    mean_after = result.mean()

    assert torch.allclose(mean_before, mean_after, atol=1e-2)


def test_zero_contrast():
    image = torch.rand((16, 16, 3))

    mean = image.mean()

    result = contrast_linear_global(image, 0.0)

    expected = torch.full_like(image, mean.item())

    assert torch.allclose(result, expected, atol=1e-5)


def test_increase_contrast_increases_variance():
    image = torch.rand((64, 64, 3))

    std_before = image.std()

    result = contrast_linear_global(image, 2.0)

    std_after = result.std()

    assert std_after > std_before


def test_decrease_contrast_decreases_variance():
    image = torch.rand((64, 64, 3))

    std_before = image.std()

    result = contrast_linear_global(image, 0.5)

    std_after = result.std()

    assert std_after < std_before


def test_two_pixel_example_factor_two():
    image = torch.tensor([[[0.25, 0.25, 0.25]], [[0.75, 0.75, 0.75]]])

    result = contrast_linear_global(image, 2.0)

    expected = torch.tensor([[[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]]])

    assert torch.allclose(result, expected, atol=1e-6)


def test_two_pixel_example_factor_half():
    image = torch.tensor([[[0.25, 0.25, 0.25]], [[0.75, 0.75, 0.75]]])

    result = contrast_linear_global(image, 0.5)

    expected = torch.tensor([[[0.375, 0.375, 0.375]], [[0.625, 0.625, 0.625]]])

    assert torch.allclose(result, expected, atol=1e-6)


def test_uniform_image_unchanged():
    image = torch.full((32, 32, 3), 0.42)

    result = contrast_linear_global(image, 2.0)

    assert torch.allclose(result, image, atol=1e-6)


def test_black_image_unchanged():
    image = torch.zeros((32, 32, 3))

    result = contrast_linear_global(image, 2.0)

    assert torch.allclose(result, image, atol=1e-6)


def test_white_image_unchanged():
    image = torch.ones((32, 32, 3))

    result = contrast_linear_global(image, 2.0)

    assert torch.allclose(result, image, atol=1e-6)


def test_output_range():
    image = torch.rand((64, 64, 3))

    result = contrast_linear_global(image, 2.0)

    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)
