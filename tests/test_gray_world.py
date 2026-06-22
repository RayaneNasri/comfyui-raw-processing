import torch

from torch.testing import assert_close
from algorithms.white_balance import gray_world


def test_preserves_shape():
    """Checks that the output image has the same dimensions as the input."""
    # Format: H, W, C
    img = torch.rand(100, 100, 3)
    out = gray_world(img)
    assert out.shape == img.shape, "The tensor shape has changed!"


def test_already_gray_balanced():
    """
    If the image already has the exact same mean for all three channels
    (i.e., it's perfectly balanced), the algorithm should not modify it.
    """
    img = torch.rand(50, 50, 1).repeat(1, 1, 3)
    out = gray_world(img)
    assert_close(out, img, msg="An already balanced image should not be modified.")


def test_color_correction():
    """
    Tests the core theory of Gray World: after processing,
    all color channels must share the exact same mean value.
    """
    img = torch.ones(10, 10, 3)
    img[..., 0] *= 0.8  # Red channel mean is 0.8
    img[..., 1] *= 0.4  # Green channel mean is 0.4
    img[..., 2] *= 0.2  # Blue channel mean is 0.2

    out = gray_world(img)

    mean_r = out[..., 0].mean()
    mean_g = out[..., 1].mean()
    mean_b = out[..., 2].mean()

    assert torch.allclose(mean_r, mean_g), (
        f"Red ({mean_r}) and Green ({mean_g}) means differ!"
    )
    assert torch.allclose(mean_g, mean_b), (
        f"Green ({mean_g}) and Blue ({mean_b}) means differ!"
    )


def test_black_pixels_preserved():
    """Checks the physical property of multiplication: black remains black."""
    img = torch.ones(10, 10, 3) * 0.5
    img[5, 5, :] = 0.0

    out = gray_world(img)

    expected_pixel = torch.tensor([0.0, 0.0, 0.0])
    assert_close(
        out[5, 5, :], expected_pixel, msg="A black pixel must remain black (0.0)."
    )


def test_zero_mean_channel_no_nan():
    """
    Tests that the algorithm does not crash or produce NaNs
    if one specific channel is entirely empty (mean = 0).
    """
    # Image with Red and Green, but absolutely NO Blue
    img = torch.ones(10, 10, 3) * 0.5
    img[..., 2] = 0.0

    out = gray_world(img)

    # Check that there are no NaN (Not a Number) or Inf values
    assert not torch.isnan(out).any(), (
        "The output contains NaN! Check for division by zero."
    )
    assert not torch.isinf(out).any(), (
        "The output contains Infinity! Check for division by zero."
    )

    # The blue channel should remain logically black
    assert torch.all(out[..., 2] == 0.0), "The empty channel should remain at 0.0"


def test_totally_black_image():
    """
    Tests the absolute edge case: a completely black image.
    The algorithm should just return a black image without crashing.
    """
    img = torch.zeros(10, 10, 3)
    out = gray_world(img)

    assert not torch.isnan(out).any(), "A totally black image produced NaNs."
    assert torch.all(out == 0.0), "A totally black image must remain totally black."
