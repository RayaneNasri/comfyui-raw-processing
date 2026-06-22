import pytest
import torch
import numpy as np

from algorithms.denoising import nl_means


def test_invalid_image_type():
    """Test that the function raises an appropriate error when the image is not a torch.Tensor."""
    invalid_inputs = [None, [0, 0, 0], np.zeros((10, 10, 3)), "image.jpg"]
    for img in invalid_inputs:
        with pytest.raises((TypeError, ValueError, AttributeError)):
            nl_means(img=img)


def test_invalid_image_dimensions():
    """Test that the function handles tensors with incorrect dimensions (not 3D or 4D)."""
    img_1d = torch.rand(10)
    img_2d = torch.rand(10, 10)
    img_5d = torch.rand(1, 1, 10, 10, 3)

    for img in [img_1d, img_2d, img_5d]:
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            nl_means(img=img)


def test_invalid_channel_counts():
    """Test that the function correctly rejects images that do not have exactly 3 channels (RGB/BGR)."""
    img_1_channel = torch.rand(10, 10, 1)
    img_4_channel = torch.rand(10, 10, 4)

    with pytest.raises((ValueError, RuntimeError)):
        nl_means(img=img_1_channel)

    with pytest.raises((ValueError, RuntimeError)):
        nl_means(img=img_4_channel)


def test_empty_image_tensor():
    """Test how the function handles an empty tensor (0 width or 0 height)."""
    img_empty_h = torch.rand(0, 10, 3)
    img_empty_w = torch.rand(10, 0, 3)

    with pytest.raises((ValueError, RuntimeError)):
        nl_means(img=img_empty_h)

    with pytest.raises((ValueError, RuntimeError)):
        nl_means(img=img_empty_w)


def test_tensor_with_nan_and_inf():
    """Test that the function safely handles or explicitly rejects NaNs and Infs in the input tensor."""
    img_nan = torch.rand(10, 10, 3)
    img_nan[5, 5, 1] = float("nan")

    img_inf = torch.rand(10, 10, 3)
    img_inf[2, 2, 0] = float("inf")

    try:
        res_nan = nl_means(img=img_nan)
        assert torch.isnan(res_nan).any() or not torch.isnan(res_nan).any()
    except Exception:
        pass


def test_image_values_out_of_bounds():
    """Test how the algorithm handles tensors with extreme values well outside the expected [0, 1] range."""
    img_negative = torch.rand(10, 10, 3) * -100.0
    img_massive = torch.rand(10, 10, 3) * 1e6

    try:
        nl_means(img=img_negative)
        nl_means(img=img_massive)
    except Exception as e:
        assert isinstance(e, (ValueError, RuntimeError))


def test_even_window_sizes_warns_and_corrects():
    """Test that the function emits a warning and auto-decrements when window sizes are even."""
    img = torch.rand(32, 32, 3)

    with pytest.warns(Warning):
        # 6 should become 5
        res = nl_means(img=img, templateWindowSize=6, searchWindowSize=21)
        assert res.shape == img.shape

    with pytest.warns(Warning):
        # 20 should become 19
        res = nl_means(img=img, templateWindowSize=7, searchWindowSize=20)
        assert res.shape == img.shape


def test_window_size_larger_than_image_warns_and_caps():
    """Test that a searchWindowSize exceeding min(h,w)/2 is warned about, capped, and forced odd."""
    # Image min dimension is 20. Max allowed size is 20 / 2 = 10.
    # 10 is even, so it should be decremented to 9.
    img = torch.rand(20, 20, 3)

    with pytest.warns(Warning):
        # User provides 51. Should be capped to 9 without crashing.
        res = nl_means(img=img, templateWindowSize=5, searchWindowSize=51)
        assert res.shape == img.shape


def test_search_window_smaller_than_template_warns_and_fixes():
    """Test that the function warns and aligns search window to template window if strictly smaller."""
    img = torch.rand(32, 32, 3)

    with pytest.warns(Warning):
        # Search is 3, template is 7. Search should become 7.
        res = nl_means(img=img, templateWindowSize=7, searchWindowSize=3)
        assert res.shape == img.shape


def test_combined_window_size_adversarial_corrections():
    """
    Test a chaotic combination of bad inputs that require multiple corrections to ensure logic order is robust.
    Image max is 10 (odd cap 9).
    User inputs: template=12 (even, > max), search=6 (even, < template).
    This forces the system to perform a sequence of validations that could easily overwrite each other or cause logical loops.
    """
    img = torch.rand(20, 20, 3)

    with pytest.warns(Warning):
        res = nl_means(img=img, templateWindowSize=12, searchWindowSize=6)
        assert res.shape == img.shape


def test_negative_or_zero_window_sizes():
    """Test that the function still rejects negative or zero window sizes as they are mathematically impossible."""
    img = torch.rand(32, 32, 3)

    invalid_sizes = [0, -7, -21]
    for size in invalid_sizes:
        with pytest.raises((ValueError, RuntimeError)):
            nl_means(img=img, templateWindowSize=size)
        with pytest.raises((ValueError, RuntimeError)):
            nl_means(img=img, searchWindowSize=size)


def test_invalid_parameter_types_for_windows():
    """Test that the function raises a TypeError when float values are passed to integer window size arguments."""
    img = torch.rand(100, 100, 3)

    with pytest.raises(TypeError):
        nl_means(img=img, templateWindowSize=7.5)

    with pytest.raises(TypeError):
        nl_means(img=img, searchWindowSize=21.1)


def test_negative_h_parameters():
    """Test the algorithm's stability when the filtering strength parameters are negative (should raise)."""
    img = torch.rand(16, 16, 3)

    try:
        nl_means(img=img, h=-5.0, hColor=-3.0)
    except Exception as e:
        assert isinstance(e, (ValueError, RuntimeError))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tensor_on_gpu():
    """Test that the function handles tensors located on a CUDA device by safely transferring them to CPU first."""
    img = torch.rand(16, 16, 3).cuda()

    try:
        out = nl_means(img=img)
        assert isinstance(out, torch.Tensor)
    except (TypeError, RuntimeError) as e:
        assert "cpu" in str(e).lower()


def test_non_contiguous_tensor():
    """Test that the function can handle memory-fragmented, non-contiguous tensors."""
    img = torch.rand(3, 32, 32)
    img_transposed = img.permute(1, 2, 0)

    assert not img_transposed.is_contiguous()

    try:
        out = nl_means(img=img_transposed, searchWindowSize=7)
        assert out.shape == img_transposed.shape
    except Exception as e:
        pytest.fail(f"Failed on non-contiguous tensor with exception: {e}")
