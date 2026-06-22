import pytest
import torch
import numpy as np

from algorithms.denoising import median_filter


def test_median_filter_valid_baseline():
    """
    Tests the baseline functionality with standard, expected inputs to ensure
    the function works correctly under normal conditions.
    """
    img = torch.rand((10, 10, 3), dtype=torch.float32)
    ksize = 3

    result = median_filter(img, ksize)

    assert isinstance(result, torch.Tensor), "Output must be a torch.Tensor"
    assert result.shape == img.shape, "Output shape must match the input shape"


@pytest.mark.parametrize(
    "invalid_img", [None, [1, 2, 3], np.random.rand(10, 10, 3), "not_a_tensor"]
)
def test_median_filter_invalid_img_type(invalid_img):
    """
    Tests how the function handles 'img' arguments that are not torch.Tensors.
    It should raise a clear TypeError or ValueError.
    """
    with pytest.raises((TypeError, AttributeError, ValueError)):  # type: ignore
        median_filter(invalid_img, 3)


@pytest.mark.parametrize("ksize", [4, 6, 10])
def test_median_filter_even_ksize_warns_and_fixes(ksize):
    """
    Tests the function's handling of even kernel sizes strictly greater than 2.
    It should emit a warning and auto-decrement the value by 1 to make it odd,
    processing the image successfully without raising an exception.
    """
    img = torch.rand((10, 10, 3), dtype=torch.float32)
    with pytest.warns(Warning):
        result = median_filter(img, ksize)
        assert result.shape == img.shape


def test_median_filter_ksize_2_raises():
    """
    Tests the edge case where ksize is 2. Since the new logic subtracts 1 from even
    numbers, it would become 1. However, median filters require ksize > 1.
    Therefore, passing 2 should still raise an exception.
    """
    img = torch.rand((10, 10, 3), dtype=torch.float32)
    with pytest.raises((ValueError, RuntimeError)):  # type: ignore
        median_filter(img, 2)


@pytest.mark.parametrize("ksize", [1, 0, -1, -3, -5])
def test_median_filter_ksize_too_small(ksize):
    """
    Tests the function's handling of kernel sizes that are less than or equal to 1.
    The documentation specifies ksize must be greater than 1 (and odd).
    """
    img = torch.rand((10, 10, 3), dtype=torch.float32)
    with pytest.raises((ValueError, RuntimeError)):  # type: ignore
        median_filter(img, ksize)


@pytest.mark.parametrize(
    "invalid_ksize",
    [
        3.0,  # Float
        "3",  # String
        [3],  # List
        (3, 3),  # Tuple (OpenCV's medianBlur takes a single int, not a tuple)
        None,  # NoneType
    ],
)
def test_median_filter_invalid_ksize_type(invalid_ksize):
    """
    Tests the type validation for the 'ksize' parameter.
    It should strictly accept an integer.
    """
    img = torch.rand((10, 10, 3), dtype=torch.float32)
    with pytest.raises((TypeError, ValueError)):  # type: ignore
        median_filter(img, invalid_ksize)


@pytest.mark.parametrize(
    "shape",
    [
        (10,),  # 1D tensor
        (1, 10, 10, 3),  # 4D tensor (batch dimension included)
        (5, 5, 5, 5),  # 4D tensor
        (0, 10, 3),  # Empty tensor (0 height)
        (10, 0, 3),  # Empty tensor (0 width)
    ],
)
def test_median_filter_unexpected_tensor_dimensions(shape):
    """
    Tests the function's robustness against tensors with unexpected or empty dimensions.
    Unhandled dimensions usually trigger a cryptic OpenCV C++ exception. The wrapper
    should provide a clean Python error.
    """
    img = torch.rand(shape)
    with pytest.raises((ValueError, RuntimeError)):  # type: ignore
        median_filter(img, 3)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32, torch.int64, torch.bool])
def test_median_filter_unsupported_dtypes(dtype):
    """
    Tests the function using tensor data types that are typically not supported
    by OpenCV's medianBlur (which generally expects 8-bit ints or 32-bit floats).
    The wrapper should either cast them safely or explicitly reject them.
    """
    img = torch.ones((10, 10, 3), dtype=dtype)
    try:
        result = median_filter(img, 3)
        assert isinstance(result, torch.Tensor)
    except (TypeError, RuntimeError, ValueError):
        # It is acceptable for the function to explicitly reject unsupported dtypes
        pass


def test_median_filter_large_ksize_float32_gotcha():
    """
    Tests a known OpenCV edge case: cv2.medianBlur only supports float32 arrays
    if the aperture size (ksize) is 3 or 5. If ksize > 5, it requires 8-bit images.
    A robust wrapper must catch this to prevent a C++ assertion failure or cast internally.
    """
    img = torch.rand((20, 20, 3), dtype=torch.float32)
    ksize = 7  # > 5, should trigger OpenCV error on float32 if not handled/cast

    try:
        median_filter(img, ksize)
    except Exception as e:
        # A gracefully handled Python exception is acceptable.
        # A hard C++ crash is the failure condition we are testing against.
        assert isinstance(e, (ValueError, RuntimeError, TypeError))


def test_median_filter_nan_and_inf_values():
    """
    Tests the function's resilience when the input tensor contains NaNs or Infinities.
    This ensures the underlying C++ code doesn't crash catastrophically.
    """
    img = torch.rand((10, 10, 3), dtype=torch.float32)
    img[0, 0, 0] = float("nan")
    img[1, 1, 1] = float("inf")

    try:
        res = median_filter(img, 3)
        assert isinstance(res, torch.Tensor)
    except Exception:
        pass  # Handled exceptions are fine, segmentation faults are not


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_median_filter_gpu_tensor():
    """
    Tests if the wrapper correctly handles tensors located on a GPU device.
    OpenCV operates on CPU numpy arrays. Passing a CUDA tensor directly will
    cause a crash unless the wrapper explicitly calls `.cpu().numpy()`.
    """
    img = torch.rand((10, 10, 3), dtype=torch.float32).cuda()

    try:
        result = median_filter(img, 3)
        assert (
            result.is_cuda or not result.is_cuda
        )  # Main point is that it shouldn't crash
    except TypeError:
        pytest.fail(
            "Function crashed on GPU tensor. It lacks proper .cpu()/.numpy() conversion."
        )


def test_median_filter_non_contiguous_tensor():
    """
    Tests that the function can handle memory-fragmented, non-contiguous tensors,
    which often cause issues when converting from torch to numpy/OpenCV representations.
    """
    img = torch.rand((3, 32, 32), dtype=torch.float32)
    img_transposed = img.permute(
        1, 2, 0
    )  # Shape becomes (32, 32, 3) but non-contiguous

    assert not img_transposed.is_contiguous()

    try:
        out = median_filter(img_transposed, 3)
        assert out.shape == img_transposed.shape
    except Exception as e:
        pytest.fail(f"Failed on non-contiguous tensor with exception: {e}")
