import pytest
import torch
import numpy as np

from algorithms.denoising import gaussian_filter


def test_gaussian_filter_valid_baseline():
    """
    Tests the baseline functionality with standard, expected inputs to ensure
    the function works correctly under normal conditions (positive odd ksize).
    """
    img = torch.rand((10, 10, 3), dtype=torch.float32)
    ksize = (3, 3)
    sigmaX = 1.0

    result = gaussian_filter(img, ksize, sigmaX)

    assert isinstance(result, torch.Tensor), "Output must be a torch.Tensor"
    assert result.shape == img.shape, "Output shape must match the input shape"


def test_gaussian_filter_valid_zero_ksize():
    """
    Tests the documented behavior where ksize can be (0, 0) and the kernel
    dimensions are computed dynamically from the provided sigmas.
    """
    img = torch.rand((10, 10, 3), dtype=torch.float32)
    ksize = (0, 0)
    sigmaX = 1.5
    sigmaY = 2.0

    result = gaussian_filter(img, ksize, sigmaX, sigmaY)
    assert result.shape == img.shape, (
        "Output shape must match the input shape when ksize is (0,0)"
    )


@pytest.mark.parametrize(
    "invalid_img", [None, [1, 2, 3], np.random.rand(10, 10, 3), "not_a_tensor"]
)
def test_gaussian_filter_invalid_img_type(invalid_img):
    """
    Tests the function's type validation for the 'img' argument.
    Passing non-torch.Tensor objects should raise a TypeError or ValueError.
    """
    with pytest.raises((TypeError, AttributeError, ValueError)):  # type: ignore
        gaussian_filter(invalid_img, (3, 3), 1.0)


@pytest.mark.parametrize(
    "shape",
    [
        (10,),  # 1D tensor
        (10, 10),  # 2D tensor (missing channels)
        (1, 10, 10, 3),  # 4D tensor (batch dimension included)
        (0, 10, 3),  # Empty tensor (0 height)
        (10, 0, 3),  # Empty tensor (0 width)
        (10, 10, 0),  # Empty tensor (0 channels)
    ],
)
def test_gaussian_filter_unexpected_tensor_dimensions(shape):
    """
    Tests the function's robustness against tensors with unexpected, missing,
    or empty dimensions, which usually crash the underlying OpenCV C++ backend.
    """
    img = torch.rand(shape)
    with pytest.raises((ValueError, RuntimeError)):  # type: ignore
        gaussian_filter(img, (3, 3), 1.0)


@pytest.mark.parametrize(
    "ksize",
    [
        (2, 2),  # Even dimensions are not allowed for Gaussian kernels
        (4, 4),
        (3, 2),  # Mixed odd/even
        (-1, -1),  # Negative dimensions
        (-3, 3),  # Mixed negative/positive
        (0, 3),  # Partially zero (usually OpenCV requires both >0 or both ==0)
        (3, 0),
    ],
)
def test_gaussian_filter_invalid_ksize_values(ksize):
    """
    Tests the function's validation of the 'ksize' parameter logic.
    The documentation states they must be positive and odd, OR both zeros.
    Any other combination should trigger an exception before OpenCV processes it.
    """
    img = torch.rand((10, 10, 3))
    with pytest.raises((ValueError, RuntimeError)):  # type: ignore
        gaussian_filter(img, ksize, 1.0)


@pytest.mark.parametrize(
    "invalid_ksize",
    [
        3,  # Int instead of tuple
        [3, 3],  # List instead of tuple
        (3,),  # Tuple of length 1
        (3, 3, 3),  # Tuple of length 3
        (3.0, 3.0),  # Tuple of floats
        ("3", "3"),  # Tuple of strings
        None,
    ],
)
def test_gaussian_filter_invalid_ksize_type(invalid_ksize):
    """
    Tests strict type enforcement on the 'ksize' tuple parameter.
    It must be strictly a tuple of exactly two integers.
    """
    img = torch.rand((10, 10, 3))
    with pytest.raises((TypeError, ValueError)):  # type: ignore
        gaussian_filter(img, invalid_ksize, 1.0)


@pytest.mark.parametrize(
    "sigmas",
    [
        (-1.0, 0.0),  # Negative sigmaX
        (1.0, -1.0),  # Negative sigmaY
        (float("inf"), 1.0),  # Infinite sigma
        (1.0, float("nan")),  # NaN sigma
    ],
)
def test_gaussian_filter_invalid_sigmas(sigmas):
    """
    Tests the function's resilience against mathematically invalid or extreme
    standard deviation (sigma) values.
    """
    img = torch.rand((10, 10, 3))
    sigmaX, sigmaY = sigmas
    with pytest.raises((ValueError, RuntimeError)):  # type: ignore
        gaussian_filter(img, (3, 3), sigmaX, sigmaY)


def test_gaussian_filter_unsupported_border_wrap():
    """
    Tests the explicit constraint mentioned in the docstring:
    'BORDER_WRAP is not supported.'
    The wrapper should actively intercept this and raise a ValueError.
    """
    img = torch.rand((10, 10, 3))
    with pytest.raises((ValueError, RuntimeError)):  # type: ignore
        gaussian_filter(img, (3, 3), 1.0, strBorderType="BORDER_WRAP")


@pytest.mark.parametrize(
    "invalid_string", ["NON_EXISTENT_BORDER", "", "123", None, 123]
)
def test_gaussian_filter_invalid_border_type_and_hint(invalid_string):
    """
    Tests handling of invalid string types or completely unknown modes
    for strBorderType and strHint.
    """
    img = torch.rand((10, 10, 3))
    with pytest.raises((KeyError, ValueError, TypeError)):  # type: ignore
        gaussian_filter(img, (3, 3), 1.0, strBorderType=invalid_string)

    with pytest.raises((KeyError, ValueError, TypeError)):  # type: ignore
        gaussian_filter(img, (3, 3), 1.0, strHint=invalid_string)


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float16, torch.complex64, torch.bool]
)
def test_gaussian_filter_unsupported_dtypes(dtype):
    """
    Tests the function using unusual or unsupported PyTorch dtypes.
    OpenCV's GaussianBlur has restricted support for datatypes (usually uint8, float32, float64).
    The wrapper must explicitly reject these or cast them safely to avoid a C++ crash.
    """
    img = torch.ones((10, 10, 3), dtype=dtype)
    try:
        gaussian_filter(img, (3, 3), 1.0)
    except (TypeError, RuntimeError, ValueError):
        # A gracefully handled Python exception is acceptable.
        # A hard C++ segmentation fault is what we are testing against.
        pass


def test_gaussian_filter_nan_inf_values_in_tensor():
    """
    Tests how the function behaves when the image tensor contains NaNs or Infs.
    This ensures that anomalous data does not cause an unexpected severe crash.
    """
    img = torch.rand((10, 10, 3))
    img[0, 0, 0] = float("nan")
    img[1, 1, 1] = float("inf")

    try:
        gaussian_filter(img, (3, 3), 1.0)
    except Exception:
        # Handled exceptions or propagation of NaNs are acceptable.
        pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gaussian_filter_gpu_tensor():
    """
    Tests if the wrapper correctly handles tensors located on a CUDA device.
    Since OpenCV expects CPU-bound numpy arrays, passing a GPU tensor directly
    will cause an immediate crash unless the wrapper performs `.cpu().numpy()`.
    """
    img = torch.rand((10, 10, 3)).cuda()

    try:
        result = gaussian_filter(img, (3, 3), 1.0)
        assert result.device == img.device, (
            "If input is on GPU, output should return to GPU"
        )
    except TypeError:
        pytest.fail(
            "Function crashed on GPU tensor. It lacks proper tensor-to-numpy conversion logic."
        )
