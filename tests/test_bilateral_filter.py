import pytest
import torch

from algorithms.denoising import bilateral_filter


def test_bilateral_filter_valid_baseline():
    """
    Tests the baseline functionality with standard, expected inputs to ensure
    the function works correctly under normal conditions.
    """
    img = torch.rand((10, 10, 3), dtype=torch.float32)

    result = bilateral_filter(img, d=5, sigmaColor=75.0, sigmaSpace=75.0)

    assert isinstance(result, torch.Tensor), "Output must be a torch.Tensor"
    assert result.shape == img.shape, "Output shape must match the input shape"


@pytest.mark.parametrize(
    "d",
    [
        5.5,  # Float instead of int
        "5",  # String instead of int
        [5],  # List
        None,  # NoneType
    ],
)
def test_bilateral_filter_invalid_d_type(d):
    """
    Tests strict type enforcement on the 'd' (diameter) parameter.
    It must be strictly an integer.
    """
    img = torch.rand((10, 10, 3))
    with pytest.raises(TypeError): 
        bilateral_filter(img, d=d, sigmaColor=75.0, sigmaSpace=75.0)


@pytest.mark.parametrize("d_val", [0, -5])
def test_bilateral_filter_non_positive_d(d_val):
    """
    Tests edge case where d <= 0. According to OpenCV documentation,
    if d is non-positive, it is computed from sigmaSpace. This should
    execute successfully without raising an error.
    """
    img = torch.rand((10, 10, 3))
    result = bilateral_filter(img, d=d_val, sigmaColor=75.0, sigmaSpace=75.0)
    assert isinstance(result, torch.Tensor)


@pytest.mark.parametrize(
    "invalid_sigma",
    [
        -1.0,  # Negative value
        -100,  # Negative integer
    ],
)
def test_bilateral_filter_negative_sigmas(invalid_sigma):
    """
    Tests the mathematical validation for standard deviations (sigmas).
    They represent distances/variances and cannot be negative.
    """
    img = torch.rand((10, 10, 3))
    with pytest.raises(ValueError): 
        bilateral_filter(img, d=5, sigmaColor=invalid_sigma, sigmaSpace=75.0)

    with pytest.raises(ValueError): 
        bilateral_filter(img, d=5, sigmaColor=75.0, sigmaSpace=invalid_sigma)


@pytest.mark.parametrize(
    "invalid_sigma_type",
    [
        "75.0",  # String
        [75.0],  # List
        None,  # NoneType
    ],
)
def test_bilateral_filter_invalid_sigma_types(invalid_sigma_type):
    """
    Tests strict type enforcement on sigmaColor and sigmaSpace.
    They must be instances of int or float.
    """
    img = torch.rand((10, 10, 3))
    with pytest.raises(TypeError):
        bilateral_filter(img, d=5, sigmaColor=invalid_sigma_type, sigmaSpace=75.0)

    with pytest.raises(TypeError):
        bilateral_filter(img, d=5, sigmaColor=75.0, sigmaSpace=invalid_sigma_type)


@pytest.mark.parametrize(
    "anomalous_sigma",
    [
        float("inf"),  # Infinity
        float("nan"),  # Not a Number
    ],
)
def test_bilateral_filter_nan_inf_sigmas(anomalous_sigma):
    """
    Tests the robust handling of NaN and Inf values for sigmas.
    The wrapper must explicitly intercept these using math.isinf and math.isnan.
    """
    img = torch.rand((10, 10, 3))
    with pytest.raises(ValueError):
        bilateral_filter(img, d=5, sigmaColor=anomalous_sigma, sigmaSpace=75.0)

    with pytest.raises(ValueError):
        bilateral_filter(img, d=5, sigmaColor=75.0, sigmaSpace=anomalous_sigma)


@pytest.mark.parametrize(
    "border_type",
    [
        "BORDER_WRAP",  # Often unsupported or problematic depending on OpenCV version
        "NON_EXISTENT",  # Unknown border
        "",  # Empty string
        123,  # Integer instead of string
        None,  # NoneType
    ],
)
def test_bilateral_filter_invalid_border_type(border_type):
    """
    Tests handling of invalid or incorrectly typed border types.
    """
    img = torch.rand((10, 10, 3))
    with pytest.raises((ValueError, TypeError, KeyError)): # type: ignore
        bilateral_filter(
            img, d=5, sigmaColor=75.0, sigmaSpace=75.0, strBorderType=border_type
        )


def test_bilateral_filter_large_sigmas_and_d():
    """
    Tests the filter under extreme but mathematically valid parameters (very large d and sigmas).
    This stresses the OpenCV backend to ensure it doesn't run out of memory or crash.
    """
    img = torch.rand((20, 20, 3))
    # Large diameter could cause heavy computation
    result = bilateral_filter(img, d=100, sigmaColor=10000.0, sigmaSpace=10000.0)
    assert result.shape == img.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bilateral_filter_gpu_tensor():
    """
    Tests the .cpu().numpy() memory movement logic inside the wrapper.
    If an image is on the GPU, OpenCV will crash unless it is safely brought to CPU RAM first.
    """
    img = torch.rand((10, 10, 3)).cuda()

    try:
        result = bilateral_filter(img, d=5, sigmaColor=75.0, sigmaSpace=75.0)
        assert isinstance(result, torch.Tensor)
    except TypeError:
        pytest.fail(
            "Function crashed on GPU tensor. The .cpu() logic failed or was bypassed."
        )


def test_bilateral_filter_decorator_integration_out_of_bounds():
    """
    Tests if the @validate_image_input decorator correctly intercepts out-of-bounds pixel values
    before they reach the clipping logic inside the wrapper.
    """
    # Create tensor with values outside [0, 1] range
    img = torch.rand((10, 10, 3)) * 2.0

    with pytest.raises(ValueError):
        bilateral_filter(img, d=5, sigmaColor=75.0, sigmaSpace=75.0)


def test_bilateral_filter_decorator_integration_invalid_dims():
    """
    Tests if the @validate_image_input decorator correctly intercepts non-3D tensors.
    """
    img = torch.rand((1, 10, 10, 3))  # 4D batched tensor

    with pytest.raises(ValueError):
        bilateral_filter(img, d=5, sigmaColor=75.0, sigmaSpace=75.0)
