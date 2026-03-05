import pytest
import torch
import numpy as np

from src.algorithms.denoising import avg_filter 

def test_avg_filter_valid_baseline():
    """
    Tests the baseline functionality with standard, expected inputs to ensure 
    the function works under normal conditions.
    """
    img = torch.rand((10, 10, 3), dtype=torch.float32)
    ksize = (3, 3)
    border_type = "BORDER_DEFAULT"
    
    result = avg_filter(img, ksize, border_type)
    
    assert isinstance(result, torch.Tensor), "Output must be a torch.Tensor"
    assert result.shape == img.shape, "Output shape must match input shape"

@pytest.mark.parametrize("invalid_img", [
    None,
    [1, 2, 3],
    np.random.rand(10, 10, 3),
    "not_a_tensor"
])
def test_avg_filter_invalid_img_type(invalid_img):
    """
    Tests how the function handles 'img' arguments that are not torch.Tensors.
    A robust implementation should raise a TypeError.
    """
    with pytest.raises((TypeError, AttributeError, ValueError)):
        avg_filter(invalid_img, (3, 3), "BORDER_DEFAULT")

@pytest.mark.parametrize("shape", [
    (10,),              # 1D tensor
    (10, 10),           # 2D tensor (missing channels)
    (1, 10, 10, 3),     # 4D tensor (batch dimension included)
    (0, 10, 3),         # Empty tensor (0 height)
    (10, 0, 3)          # Empty tensor (0 width)
])
def test_avg_filter_unexpected_tensor_dimensions(shape):
    """
    Tests the function's robustness against tensors with unexpected dimensions.
    OpenCV functions generally expect 2D or 3D numpy arrays; unhandled dimensions 
    should trigger a clear ValueError rather than a cryptic OpenCV C++ exception.
    """
    img = torch.rand(shape)
    with pytest.raises((ValueError, RuntimeError)):
        avg_filter(img, (3, 3), "BORDER_DEFAULT")

@pytest.mark.parametrize("ksize", [
    (0, 0),             # Zero kernel size
    (-1, -1),           # Negative kernel size
    (0, 3),             # Partially zero
    (-3, 3)             # Partially negative
])
def test_avg_filter_invalid_kernel_values(ksize):
    """
    Tests the function's handling of mathematically invalid kernel sizes for blurring.
    OpenCV's blur will crash if ksize <= 0. The wrapper should catch this early.
    """
    img = torch.rand((10, 10, 3))
    with pytest.raises((ValueError, RuntimeError)):
        avg_filter(img, ksize, "BORDER_DEFAULT")

@pytest.mark.parametrize("ksize", [
    3,                  # Integer instead of tuple
    [3, 3],             # List instead of tuple
    (3,),               # Tuple with missing element
    (3, 3, 3),          # Tuple with too many elements
    (3.5, 3.5),         # Tuple of floats
    ("3", "3")          # Tuple of strings
])
def test_avg_filter_invalid_kernel_type_and_length(ksize):
    """
    Tests the function's type validation for the 'ksize' parameter.
    It should strictly accept a tuple of exactly two integers.
    """
    img = torch.rand((10, 10, 3))
    with pytest.raises((TypeError, ValueError)):
        avg_filter(img, ksize, "BORDER_DEFAULT")

@pytest.mark.parametrize("border_type", [
    "UNKNOWN_BORDER",   # Non-existent border type
    "",                 # Empty string
    "border_default",   # Lowercase (if exact string matching is strictly case-sensitive)
    None,               # NoneType
    123                 # Integer instead of string
])
def test_avg_filter_invalid_border_type(border_type):
    """
    Tests how the function handles invalid, empty, or misspelled border type strings.
    It should raise a KeyError or ValueError instead of failing silently or crashing.
    """
    img = torch.rand((10, 10, 3))
    with pytest.raises((KeyError, ValueError, TypeError)):
        avg_filter(img, (3, 3), border_type)

@pytest.mark.parametrize("dtype", [
    torch.float16,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.bool
])
def test_avg_filter_unusual_dtypes(dtype):
    """
    Tests the function using unusual or unsupported tensor data types.
    Converting between PyTorch and OpenCV (numpy) can cause severe issues if
    the dtypes are not mapped correctly or explicitly cast.
    """
    img = torch.ones((10, 10, 3), dtype=dtype)
    try:
        result = avg_filter(img, (3, 3), "BORDER_DEFAULT")
        assert result.dtype == torch.float32, "Data type was unexpectedly altered in a destructive way"
    except (TypeError, RuntimeError, ValueError) as e:
        # It is acceptable for the function to explicitly reject unsupported dtypes
        pass

def test_avg_filter_extreme_kernel_size():
    """
    Tests an edge case where the blurring kernel is significantly larger than 
    the input image itself. This checks the extrapolation/padding logic of OpenCV.
    """
    img = torch.rand((2, 2, 3))
    ksize = (100, 100)
    border_type = "BORDER_REFLECT"
    
    result = avg_filter(img, ksize, border_type)
    assert result.shape == img.shape, "Shape should remain constant even with massive kernels"

def test_avg_filter_nan_and_inf_values():
    """
    Tests the function's resilience when the input tensor contains NaNs or Infinities.
    Depending on the implementation, this should either propagate NaNs/Infs, 
    or be explicitly caught and handled by the wrapper.
    """
    img = torch.rand((10, 10, 3))
    img[0, 0, 0] = float('nan')
    img[1, 1, 1] = float('inf')
    
    # We just expect it not to hard-crash the Python interpreter (e.g. C++ level segmentation fault)
    try:
        avg_filter(img, (3, 3), "BORDER_DEFAULT")
    except Exception as e:
        pass # A handled exception is acceptable, a hard crash is not

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_avg_filter_gpu_tensor():
    """
    Tests if the wrapper correctly handles tensors located on a GPU device.
    Since OpenCV requires numpy arrays (which live on CPU RAM), passing a CUDA 
    tensor will cause a crash unless the wrapper explicitly calls `.cpu().numpy()`.
    """
    img = torch.rand((10, 10, 3)).cuda()
    
    try:
        result = avg_filter(img, (3, 3), "BORDER_DEFAULT")
        assert result.is_cuda, "If input is on GPU, output should ideally be on GPU"
    except TypeError:
        pytest.fail("Function crashed on GPU tensor. It lacks proper .cpu()/.numpy() conversion.")