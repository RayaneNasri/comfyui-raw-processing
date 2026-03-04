import torch
import pytest

from algorithms.gc import gamma_correction 

def test_gamma_correction_known_values():
    """Tests if the math is applied correctly using predictable values."""
    # Input with pure black, mid-gray, and pure white
    img = torch.tensor([[[0.0, 0.5, 1.0]]]) # Shape: H=1, W=1, C=3
    
    out = gamma_correction(img, gamma=2.0, alpha=1.0)
    
    # Expected math: x^(1/2.0)
    # 0.0^(0.5) = 0.0 | 0.5^(0.5) = ~0.7071 | 1.0^(0.5) = 1.0
    expected = torch.tensor([[[0.0, 0.70710678, 1.0]]])
    assert torch.allclose(out, expected, atol=1e-5), "The (1/gamma) power calculation is incorrect."

def test_gamma_correction_alpha_multiplier():
    """Ensures the alpha multiplier scales the output correctly."""
    img = torch.ones((10, 10, 3)) # 100% white image
    out = gamma_correction(img, gamma=2.2, alpha=0.5)
    
    # 1.0^(1/2.2) * 0.5 = 0.5
    assert torch.all(out == 0.5), "The alpha multiplier was not applied correctly."

def test_gamma_correction_identity():
    """Tests that a gamma of 1.0 leaves the image unchanged (when alpha is 1.0)."""
    img = torch.rand((5, 5, 3))
    out = gamma_correction(img, gamma=1.0, alpha=1.0)
    
    assert torch.allclose(img, out), "Image should remain unchanged when gamma=1.0."

def test_gamma_correction_zero_gamma_raises_error():
    """A gamma of 0 will cause a division by zero (1/0). The function must handle it."""
    img = torch.rand((5, 5, 3))
    # PyTorch might throw a RuntimeError, or your code might raise a ValueError.
    with pytest.raises((ZeroDivisionError, RuntimeError, ValueError)):
        gamma_correction(img, gamma=0.0)


def test_gamma_correction_preserves_shape():
    """The output tensor must have the exact same shape as the input tensor."""
    shape = (1080, 1920, 3)
    img = torch.rand(shape)
    out = gamma_correction(img)
    
    assert out.shape == shape, f"Shape changed! Expected {shape}, got {out.shape}"

def test_gamma_correction_rejects_wrong_type():
    """The function should fail loudly if given a list or numpy array instead of a Tensor."""
    img_list = [[[0.5, 0.5, 0.5]]]
    with pytest.raises((TypeError, AttributeError)):
        gamma_correction(img_list)