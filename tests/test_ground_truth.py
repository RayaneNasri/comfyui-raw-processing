import torch
import pytest
from torch.testing import assert_close

from algorithms.white_balance import ground_truth

def test_preserves_shape():
    """Checks that the output image has the same dimensions as the input."""
    img = torch.rand(100, 100, 3)
    patch = img[10:20, 10:20, :] # Patch de 10x10
    out = ground_truth(img, patch, method='mean')
    assert out.shape == img.shape, "The tensor shape has changed!"

def test_empty_patch_raises_error():
    """
    Checks that the function raises a ValueError if the patch is empty.
    (e.g., a tensor of shape 0x0x3)
    """
    img = torch.rand(100, 100, 3)
    empty_patch = torch.empty(0, 0, 3) # Patch vide !
    
    with pytest.raises(ValueError, match="empty"):
        # Le test passe si ground_truth lève bien une ValueError contenant le mot "empty"
        ground_truth(img, empty_patch, method='mean')

def test_already_balanced_mean():
    """If the patch is already perfectly gray, no changes should occur."""
    img = torch.rand(50, 50, 1).repeat(1, 1, 3)
    patch = img[0:10, 0:10, :]
    out = ground_truth(img, patch, method='mean')
    assert_close(out, img, msg="An already balanced image should not be modified.")

def test_mode_mean_correction():
    """
    Tests the 'mean' mode. After correction, the patch area in the OUTPUT 
    image must have the exact same mean for all 3 channels.
    """
    img = torch.ones(20, 20, 3)
    img[..., 0] *= 0.8  # Red
    img[..., 1] *= 0.4  # Green
    img[..., 2] *= 0.2  # Blue
    
    patch = img[5:15, 5:15, :] # Extracting the patch
    
    out = ground_truth(img, patch, method='mean')
    
    # We analyze the patch area in the OUTPUT image
    out_patch = out[5:15, 5:15, :]
    mean_r = out_patch[..., 0].mean()
    mean_g = out_patch[..., 1].mean()
    mean_b = out_patch[..., 2].mean()

    assert torch.allclose(mean_r, mean_g), "Red and Green means in the patch differ!"
    assert torch.allclose(mean_g, mean_b), "Green and Blue means in the patch differ!"

def test_mode_max_correction():
    """
    Tests the 'max' mode (like White Patch, but localized). 
    After correction, the max values of the patch area in the OUTPUT 
    image must all be equal to 1.0.
    """
    img = torch.ones(20, 20, 3) * 0.5
    # Add a bright reddish spot inside our patch area
    img[10, 10, :] = torch.tensor([0.8, 0.4, 0.2])
    
    patch = img[5:15, 5:15, :]
    
    out = ground_truth(img, patch, method='max')
    
    out_patch = out[5:15, 5:15, :]
    assert torch.allclose(out_patch[..., 0].max(), torch.tensor(1.0)), "Max Red is not 1.0"
    assert torch.allclose(out_patch[..., 1].max(), torch.tensor(1.0)), "Max Green is not 1.0"
    assert torch.allclose(out_patch[..., 2].max(), torch.tensor(1.0)), "Max Blue is not 1.0"

def test_black_pixels_preserved():
    """Multiplication property: black must remain black."""
    img = torch.ones(20, 20, 3) * 0.5
    img[0, 0, :] = 0.0 # Pixel noir pur hors du patch
    
    patch = img[5:15, 5:15, :] 
    out = ground_truth(img, patch, method='mean')
    
    expected_pixel = torch.tensor([0.0, 0.0, 0.0])
    assert_close(out[0, 0, :], expected_pixel, msg="A black pixel must remain black (0.0).")

def test_zero_value_patch_no_nan():
    """
    Tests division by zero prevention. If a channel in the patch is entirely 0, 
    it should not produce NaNs or Infs in the output image.
    """
    img = torch.ones(20, 20, 3) * 0.5
    img[..., 2] = 0.0 # Le canal bleu de toute l'image (et donc du patch) est à 0
    
    patch = img[5:15, 5:15, :]
    
    # Test pour le mode MEAN
    out_mean = ground_truth(img, patch, method='mean')
    assert not torch.isnan(out_mean).any(), "Mean mode produced NaNs."
    assert not torch.isinf(out_mean).any(), "Mean mode produced Infs."
    
    # Test pour le mode MAX
    out_max = ground_truth(img, patch, method='max')
    assert not torch.isnan(out_max).any(), "Max mode produced NaNs."
    assert not torch.isinf(out_max).any(), "Max mode produced Infs."

def test_invalid_mode_raises_error():
    """Checks that giving a mode other than 'mean' or 'max' raises a ValueError."""
    img = torch.ones(10, 10, 3)
    patch = img[0:5, 0:5, :]
    
    with pytest.raises(ValueError):
         ground_truth(img, patch, method='median') # 'median' n'existe pas !