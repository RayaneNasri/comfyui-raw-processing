import torch
import pytest
from torch.testing import assert_close

from algorithms.white_balance import white_patch_ref

def test_preserves_shape():
    """Checks that the output image has the same dimensions as the input."""
    # Format: H, W, C
    img = torch.rand(100, 100, 3) 
    out = white_patch_ref(img, 1.)
    assert out.shape == img.shape, "The tensor shape has changed!"

def test_already_white_balanced():
    """
    If the image already contains pixels at 1.0 for each channel,
    the algorithm should not change anything.
    """
    img = torch.rand(50, 50, 3)
    # Force the maximum of each channel to 1.0 on the very first pixel (0, 0)
    img[0, 0, :] = 1.0  
    
    out = white_patch_ref(img, 1.)
    assert_close(out, img, msg="An already balanced image should not be modified.")

def test_color_correction():
    """
    Tests the core theory: simulates an image with a strong reddish tint.
    """
    # 1. Create a uniform base gray image (value 0.5)
    base_gray = torch.ones(10, 10, 3) * 0.5 
    
    # 2. Apply colored light (Strong red, medium green, weak blue)
    tinted_img = base_gray.clone()
    tinted_img[..., 0] *= 1.0   # Max red stays at 0.5
    tinted_img[..., 1] *= 0.5   # Max green goes to 0.25
    tinted_img[..., 2] *= 0.25  # Max blue goes to 0.125
    
    # Insert the scene's "white point" at pixel [0, 0]
    tinted_img[0, 0, :] = torch.tensor([1.0, 0.5, 0.25])

    # 3. Apply the algorithm
    out = white_patch_ref(tinted_img, 1.)
    
    # 4. Assertions
    # The maximums of the output image MUST all be 1.0
    assert torch.allclose(out[..., 0].max(), torch.tensor(1.0)), "Max of Red channel is not 1.0"
    assert torch.allclose(out[..., 1].max(), torch.tensor(1.0)), "Max of Green channel is not 1.0"
    assert torch.allclose(out[..., 2].max(), torch.tensor(1.0)), "Max of Blue channel is not 1.0"

    # The gray pixel [1, 1] which was at (0.5, 0.25, 0.125) must return 
    # to its original gray (0.5, 0.5, 0.5).
    expected_pixel = torch.tensor([0.5, 0.5, 0.5])
    assert_close(out[1, 1, :], expected_pixel, msg="Intermediate values were not scaled correctly.")

def test_black_pixels_preserved():
    """Checks the physical property of multiplication: black remains black."""
    img = torch.zeros(10, 10, 3)
    # Add a bright pixel to avoid division by zero in this specific test
    img[0, 0, :] = torch.tensor([1.0, 0.8, 0.6]) 
    
    out = white_patch_ref(img, 1.)
    
    # The black pixel at [5, 5] must remain 0.0 for all three channels
    expected_pixel = torch.tensor([0.0, 0.0, 0.0])
    assert_close(out[5, 5, :], expected_pixel, msg="A black pixel must remain black (0.0).")

def test_zero_max_channel_no_nan():
    """
    Tests that the algorithm does not crash or produce NaNs 
    if one specific channel is entirely empty (max = 0).
    """
    # Image with Red and Green, but absolutely NO Blue
    img = torch.ones(10, 10, 3) * 0.5
    img[0, 0, 0] = 1.0  # White point for Red
    img[0, 0, 1] = 1.0  # White point for Green
    img[..., 2] = 0.0   # Blue channel is entirely black
    
    out = white_patch_ref(img, 1.)
    
    # Check that there are no NaN (Not a Number) or Inf values
    assert not torch.isnan(out).any(), "The output contains NaN! Check for division by zero."
    assert not torch.isinf(out).any(), "The output contains Infinity! Check for division by zero."
    
    # The blue channel should remain logically black
    assert torch.all(out[..., 2] == 0.0), "The empty channel should remain at 0.0"

def test_totally_black_image():
    """
    Tests the absolute edge case: a completely black image.
    The algorithm should just return a black image without crashing.
    """
    img = torch.zeros(10, 10, 3)
    out = white_patch_ref(img, 1.)
    
    assert not torch.isnan(out).any(), "A totally black image produced NaNs."
    assert torch.all(out == 0.0), "A totally black image must remain totally black."

def test_percentil_ignores_outliers():
    """
    Tests that a lower percentile (e.g., 0.90) correctly ignores 
    extreme 'hot pixels' (outliers) and scales based on the main distribution.
    """
    # 1. On crée une image de 10x10 (100 pixels) grise unie à 0.5
    img = torch.ones(10, 10, 3) * 0.5
    
    # 2. On introduit UN SEUL "pixel chaud" très brillant à 1.0
    # Cela représente exactement 1% des pixels de l'image.
    img[0, 0, :] = 1.0
    
    # 3. On utilise le 90ème percentile (0.90). 
    # Puisque 99% de l'image est à 0.5, le 90ème percentile DOIT être 0.5 (et non 1.0).
    out = white_patch_ref(img, percentil=0.90)
    
    # 4. Si la référence trouvée est bien 0.5, le multiplicateur est de (1.0 / 0.5) = 2.0.
    # Nos pixels gris d'origine (0.5) doivent donc être devenus blancs (1.0).
    expected_base_pixel = torch.tensor([1.0, 1.0, 1.0])
    
    # On vérifie un pixel normal au hasard (hors du pixel chaud)
    assert_close(out[5, 5, :], expected_base_pixel, msg="The percentile algorithm did not ignore the outlier.")

def test_percentil_out_of_bounds():
    """
    Tests that the function raises a ValueError if the percentil 
    parameter is not strictly between 0.0 and 1.0.
    """
    img = torch.rand(10, 10, 3)
    
    with pytest.raises(ValueError, match="between"):
        # On s'attend à ce que votre code lève une erreur contenant le mot "between" ou "0" ou "1"
        white_patch_ref(img, percentil=-0.1)
        
    with pytest.raises(ValueError, match="between"):
        white_patch_ref(img, percentil=1.1)