import torch
from torch.testing import assert_close

from algorithms.white_balance import white_patch_ref

def test_preserves_shape():
    """Checks that the output image has the same dimensions as the input."""
    # Format: H, W, C
    img = torch.rand(100, 100, 3) 
    out = white_patch_ref(img)
    assert out.shape == img.shape, "The tensor shape has changed!"

def test_already_white_balanced():
    """
    If the image already contains pixels at 1.0 for each channel,
    the algorithm should not change anything.
    """
    img = torch.rand(50, 50, 3)
    # Force the maximum of each channel to 1.0 on the very first pixel (0, 0)
    img[0, 0, :] = 1.0  
    
    out = white_patch_ref(img)
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
    out = white_patch_ref(tinted_img)
    
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
    
    out = white_patch_ref(img)
    
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
    
    out = white_patch_ref(img)
    
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
    out = white_patch_ref(img)
    
    assert not torch.isnan(out).any(), "A totally black image produced NaNs."
    assert torch.all(out == 0.0), "A totally black image must remain totally black."