import torch
from torch import Tensor 

def temperature_simple(rgb_image : Tensor, adjustment_value : float) -> Tensor:
    """
    - rgb_image : Tensor RGB image (H,W,3) with each channel represented as a float in [0,1]
    - adjustment_value : float in [-100, 100] (recommended [-20, 20])

    Given a temperature adjustment on the range -100 to 100 (recommended -20, 20),
    apply the following adjustment to each pixel in the rgb_image :
        red = red + adjustment_value/255
        green = green
        blue = blue - adjustment_value/255
    Clamp values to the range [0, 1]
    """

    # Normalize the adjustement_value:
    adjustment_value = adjustment_value/255

    # Converts adjustment_value to the same dtype and device than values of rgb_image
    adjustment_value = torch.as_tensor(
        adjustment_value,
        dtype=rgb_image.dtype,
        device=rgb_image.device
    )

    # Creates a vector [+adj, 0, -adj]
    adjustment = torch.stack((
        adjustment_value,
        torch.zeros_like(adjustment_value),
        -adjustment_value
    ))
    
    # broadcasts the adjustement on a (H x W x 3) rgb_image
    output = rgb_image + adjustment

    # Clamp values to the range [0, 1]
    output = torch.clamp(output, 0.0, 1.0)

    return output