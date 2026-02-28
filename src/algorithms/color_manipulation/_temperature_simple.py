import torch
from torch import Tensor 

def temperature_simple(rgb_image : Tensor, adjustment_value : float) -> Tensor:
    """
    Given a temperature adjustment on the range -100 to 100,
    apply the following adjustment to each pixel in the image :
        red = red + adjustment_value
        green = green
        blue = blue - adjustment_value
    """

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

    return output