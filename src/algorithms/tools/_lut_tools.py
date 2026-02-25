import torch 
import tifffile
import kornia.color as kc
import math

from tifffile import TiffPage
from torch import Tensor

HUE_SAT_MAP_DIMS_TAG = 50937
HUE_SAT_MAP_DATA_1_TAG = 50938
HUE_SAT_MAP_DATA_2_TAG = 50939

NORMALIZATION_HSV_SCALE = torch.tensor([2 * math.pi, 1., 1.]).view(3, 1, 1)

def read_hue_sat_lut_from_dcp(dcp_path: str) -> tuple[Tensor, Tensor] | None:
    """
    Load hue/saturation 3D LUTs from a DCP TIFF file.

    Returns `(lut1, lut2)` tensors of shape `(h, s, v, 3)` or `None`
    if the file cannot be read or lacks the expected tags.
    """ 
    try:
        with tifffile.TiffFile(dcp_path) as dcp_file: 
            page = dcp_file.pages[0]
            if isinstance(page, TiffPage): 
                tags = page.tags
                dims = tags[HUE_SAT_MAP_DIMS_TAG].value
                h, s, v = dims
                lut1 = torch.Tensor(tags[HUE_SAT_MAP_DATA_1_TAG].value).reshape(h, s, v, 3)
                lut2 = torch.Tensor(tags[HUE_SAT_MAP_DATA_2_TAG].value).reshape(h, s, v, 3) 
                return lut1, lut2
    except Exception: 
        return None
    
def rgb_to_hsv(rgb_image: Tensor) -> Tensor: 
    """
    Transforms an RGB image into HSV space image.
    
    Requires a `[H, W, 3]` RGB image.
    Returns a `[H, W, 3]` HSV image.  
    """
    reshaped_rgb_image = rgb_image.permute(2, 0, 1)
    reshaped_hsv_image = kc.rgb_to_hsv(reshaped_rgb_image)
    hsv_image = (reshaped_hsv_image / NORMALIZATION_HSV_SCALE).permute(1, 2, 0)
    
    return hsv_image

def hsv_to_rgb(hsv_image: Tensor) -> Tensor: 
    """
    Transforms an HSV image into RGB space image.
    
    Requires a `[H, W, 3]` HSV image.
    Returns a `[H, W, 3]` RGB image.  
    """
    reshaped_hsv_image = hsv_image.permute(2, 0, 1) * NORMALIZATION_HSV_SCALE
    reshaped_rgb_image = kc.hsv_to_rgb(reshaped_hsv_image)
    rgb_image = reshaped_rgb_image.permute(1, 2, 0)
    
    return rgb_image
    
        