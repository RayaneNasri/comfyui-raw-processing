import torch 
import tifffile
import kornia.color as kc
import math

from tifffile import TiffPage
from torch import Tensor

HUE_SAT_MAP_DIMS_TAG = 50937
HUE_SAT_MAP_DATA_1_TAG = 50938
HUE_SAT_MAP_DATA_2_TAG = 50939
CALIBRATION_ILLUMINANT_1_TAG = 50778
CALIBRATION_ILLUMINANT_2_TAG = 50779
INDOOR_COLOR_MATRIX_TAG = 50721
DAYLIGHT_COLOR_MATRIX_TAG = 50722
FORWARD_MATRIX_1_TAG = 50964
FORWARD_MATRIX_2_TAG = 50965

NORMALIZATION_HSV_SCALE = torch.tensor([2 * math.pi, 1., 1.]).view(3, 1, 1)

def _flatten_num_denum_color_matrix(color_matrix: Tensor) -> Tensor: 
    """
    Flattens a 18 sized numerator/denumerator color matrix tensor into a 3x3 tensor 
    """
    num = color_matrix[0::2]
    denum = color_matrix[1::2]
    flattened = num / denum
    
    return flattened.reshape(3, 3)

def read_hue_sat_lut_from_dcp(dcp_path: str) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int] | None:
    try:
        with tifffile.TiffFile(dcp_path) as dcp_file: 
            page = dcp_file.pages[0]
            if isinstance(page, TiffPage): 
                tags = page.tags
                for tag in tags: print(tag)
                dims = tags[HUE_SAT_MAP_DIMS_TAG].value
                h, s, v = dims
                low_temp_lut = Tensor(tags[HUE_SAT_MAP_DATA_1_TAG].value).reshape(h, s, v, 3)
                high_temp_lut = Tensor(tags[HUE_SAT_MAP_DATA_2_TAG].value).reshape(h, s, v, 3)
                indoor_color_matrix = Tensor(tags[INDOOR_COLOR_MATRIX_TAG].value)
                daylight_color_matrix = Tensor(tags[DAYLIGHT_COLOR_MATRIX_TAG].value)
                calib_illum_1 = int(tags[CALIBRATION_ILLUMINANT_1_TAG].value)
                calib_illum_2 = int(tags[CALIBRATION_ILLUMINANT_2_TAG].value)
                forward_matrix_1 = _flatten_num_denum_color_matrix(Tensor(tags[FORWARD_MATRIX_1_TAG].value))
                forward_matrix_2 = _flatten_num_denum_color_matrix(Tensor(tags[FORWARD_MATRIX_2_TAG].value))
                
                return (
                    low_temp_lut, 
                    high_temp_lut, 
                    _flatten_num_denum_color_matrix(indoor_color_matrix), 
                    _flatten_num_denum_color_matrix(daylight_color_matrix), 
                    forward_matrix_1,
                    forward_matrix_2,
                    calib_illum_1, 
                    calib_illum_2
                )
    except Exception as e: 
        print(e) # TODO : implement proper exceptions handling
        return None
    
def rgb_to_hsv(rgb_image: Tensor) -> Tensor: 
    """
    Transforms an RGB image into HSV space image.
    
    Requires a `[H, W, 3]` RGB image.
    Returns a `[H, W, 3]` HSV image.  
    """
    scale = NORMALIZATION_HSV_SCALE.to(rgb_image.device)
    if rgb_image.dim() == 4:
        reshaped_rgb_image = rgb_image.squeeze(0).permute(2, 0, 1)
    else:
        reshaped_rgb_image = rgb_image.permute(2, 0, 1)
    reshaped_hsv_image = kc.rgb_to_hsv(reshaped_rgb_image)
    hsv_image = (reshaped_hsv_image / scale).permute(1, 2, 0)
    
    return hsv_image

def hsv_to_rgb(hsv_image: Tensor) -> Tensor: 
    """
    Transforms an HSV image into RGB space image.
    
    Requires a `[H, W, 3]` HSV image.
    Returns a `[H, W, 3]` RGB image.  
    """
    scale = NORMALIZATION_HSV_SCALE.to(hsv_image.device)
    reshaped_hsv_image = hsv_image.permute(2, 0, 1) * scale
    reshaped_rgb_image = kc.hsv_to_rgb(reshaped_hsv_image)
    rgb_image = reshaped_rgb_image.permute(1, 2, 0)
    
    return rgb_image    