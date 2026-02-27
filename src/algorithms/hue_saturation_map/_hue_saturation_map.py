from torch import Tensor 

def apply_hue_saturation_map(rgb_image: Tensor, wb_gains: Tensor, low_temp_lut: Tensor, high_temp_lut: Tensor) -> Tensor: 
    """
    Applies the camera profile `HueSatMap` LUTs depending on white balance gains.
    
    Arguments.
    - `rgb_image` : image to apply hue/saturation map, requires image of shape [H, W, 3].
    - `wb_gains` : white balance gains.
    - `low_temp_lut` : low temperature LUT, referred as `ProfileHueSatMap1` in DCP files.
    - `high_temp_lut` : high temperature LUT, referred as `ProfileHueSatMap2` in DCP files.
    
    Returns. 
    - RGB image of shape [H, W, 3]
    """
    
    raise NotImplemented()
    
    