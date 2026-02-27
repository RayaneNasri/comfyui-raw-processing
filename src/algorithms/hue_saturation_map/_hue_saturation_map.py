from torch import Tensor 

def _mired_value_from_calibration_illuminant(calib_illum: int) -> float:
    factor = 1_000_00
    illuminant_to_kelvin = {
        1: 5500,  # Daylight
        3: 3200,  # Tungsten
        17: 2856, # Standard Illuminant A
        20: 5503, # D55
        21: 6504, # D65
        23: 5003  # D50
    }
    
    temperature = illuminant_to_kelvin.get(calib_illum)
    if temperature is None: 
        raise Exception # TODO : implement more formal exception errors raising (define isp_exceptions.py file and define them)
    
    return factor / temperature

def apply_hue_saturation_map(rgb_image: Tensor, wb_gains: Tensor, low_temp_lut: Tensor, high_temp_lut: Tensor, color_matrix: Tensor) -> Tensor: 
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
    
    