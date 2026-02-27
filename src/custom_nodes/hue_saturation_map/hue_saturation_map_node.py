from algorithms.hue_saturation_map._hue_saturation_map import apply_hue_saturation_map
from algorithms.tools._lut_tools import read_hue_sat_lut_from_dcp
from torch import Tensor

class HueSaturationMapNode: 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_image": ("IMAGE",),
                "wb_gains": ("WB_GAIN",),
                "dcp_path": ("STRING",),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/processing"
    
    def process(self, rgb_image: Tensor, wb_gains: Tensor, dcp_path: str):
        res = read_hue_sat_lut_from_dcp(dcp_path)
        
        if res is None: 
            raise ValueError
        
        low_temp_lut, high_temp_lut, indoor_color_matrix, daylight_color_matrix, calib_illum_1, calib_illum_2 = res
        final_rgb_image = apply_hue_saturation_map(
            rgb_image, 
            wb_gains,
            low_temp_lut, 
            high_temp_lut, 
            indoor_color_matrix, 
            daylight_color_matrix,
            calib_illum_1,
            calib_illum_2
        )
        
        return (final_rgb_image,)
        

NODE_CLASS_MAPPINGS = {
    "HueSaturationMapNode": HueSaturationMapNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HueSaturationMapNode": "Hue/Saturation Camera Profile Correction",
}