from algorithms.hue_saturation_map._hue_saturation_map import apply_hue_sat_map
from algorithms.tools._lut_tools import read_hue_sat_lut_from_dcp
from torch import Tensor

import torch

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
        
        (
            low_temp_lut, 
            high_temp_lut, 
            indoor_color_matrix, 
            daylight_color_matrix, 
            forward_matrix_1, 
            forward_matrix_2, 
            calib_illum_1, 
            calib_illum_2
        ) = res
        
        results = []
        for i in range(rgb_image.shape[0]):
            frame = apply_hue_sat_map(
                rgb_image[i],  # [H, W, C]
                wb_gains, 
                indoor_color_matrix,
                daylight_color_matrix,
                forward_matrix_1,
                forward_matrix_2,
                low_temp_lut,
                high_temp_lut,
                calib_illum_1,
                calib_illum_2
            )
            results.append(frame)
        
        return (torch.stack(results),)  # [B, H, W, C]
        

NODE_CLASS_MAPPINGS = {
    "HueSaturationMapNode": HueSaturationMapNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HueSaturationMapNode": "Hue/Saturation Camera Profile Correction",
}