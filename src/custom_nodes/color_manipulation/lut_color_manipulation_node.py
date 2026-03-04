import torch
from torch import Tensor
from algorithms.color_manipulation._lut_color_manipulation import load_cube_lut
from algorithms.color_manipulation._lut_color_manipulation import apply_lut_trilinear_interpolation

class LutColorManipulationNode: 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_image": ("IMAGE",),
                # "LUT": , # TODO : choose one LUT in a list of LUTs
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgb_image",)

    FUNCTION = "process"
    CATEGORY = "image/processing"
    
    def process(self, rgb_image: Tensor): # TODO : add the chosen LUT
        lut = load_cube_lut("/mnt/c/Users/charl/git/Blues.cube") # TODO : change path
        res = apply_lut_trilinear_interpolation(rgb_image, lut)
        return (res,)
        

NODE_CLASS_MAPPINGS = {
    "LutColorManipulationNode": LutColorManipulationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LutColorManipulationNode": "LUT Color Manipulation",
}