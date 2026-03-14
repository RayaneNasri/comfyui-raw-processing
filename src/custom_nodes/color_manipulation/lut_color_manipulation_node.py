import torch
from torch import Tensor
from algorithms.color_manipulation._lut_color_manipulation import load_cube_lut
from algorithms.color_manipulation._lut_color_manipulation import apply_lut_grid_sample

class LutColorManipulationNode: 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_image": ("IMAGE",),
                "lut_path": ("STRING", {"default": "input/lut.cube"})
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgb_image",)

    FUNCTION = "process"
    CATEGORY = "image/processing"
    
    def process(self, rgb_image: Tensor, lut_path: str):
        if lut_path.endswith(".cube"):
            lut = load_cube_lut(lut_path)
            res = apply_lut_grid_sample(rgb_image, lut)
            return (res,)
        else:
            raise ValueError("Wrong format : expected a .cube lut")
        

NODE_CLASS_MAPPINGS = {
    "LutColorManipulationNode": LutColorManipulationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LutColorManipulationNode": "LUT Color Manipulation",
}