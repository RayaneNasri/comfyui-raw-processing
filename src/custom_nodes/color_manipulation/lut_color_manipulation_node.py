import torch
from torch import Tensor
from algorithms.color_manipulation._lut_color_manipulation import load_cube_lut
from algorithms.color_manipulation._lut_color_manipulation import apply_lut_grid_sample

RELATIVE_PATH = "files/luts_color_manipulation/ON1_Color_Boost_LUTs/"

luts = {
    "Aqua" : RELATIVE_PATH + "Aqua.cube",
    "Aqua and Orange Dark" : RELATIVE_PATH + "Aqua_and_Orange_Dark.cube",
    "Blues" : RELATIVE_PATH + "Blues.cube",
    "Earth Tone Boost" : RELATIVE_PATH + "Earth_Tone_Boost.cube",
    "Green Blues" : RELATIVE_PATH + "Green_Blues.cube",
    "Green Yellow" : RELATIVE_PATH + "Green_Yellow.cube",
    "Oranges" : RELATIVE_PATH + "Oranges.cube",
    "Purple" : RELATIVE_PATH + "Purple.cube",
    "Reds" : RELATIVE_PATH + "Reds.cube",
    "Reds Oranges Yellows" : RELATIVE_PATH + "Reds_Oranges_Yellows.cube"
}

class LutColorManipulationNode: 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_image": ("IMAGE",),
                "lut_path": ("STRING", {"default": "path/lut.cube"}),
                "apply_lut_from_lut_path": ("BOOLEAN", {
                    "default": False}),
                "lut_name": (list(luts.keys()),)
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgb_image",)

    FUNCTION = "process"
    CATEGORY = "image/processing"
    
    def process(self, rgb_image: Tensor, lut_path: str, apply_lut_from_lut_path: bool, lut_name: str):
        if apply_lut_from_lut_path:
            if lut_path.endswith(".cube"):
                lut = load_cube_lut(lut_path)
                res = apply_lut_grid_sample(rgb_image, lut)
                return (res,)
            else:
                raise ValueError("Wrong format : expected a .cube lut")
        else:
            lut = load_cube_lut(luts[lut_name])
            res = apply_lut_grid_sample(rgb_image, lut)
            return (res,)
        

NODE_CLASS_MAPPINGS = {
    "LutColorManipulationNode": LutColorManipulationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LutColorManipulationNode": "LUT Color Manipulation",
}