import torch
from torch import Tensor
from algorithms.color_manipulation._lut_color_manipulation import load_cube_lut
from algorithms.color_manipulation._lut_color_manipulation import linearRGB_to_adobeRGB1998
from algorithms.color_manipulation._lut_color_manipulation import adobeRGB1998_to_linearRGB
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
                "image": ("IMAGE",),
                "color_space_image": (
                [
                    "Linear RGB",
                    "Adobe RGB (1998)",
                ],
                {
                    "default": "Linear RGB"
                }
                ),
                "lut_name": (list(luts.keys()),),
                "apply_lut_from_lut_path": ("BOOLEAN", {
                    "default": False}),
                "lut_path": ("STRING", {"default": "path/lut.cube"}),
                "color_space_lut": (
                [
                    "Linear RGB",
                    "Adobe RGB (1998)",
                ],
                {
                    "default": "Linear RGB"
                }
                )
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "process"
    CATEGORY = "image/processing"
    
    def process(self, image: Tensor, color_space_image: str, lut_path: str, apply_lut_from_lut_path: bool, lut_name: str, color_space_lut: str):
        image = image.squeeze(0)
        
        if apply_lut_from_lut_path:
            if lut_path.endswith(".cube"):
                lut = load_cube_lut(lut_path)
            else:
                raise ValueError("Wrong format : expected a .cube lut")
        else:
            lut = load_cube_lut(luts[lut_name])
        
        if color_space_image == "Linear RGB" and (color_space_lut == "Adobe RGB (1998)" or apply_lut_from_lut_path == False):
            image = linearRGB_to_adobeRGB1998(image)
            res = apply_lut_grid_sample(image, lut)
            res = adobeRGB1998_to_linearRGB(res)
        elif color_space_image == "Adobe RGB (1998)" and (color_space_lut == "Linear RGB" and apply_lut_from_lut_path == True):
            image = adobeRGB1998_to_linearRGB(image)
            res = apply_lut_grid_sample(image, lut)
            res = linearRGB_to_adobeRGB1998(res)
        else:
            res = apply_lut_grid_sample(image, lut)

        return (res.unsqueeze(0),)
        

NODE_CLASS_MAPPINGS = {
    "LutColorManipulationNode": LutColorManipulationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LutColorManipulationNode": "LUT Color Manipulation",
}