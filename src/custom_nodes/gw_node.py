import torch
from algorithms.white_balance._gray_world import gw

class GrayWorldNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    CATEGORY = "image"
    SEARCH_ALIASES = [
        "white balance adjustment",     
        "color temperature correction", 
        "gray world",        
        "chromatic adaptation",         
        "automatic white balance",                  
        "neutral point calibration",    
        "raw color balancing",         
        "illuminant estimation",         
    ]

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGB_image",)
    FUNCTION = "execute"

    def execute(self, image: torch.Tensor) -> tuple :
        input_2d = image.squeeze()
        output_2d = gw(input_2d)

        return (output_2d.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"GrayWorldNode": GrayWorldNode}

NODE_DISPLAY_NAME_MAPPINGS = {"GrayWorldNode": "Gray World White Balance"}
