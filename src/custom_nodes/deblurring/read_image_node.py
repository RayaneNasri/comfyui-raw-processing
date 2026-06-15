import torch
from torch import Tensor
from algorithms.deblurring._read_image import read_image

class ReadImageNode: 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_image_name": ("STRING",),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgb_image",)

    FUNCTION = "process"
    CATEGORY = "image/processing"
    
    def process(self, rgb_image_name: str):
        res = read_image(rgb_image_name)
        return (res.unsqueeze(0),)
        

NODE_CLASS_MAPPINGS = {
    "ReadImageNode": ReadImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReadImageNode": "Read Image",
}