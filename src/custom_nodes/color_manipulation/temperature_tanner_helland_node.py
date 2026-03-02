import torch
from torch import Tensor
from algorithms.color_manipulation._temperature_tanner_helland import temperature_tanner_helland

class TemperatureTannerHellandNode: 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_image": ("IMAGE",),
                "temperature_K": ("FLOAT", {
                    "default": 6550.0, 
                    "min": 1000.0, 
                    "max": 40000.0, 
                    "step": 1.0,
                    "display": "slider"
                }),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/processing"
    
    def process(self, rgb_image: Tensor, temperature_K: float):
        res = temperature_tanner_helland(rgb_image, temperature_K)
        return (res,)
        

NODE_CLASS_MAPPINGS = {
    "TemperatureTannerHellandNode": TemperatureTannerHellandNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TemperatureTannerHellandNode": "Temperature Tanner-Helland",
}