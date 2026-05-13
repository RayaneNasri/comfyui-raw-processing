import torch
from torch import Tensor
from algorithms.color_manipulation._temperature_simple import temperature_simple

class TemperatureSimpleNode: 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_image": ("IMAGE",),
                "adjustement": ("FLOAT", {
                    "default": 0, 
                    "min": -100,
                    "max": 100, 
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgb_image",)

    FUNCTION = "process"
    CATEGORY = "image/processing"
    
    def process(self, rgb_image: Tensor, adjustement: float):
        rgb_image = rgb_image.squeeze(0)
        res = temperature_simple(rgb_image, adjustement)
        return (res.unsqueeze(0),)
        

NODE_CLASS_MAPPINGS = {
    "TemperatureSimpleNode": TemperatureSimpleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TemperatureSimpleNode": "Temperature Simple",
}