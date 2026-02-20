import torch

from torch import Tensor
from algorithms.exposure_compensation import exposure_compensation

class ExposureCompensationNode: 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "ev_compensation": ("FLOAT", {
                    "default": 0.0, 
                    "min": -10.0, 
                    "max": 10.0, 
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/processing"
    
    def process(self, image: Tensor, ev_compensation: float):
        
        res = exposure_compensation(image, ev_compensation)
        return (res,)

NODE_CLASS_MAPPINGS = {
    "ExposureCompensationNode": ExposureCompensationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExposureCompensationNode": "Exposure Compensation",
}