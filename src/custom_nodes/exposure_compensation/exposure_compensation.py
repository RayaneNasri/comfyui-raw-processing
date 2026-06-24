from torch import Tensor
from algorithms.exposure_compensation._exposure_compensation import (
    exposure_compensation,
)


class ExposureCompensationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ev_compensation": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/processing/exposure-compensation"

    def process(self, image: Tensor, ev_compensation: float):
        
        input2d = image.squeeze()
        res = exposure_compensation(input2d, ev_compensation)
        return (res.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "ExposureCompensationNode": ExposureCompensationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExposureCompensationNode": "Exposure Compensation",
}
