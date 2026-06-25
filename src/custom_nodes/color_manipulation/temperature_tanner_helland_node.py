from torch import Tensor
from algorithms.color_manipulation._temperature_tanner_helland import (
    temperature_tanner_helland,
)


class TemperatureTannerHellandNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_image": ("IMAGE",),
                "temperature_Kelvin": (
                    "FLOAT",
                    {
                        "default": 6600.0,
                        "min": 1000.0,
                        "max": 40000.0,
                        "step": 1.0,
                        "display": "slider",
                    },
                ),
                "apply_changes": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgb_image",)

    FUNCTION = "process"
    CATEGORY = "image/processing/color-manipulation"

    def process(
        self, rgb_image: Tensor, temperature_Kelvin: float, apply_changes: bool
    ):
        rgb_image = rgb_image.squeeze(0)

        if apply_changes:
            res = temperature_tanner_helland(rgb_image, temperature_Kelvin)
        else:
            res = rgb_image

        return (res.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "TemperatureTannerHellandNode": TemperatureTannerHellandNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TemperatureTannerHellandNode": "Temperature Tanner-Helland",
}
