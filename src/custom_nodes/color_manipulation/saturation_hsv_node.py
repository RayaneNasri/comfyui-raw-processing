from torch import Tensor
from algorithms.color_manipulation._saturation_hsv import saturation_hsv


class SaturationHSVNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_image": ("IMAGE",),
                "adjustement": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgb_image",)

    FUNCTION = "process"
    CATEGORY = "image/processing"

    def process(self, rgb_image: Tensor, adjustement: float):
        rgb_image = rgb_image.squeeze(0)
        res = saturation_hsv(rgb_image, adjustement)
        return (res.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "SaturationHSVNode": SaturationHSVNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaturationHSVNode": "Saturation HSV",
}
