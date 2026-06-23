import torch
from algorithms.white_balance import white_patch_ref


class WhitePatchReferenceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "percentil": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
            }
        }

    CATEGORY = "image"
    SEARCH_ALIASES = [
        "white balance adjustment",
        "color temperature correction",
        "white patch reference",
        "chromatic adaptation",
        "automatic white balance",
        "neutral point calibration",
        "raw color balancing",
        "illuminant estimation",
    ]

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGB_image",)
    FUNCTION = "execute"

    def execute(self, image: torch.Tensor, percentil) -> tuple:
        input_2d = image.squeeze()
        output_2d = white_patch_ref(input_2d, percentil)

        return (output_2d.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"WhitePatchReferenceNode": WhitePatchReferenceNode}

NODE_DISPLAY_NAME_MAPPINGS = {"WhitePatchReferenceNode": "White Patch Reference"}
