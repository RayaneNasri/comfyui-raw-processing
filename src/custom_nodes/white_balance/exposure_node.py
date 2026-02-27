import torch


class SimpleExposureNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure_ev": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -5.0,
                        "max": 5.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
            }
        }

    CATEGORY = "image/adjustment"
    SEARCH_ALIASES = ["exposure", "brightness", "gain"]

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"

    def execute(self, image: torch.Tensor, exposure_ev: float) -> tuple:
        multiplier = 2.0**exposure_ev
        out = image * multiplier
        out = torch.clamp(out, min=0.0, max=1.0)

        return (out,)


NODE_CLASS_MAPPINGS = {"SimpleExposureNode": SimpleExposureNode}

NODE_DISPLAY_NAME_MAPPINGS = {"SimpleExposureNode": "Simple Exposure"}
