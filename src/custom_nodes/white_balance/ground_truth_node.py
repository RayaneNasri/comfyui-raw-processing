import torch

from algorithms.white_balance import ground_truth


class GroundTruthNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "method": (["max", "mean"],),
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
        "manual white balance",
        "neutral point calibration",
        "raw color balancing",
        "illuminant estimation",
    ]

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("out",)
    FUNCTION = "execute"

    def execute(
        self, image: torch.Tensor, mask: torch.Tensor, method: str, percentil: float
    ):
        input_3d = image.squeeze()
        patch = input_3d[torch.where(mask > 0.0)]
        out = ground_truth(input_3d, patch, method, percentil)
        return (out.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"GroundTruthNode": GroundTruthNode}

NODE_DISPLAY_NAME_MAPPINGS = {"GroundTruthNode": "Ground Truth White Balance"}
