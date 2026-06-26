import torch
from algorithms.denoising import gaussian_filter


class GaussianFilerNode:
    # TODO: Make the node more robust and more flexible

    @classmethod
    def INPUT_TYPES(cls):
        tooltip = {
            "kheight": "Gaussian kernel height",
            "kwidth": "Gaussian kernel width",
            "sigmaX": "Gaussian kernel standard deviation in X direction.",
            "sigmaY": "Gaussian kernel standard deviation in Y direction.",
            "borderType": "Pixel extrapolation method, see the modes available in the open-cv documentation.",
            "hint": "Implementation modfication flags. See AlgorithmHint in the open-cv documentation",
        }

        borderTypes: list[str] = [
            "BORDER_CONSTANT",
            "BORDER_REPLICATE",
            "BORDER_REFLECT",
            "BORDER_REFLECT_101",
            "BORDER_TRANSPARENT",
            "BORDER_DEFAULT",
            "BORDER_ISOLATED",
        ]

        hintTypes: list[str] = [
            "ALGO_HINT_DEFAULT",
            "ALGO_HINT_ACCURATE",
            "ALGO_HINT_APPROX",
        ]

        return {
            "required": {
                "image": ("IMAGE",),
                "kheight": ("INT", {"min": 0, "tooltip": tooltip["kheight"]}),
                "kwidth": ("INT", {"min": 0, "tooltip": tooltip["kwidth"]}),
                "sigmaX": ("FLOAT", {"min": 0.0, "tooltip": tooltip["sigmaX"]}),
                "sigmaY": ("FLOAT", {"min": 0.0, "tooltip": tooltip["sigmaY"]}),
                "borderType": (borderTypes,),
                "hint": (hintTypes,),
            }
        }

    CATEGORY = "image/processing/denoising"

    SEARCH_ALIASES = [
        "gaussian filter",
        "gaussian blur",
        "gaussian smoothing",
        "noise reduction",
        "low pass filter",
        "blur filter",
        "edge preserving smoothing",
    ]

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGB_image",)
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        kheight: int,
        kwidth: int,
        sigmaX: float,
        sigmaY: float,
        borderType: str,
        hint: str,
    ) -> tuple:

        input_2d = image.squeeze()
        output_2d = gaussian_filter(
            input_2d, (kheight, kwidth), sigmaX, sigmaY, borderType, hint
        )

        return (output_2d.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"GaussianFilerNode": GaussianFilerNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GaussianFilerNode": "Gaussian filter denoising"}
