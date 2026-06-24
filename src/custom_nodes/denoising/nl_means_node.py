import torch
from algorithms.denoising import nl_means


class NonLocalMeansNode:
    @classmethod
    def INPUT_TYPES(cls):

        tooltip: dict[str, str] = {
            "image": "Image to denoise",
            "h": "Parameter regulating filter strength for luminance component",
            "hColor": "The same as h but for color components",
            "templateWindowSize": "Size in pixels of the template patch that is used to compute weights. Should be odd",
            "searchWindowSize": "Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd",
        }

        return {
            "required": {
                "img": ("IMAGE",),
                "h": ("FLOAT", {"default": 3.0, "min": 0.0, "tooltip": tooltip["h"]}),
                "hColor": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "tooltip": tooltip["hColor"]},
                ),
                "templateWindowSize": (
                    "INT",
                    {"default": 7, "min": 0, "tooltip": tooltip["templateWindowSize"]},
                ),
                "searchWindowSize": (
                    "INT",
                    {"default": 21, "min": 0, "tooltip": tooltip["searchWindowSize"]},
                ),
            }
        }

    CATEGORY = "image/processing/denoising"
    SEARCH_ALIASES = [
        "non local means denoising",
        "fastNlMeansDenoisingColored",
        "noise reduction",
        "patch based denoising",
        "image smoothing",
        "detail preserving filter",
        "color image denoising",
        "texture smoothing",
    ]

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGB_image",)
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        h: int,
        hColor: int,
        templateWindowSize: int,
        searchWindowSize: int,
    ) -> tuple:

        input_2d = image.squeeze()
        output_2d = nl_means(input_2d, h, hColor, templateWindowSize, searchWindowSize)

        return (output_2d.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"NonLocalMeansNode": NonLocalMeansNode}

NODE_DISPLAY_NAME_MAPPINGS = {"NonLocalMeansNode": "Non local means denoising"}
