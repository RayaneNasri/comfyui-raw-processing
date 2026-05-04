import torch
from src.algorithms.denoising import avg_filter

    # img : torch.Tensor
    #     Input image
    # ksize : tuple
    #     Blurring kernel size
    # strBorderType : str
    #     Border mode used to extrapolate pixels outside of the image, see the modes available in the open-cv documentation

    # Returns
    # -------
    # img_wb : torch.Tensor
    #     Filtred Image

class AvgFilterNode:

    @classmethod
    def INPUT_TYPES(cls):
        
        tooltip: dict[str, str] = {
            "kheight": "Blurring kernel height",
            "kwidth": "Blurring kernel width",
            "BorderType": "Border mode used to extrapolate pixels outside of the image, see the modes available in the open-cv documentation"
        }

        borderTypes: list[str] = [
            "BORDER_CONSTANT",
            "BORDER_REPLICATE",
            "BORDER_REFLECT",
            "BORDER_REFLECT_101",
            "BORDER_TRANSPARENT",
            "BORDER_DEFAULT",
            "BORDER_ISOLATED"
        ]

        return {
            "required": {
                "image": ("IMAGE",),
                "kheight": ("INT", {"min": 1, "tooltip": tooltip["kheight"]}),
                "kwidth": ("INT", {"min": 1, "tooltip": tooltip["kwidth"]}),
                "borderType": (borderTypes,)
            }
        }
    
    CATEGORY = "image"
    
    SEARCH_ALIASES = [
        "average filter",
        "mean filter",
        "box blur",
        "image smoothing",
        "noise reduction",
        "low pass filter",
        "blur filter",
        "uniform filter",
    ]

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGB_image",)
    FUNCTION = "execute"

    def execute(self, 
                image: torch.Tensor, 
                kheight: int,
                kwidth: int,
                borderType: str
            ) -> tuple :
        
        input_2d = image.squeeze()
        output_2d = avg_filter(input_2d, (kheight, kwidth), borderType)

        return (output_2d.unsqueeze(0),)
    
NODE_CLASS_MAPPINGS = {"AvgFilterNode": AvgFilterNode}

NODE_DISPLAY_NAME_MAPPINGS = {"AvgFilterNode": "Average filter denoising"}