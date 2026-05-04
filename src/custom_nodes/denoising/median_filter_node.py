import torch
from algorithms.denoising import median_filter

class MedianFilterNode:

    @classmethod
    def INPUT_TYPES(cls):
        
        tooltip : dict[str, str] = {
            "ksize": "Aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ..." 
        }

        return {
            "required": {
                "img": ("IMAGE",),
                "ksize": ("INT", {"default": 3, "min": 3, "tooltip": tooltip["ksize"]})
            }
        }
    
    CATEGORY = "image"
    SEARCH_ALIASES = [
        "median filter",
        "median blur",
        "salt and pepper noise removal",
        "despeckle",
        "non-linear smoothing",
        "outlier pixel removal",
        "noise reduction",
    ]

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGB_image",)
    FUNCTION = "execute"

    def execute(self, 
                img: torch.Tensor, 
                ksize: int
            ) -> tuple :
        
        input_2d = img.squeeze()
        output_2d = median_filter(input_2d, ksize)

        return (output_2d.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"MedianFilterNode": MedianFilterNode}

NODE_DISPLAY_NAME_MAPPINGS = {"MedianFilterNode": "Median filter denoising"}