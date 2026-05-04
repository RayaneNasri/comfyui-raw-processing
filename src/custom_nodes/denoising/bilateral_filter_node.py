import torch
from algorithms.denoising import bilateral_filter

class BilatFilterNode:

    @classmethod
    def INPUT_TYPES(cls):

        tooltip: dict[str, str] = {
            "d": 
                "Diameter of each pixel neighborhood that is used during filtering.\
                If it is non-positive, it is computed from sigmaSpace.",
            "sigmaColor": 
                "Filter sigma in the color space. \
                A larger value of the parameter means that farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color.",
            "sigmaSpace": 
                "Filter sigma in the coordinate space.\
                A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough.",
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
            "image": ("IMAGE",),
            "d": ("INT", {"tooltip": tooltip["d"]}),
            "sigmaColor": ("FLOAT", {"min": 0., "tooltip": tooltip["sigmaColor"]}),
            "sigmaSpace": ("FLOAT", {"min": 0., "tooltip": tooltip["sigmaSpace"]}),
            "borderType": (borderTypes,)
        }
    
    CATEGORY = "image"

    SEARCH_ALIASES = [
        "bilateral filter",
        "edge preserving smoothing",
        "bilateral blur",
        "noise reduction with edge preservation",
        "surface blur",
        "smoothing filter",
    ]

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGB_image",)
    FUNCTION = "execute"

    def execute(self, 
                image: torch.Tensor, 
                d: int,
                sigmaColor: float,
                sigmaSpace: float,
                borderType: str
            ) -> tuple :
        
        input_2d = image.squeeze()
        output_2d = bilateral_filter(input_2d, d, sigmaColor, sigmaSpace, borderType)

        return (output_2d.unsqueeze(0),)
    
NODE_CLASS_MAPPINGS = {"BilatFilterNode": BilatFilterNode}

NODE_DISPLAY_NAME_MAPPINGS = {"BilatFilterNode": "Bilateral filter denoising"}

