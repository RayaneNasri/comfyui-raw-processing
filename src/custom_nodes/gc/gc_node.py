import torch
from algorithms.gc import gamma_correction

class GammaCorrectionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gamma": ("FLOAT", {
                    "default": 2.2, 
                    "min": 0.01, 
                }),
                "alpha": ("FLOAT", {
                    "default": 1., 
                    "min": 0., 
                }),
            }
        }

    CATEGORY = "image"
    SEARCH_ALIASES = [
            "gamma correction adjustment",
            "luminance curve tweak",
            "midtone brightness control",
            "nonlinear intensity scaling",
            "display encoding transformation",
            "perceptual contrast tuning",
            "image exposure compensation",
            "sRGB transfer function",
            "power law transformation",
            "shadow and highlight recovery",
            "digital darkroom grading",
        ]

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGB_image",)
    FUNCTION = "execute"

    def execute(self, image: torch.Tensor, gamma, alpha) -> tuple :
        input_2d = image.squeeze()
        output_2d = gamma_correction(input_2d, gamma, alpha)

        return (output_2d.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"GammaCorrectionNode": GammaCorrectionNode}

NODE_DISPLAY_NAME_MAPPINGS = {"GammaCorrectionNode": "Gamma Correction"}
