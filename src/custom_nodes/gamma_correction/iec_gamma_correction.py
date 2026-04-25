import torch
from algorithms.gc.iec_gamma_correction import iec_gamma_correction


class IECGammaCorrectionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
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

    def execute(self, image: torch.Tensor) -> tuple:
        input_2d = image.squeeze()
        output_2d = iec_gamma_correction(input_2d)

        return (output_2d.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"IECGammaCorrectionNode": IECGammaCorrectionNode}

NODE_DISPLAY_NAME_MAPPINGS = {"IECGammaCorrectionNode": "IEC Gamma Correction"}
