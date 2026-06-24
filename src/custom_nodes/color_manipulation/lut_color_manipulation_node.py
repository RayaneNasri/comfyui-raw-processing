from torch import Tensor
from algorithms.color_manipulation._lut_color_manipulation import load_cube_lut
from algorithms.color_manipulation._lut_color_manipulation import (
    linearRGB_to_adobeRGB1998,
)
from algorithms.color_manipulation._lut_color_manipulation import (
    adobeRGB1998_to_linearRGB,
)
from algorithms.color_manipulation._lut_color_manipulation import apply_lut_grid_sample

RELATIVE_PATH = "resources/lut_presets/"

luts = {
    "Aqua": RELATIVE_PATH + "Aqua.cube",
    "Aqua and Orange Dark": RELATIVE_PATH + "Aqua_and_Orange_Dark.cube",
    "Blues": RELATIVE_PATH + "Blues.cube",
    "Earth Tone Boost": RELATIVE_PATH + "Earth_Tone_Boost.cube",
    "Green Blues": RELATIVE_PATH + "Green_Blues.cube",
    "Green Yellow": RELATIVE_PATH + "Green_Yellow.cube",
    "Oranges": RELATIVE_PATH + "Oranges.cube",
    "Purple": RELATIVE_PATH + "Purple.cube",
    "Reds": RELATIVE_PATH + "Reds.cube",
    "Reds Oranges Yellows": RELATIVE_PATH + "Reds_Oranges_Yellows.cube",
}


class LutColorManipulationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_space_image": (
                    [
                        "Linear RGB",
                        "Adobe RGB (1998)",
                    ],
                    {"default": "Linear RGB"},
                ),
                "lut_name": (list(luts.keys()),),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "process"
    CATEGORY = "image/processing"

    def process(self, image: Tensor, color_space_image: str, lut_name: str):
        image = image.squeeze(0)

        lut = load_cube_lut(luts[lut_name])

        # lut BGR -> RGB
        lut = lut[..., [2, 1, 0]]

        if color_space_image == "Linear RGB":
            image = linearRGB_to_adobeRGB1998(image)
            res = apply_lut_grid_sample(image, lut)
            res = adobeRGB1998_to_linearRGB(res)
        else:  # color_space_image == "Linear RGB"
            res = apply_lut_grid_sample(image, lut)

        return (res.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "LutColorManipulationNode": LutColorManipulationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LutColorManipulationNode": "LUT Color Manipulation",
}
