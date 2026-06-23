from torch import Tensor
from algorithms.color_manipulation._lut_color_manipulation import load_cube_lut
from algorithms.color_manipulation._lut_color_manipulation import (
    linearRGB_to_adobeRGB1998,
)
from algorithms.color_manipulation._lut_color_manipulation import (
    adobeRGB1998_to_linearRGB,
)
from algorithms.color_manipulation._lut_color_manipulation import apply_lut_grid_sample

RELATIVE_PATH = "files/luts_color_manipulation/ON1_Color_Boost_LUTs/"

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


class PersonalLutColorManipulationNode:
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
                "lut_path": ("STRING", {"default": "path/lut.cube"}),
                "color_space_lut": (
                    [
                        "Linear RGB",
                        "Adobe RGB (1998)",
                    ],
                    {"default": "Linear RGB"},
                ),
                "order_color_channels_lut": (
                    [
                        "RGB",
                        "BGR",
                    ],
                    {"default": "RGB"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "process"
    CATEGORY = "image/processing"

    def process(
        self,
        image: Tensor,
        color_space_image: str,
        order_color_channels_lut: str,
        lut_path: str,
        color_space_lut: str,
    ):
        image = image.squeeze(0)

        if lut_path.endswith(".cube"):
            lut = load_cube_lut(lut_path)

        if order_color_channels_lut == "BGR":
            lut = lut[..., [2, 1, 0]]

        if color_space_image == "Linear RGB" and color_space_lut == "Adobe RGB (1998)":
            image = linearRGB_to_adobeRGB1998(image)
            res = apply_lut_grid_sample(image, lut)
            res = adobeRGB1998_to_linearRGB(res)
        elif (
            color_space_image == "Adobe RGB (1998)" and color_space_lut == "Linear RGB"
        ):
            image = adobeRGB1998_to_linearRGB(image)
            res = apply_lut_grid_sample(image, lut)
            res = linearRGB_to_adobeRGB1998(res)
        else:
            res = apply_lut_grid_sample(image, lut)

        return (res.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "PersonalLutColorManipulationNode": PersonalLutColorManipulationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PersonalLutColorManipulationNode": "Personal LUT Color Manipulation",
}
