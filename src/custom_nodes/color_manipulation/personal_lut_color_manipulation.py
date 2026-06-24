from aiohttp import web
from torch import Tensor
from server import PromptServer  # type: ignore
from algorithms.color_manipulation._lut_color_manipulation import load_cube_lut
from algorithms.color_manipulation._lut_color_manipulation import (
    linearRGB_to_adobeRGB1998,
)
from algorithms.color_manipulation._lut_color_manipulation import (
    adobeRGB1998_to_linearRGB,
)
from algorithms.color_manipulation._lut_color_manipulation import apply_lut_grid_sample

import asyncio
import os
import folder_paths  # type: ignore

_CUBE_DIR = os.path.join(folder_paths.models_dir, "luts")
os.makedirs(_CUBE_DIR, exist_ok=True)


@PromptServer.instance.routes.post("/personal_lut/upload_cube")
async def upload_cube(request):
    reader = await request.multipart()
    field = await reader.next()

    if field is None or not field.filename:
        return web.json_response({"error": "Aucun fichier reçu"}, status=400)

    filename = os.path.basename(field.filename)
    if not filename.lower().endswith(".cube"):
        return web.json_response({"error": "Le fichier doit être un .cube"}, status=400)

    dest = os.path.join(_CUBE_DIR, filename)
    tmp_dest = dest + ".part"

    # Write to a temporary file first so a partial upload never leaves a
    # corrupted .cube in the presets folder.
    try:
        with open(tmp_dest, "wb") as f:
            while chunk := await field.read_chunk():
                f.write(chunk)
        os.replace(tmp_dest, dest)  # atomic rename, only on success
    except (ConnectionResetError, asyncio.CancelledError, OSError):
        if os.path.isfile(tmp_dest):
            os.remove(tmp_dest)
        raise

    return web.json_response({"filename": filename, "path": dest})


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
                # This widget is hidden in the UI and driven by the JS file picker.
                # Its value is the absolute server-side path to the uploaded .cube file.
                "lut_path": ("STRING", {"default": ""}),
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

        if not lut_path:
            raise ValueError(
                "Aucun fichier .cube sélectionné. "
                "Utilisez le bouton « Ouvrir un fichier .cube… » dans le node."
            )

        if not lut_path.endswith(".cube"):
            raise ValueError("Invalid LUT file format. Only .cube files are supported.")

        if not os.path.isfile(lut_path):
            raise FileNotFoundError(f"Fichier LUT introuvable : {lut_path}")

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

WEB_DIRECTORY = "./web"
