from aiohttp import web
from algorithms.hue_saturation_map._hue_saturation_map import apply_hue_sat_map
from algorithms.tools._lut_tools import read_hue_sat_lut_from_dcp
from torch import Tensor
from server import PromptServer  # type: ignore

import asyncio
import os
import folder_paths  # type: ignore
import json

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_ALIASES_FILE = os.path.join(_NODE_DIR, "resources", "dcp_aliases.json")

_DCP_DIR = os.path.join(folder_paths.models_dir, "dcp")
os.makedirs(_DCP_DIR, exist_ok=True)

if "dcp" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["dcp"] = ([_DCP_DIR], {".dcp"})


def _load_aliases() -> dict[str, list[str]]:
    if not os.path.isfile(_ALIASES_FILE):
        return {}
    with open(_ALIASES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_alias_map(aliases: dict[str, list[str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for canonical, alias_list in aliases.items():
        result[canonical] = canonical
        for alias in alias_list:
            result[alias] = canonical
    return result


_ALIASES: dict[str, list[str]] = _load_aliases()
_ALIAS_MAP: dict[str, str] = _build_alias_map(_ALIASES)


def _available_stems() -> list[str]:
    try:
        files = folder_paths.get_filename_list("dcp")
    except Exception:
        files = []
    return sorted(os.path.splitext(f)[0] for f in files) if files else []


def _display_labels(stems: list[str]) -> list[str]:
    labels = []
    for stem in stems:
        aliases = _ALIASES.get(stem, [])
        if aliases:
            shown = ", ".join(aliases[:2])
            suffix = f"  (+{len(aliases) - 2} autres)" if len(aliases) > 2 else ""
            labels.append(f"{stem}  ({shown}{suffix})")
        else:
            labels.append(stem)
    return labels


_STEMS: list[str] = _available_stems()
_LABELS: list[str] = _display_labels(_STEMS)
_LABEL_TO_STEM: dict[str, str] = dict(zip(_LABELS, _STEMS))


@PromptServer.instance.routes.get("/hue_sat_map/dcp_presets")
async def get_dcp_presets(request):
    return web.json_response(
        {
            "presets": [
                {"label": label, "stem": stem} for label, stem in _LABEL_TO_STEM.items()
            ]
        }
    )


@PromptServer.instance.routes.post("/hue_sat_map/upload_dcp")
async def upload_dcp(request):
    reader = await request.multipart()
    field = await reader.next()

    if field is None or not field.filename:
        return web.json_response({"error": "Aucun fichier reçu"}, status=400)

    filename = os.path.basename(field.filename)
    if not filename.lower().endswith(".dcp"):
        return web.json_response({"error": "Le fichier doit être un .dcp"}, status=400)

    dest = os.path.join(_DCP_DIR, filename)
    tmp_dest = dest + ".part"

    # On écrit d'abord dans un fichier temporaire : si la connexion est
    # coupée en plein transfert (client qui ferme l'onglet, rechargement,
    # serveur qui redémarre...), on ne veut surtout pas qu'un .dcp à moitié
    # écrit traîne dans le dossier des presets et soit proposé comme un
    # profil valide alors qu'il est corrompu.
    try:
        with open(tmp_dest, "wb") as f:
            while chunk := await field.read_chunk():
                f.write(chunk)
        os.replace(
            tmp_dest, dest
        )  # rename atomique, uniquement si l'upload est complet
    except (ConnectionResetError, asyncio.CancelledError, OSError):
        if os.path.isfile(tmp_dest):
            os.remove(tmp_dest)
        raise

    # Rafraîchir folder_paths pour que le fichier soit immédiatement résolvable
    folder_paths.folder_names_and_paths["dcp"] = ([_DCP_DIR], {".dcp"})

    return web.json_response({"filename": filename})


def _resolve_dcp_path(mode: str, value: str) -> str:
    """
    mode="preset" value est un stem canonique
    mode="custom" value est un nom de fichier (basename) uploadé dans _DCP_DIR
    """
    if mode == "preset":
        stem = _LABEL_TO_STEM.get(value, value)
        path = folder_paths.get_full_path("dcp", f"{stem}.dcp")
    else:
        path = os.path.join(_DCP_DIR, value)

    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"Fichier DCP introuvable : {value}\nDossier : {_DCP_DIR}"
        )
    return path


def resolve_dcp_path_for_model(model_name: str) -> str | None:
    canonical = _ALIAS_MAP.get(model_name)
    if canonical is None:
        return None
    path = folder_paths.get_full_path("dcp", f"{canonical}.dcp")
    return path if path and os.path.isfile(path) else None


class HueSaturationMapNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_image": ("IMAGE",),
                "wb_gains": ("WB_GAIN",),
                # Valeur opaque gérée entièrement par le widget JS :
                # format   "preset:<label>"  ou  "custom:<filename>"
                "dcp_selection": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/processing/hue-saturation-map"

    def process(
        self,
        rgb_image: Tensor,
        wb_gains: Tensor,
        dcp_selection: str,
    ) -> tuple[Tensor]:

        if ":" not in dcp_selection:
            raise ValueError(
                f"dcp_selection mal formé : {dcp_selection!r}\n"
                "Attendu : 'preset:<label>' ou 'custom:<filename>'"
            )

        mode, _, value = dcp_selection.partition(":")
        if mode not in ("preset", "custom"):
            raise ValueError(f"Mode inconnu : {mode!r}")
        dcp_path = _resolve_dcp_path(mode, value)

        input2d = rgb_image.squeeze()
        res = read_hue_sat_lut_from_dcp(dcp_path)

        if res is None:
            raise ValueError(f"Impossible de lire le fichier DCP : {dcp_path}")

        (
            low_temp_lut,
            high_temp_lut,
            indoor_color_matrix,
            daylight_color_matrix,
            forward_matrix_1,
            forward_matrix_2,
            calib_illum_1,
            calib_illum_2,
        ) = res

        frame = apply_hue_sat_map(
            input2d,
            wb_gains,
            indoor_color_matrix,
            daylight_color_matrix,
            forward_matrix_1,
            forward_matrix_2,
            low_temp_lut,
            high_temp_lut,
            calib_illum_1,
            calib_illum_2,
        )

        return (frame.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "HueSaturationMapNode": HueSaturationMapNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HueSaturationMapNode": "Hue/Saturation Camera Profile Correction",
}

WEB_DIRECTORY = "./web"
