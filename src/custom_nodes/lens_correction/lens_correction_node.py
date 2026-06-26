import torch

from algorithms.lens_correction._chromatic_aberration import (
    correct_chromatic_aberration,
)
from algorithms.lens_correction._distortion import correct_distortion
from algorithms.lens_correction._metadata import try_read_vignette_gain_map
from algorithms.lens_correction._vignetting import correct_vignetting


class LensCorrectionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Vignetting: polynomial radial gain model
                "vignette_alpha": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 2.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "vignette_beta": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 2.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                # Geometric distortion: Brown-Conrady radial model
                "distortion_k1": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.001,
                        "display": "slider",
                    },
                ),
                "distortion_k2": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -0.5,
                        "max": 0.5,
                        "step": 0.001,
                        "display": "slider",
                    },
                ),
                # Chromatic aberration: per-channel lateral scaling
                "ca_red_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.9,
                        "max": 1.1,
                        "step": 0.0005,
                        "display": "slider",
                    },
                ),
                "ca_blue_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.9,
                        "max": 1.1,
                        "step": 0.0005,
                        "display": "slider",
                    },
                ),
            },
            "optional": {
                # Supply the original RAW file path to auto-read DNG vignette
                # gain maps from OpcodeList metadata.  Leave empty to use the
                # manual vignette_alpha / vignette_beta sliders instead.
                "image_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/processing/lens-correction"
    SEARCH_ALIASES = [
        "lens correction",
        "optical correction",
        "vignetting",
        "distortion",
        "chromatic aberration",
        "barrel distortion",
        "pincushion",
        "CA correction",
    ]

    def process(
        self,
        image: torch.Tensor,
        vignette_alpha: float,
        vignette_beta: float,
        distortion_k1: float,
        distortion_k2: float,
        ca_red_scale: float,
        ca_blue_scale: float,
        image_path: str = "",
    ) -> tuple[torch.Tensor]:
        _B, H, W, _C = image.shape

        # try to load a vignette gain map from DNG metadata once per batch.
        gain_map = try_read_vignette_gain_map(image_path, H, W)

        frame = image.squeeze()  # (H, W, 3)
        frame = correct_vignetting(frame, vignette_alpha, vignette_beta, gain_map)
        frame = correct_distortion(frame, distortion_k1, distortion_k2)
        frame = correct_chromatic_aberration(frame, ca_red_scale, ca_blue_scale)

        return (frame.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "LensCorrectionNode": LensCorrectionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LensCorrectionNode": "Lens/Optical Correction",
}
