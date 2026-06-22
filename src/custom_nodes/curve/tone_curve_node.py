from __future__ import annotations

import numpy as np
import torch

from algorithms.curve.curve_spec import CurveSpec
from algorithms.curve.curve_engine import apply_lut_torch, build_lut

from .curve_node_base import CurveNodeBase


class ToneCurveNode(CurveNodeBase):
    """Tone curve with per-channel control (Master + R + G + B).

    The master curve is applied first to all channels uniformly, then
    each per-channel curve is applied independently on top.
    """

    CATEGORY = "image/curves"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_size": (
                    "INT",
                    {"default": 256, "min": 64, "max": 4096, "step": 64},
                ),
                "curve_master": ("CURVE", {"default": "[]"}),
                "curve_r": ("CURVE", {"default": "[]"}),
                "curve_g": ("CURVE", {"default": "[]"}),
                "curve_b": ("CURVE", {"default": "[]"}),
            },
            "optional": {
                "preset_name": ("STRING", {"default": ""}),
                "save_preset_as": ("STRING", {"default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    # --- Abstract interface ---

    def _default_spec(self, **kwargs) -> CurveSpec:
        return CurveSpec.identity(n_points=5)

    def _apply_lut(self, image: torch.Tensor, lut: np.ndarray) -> torch.Tensor:
        # Requis par l'ABC — délègue au master
        return apply_lut_torch(image, lut)

    def _compute_histogram_data(self, image_tensor: torch.Tensor) -> list:
        try:
            img = image_tensor[0] if image_tensor.ndim == 4 else image_tensor

            if img.shape[-1] >= 3:
                weights = torch.tensor(
                    [0.2126, 0.7152, 0.0722], dtype=img.dtype, device=img.device
                )
                luma = (img[..., :3] * weights).sum(dim=-1)
            else:
                luma = img[..., 0]

            hist = torch.histc(luma, bins=256, min=0.0, max=1.0)
            return hist.cpu().tolist()

        except Exception as e:
            print(f"[Artishow Curve] Erreur histogramme: {e}")
            return []

    # --- Override execute() ---

    def execute(self, image: torch.Tensor, lut_size: int, **kwargs) -> tuple:
        node_id = kwargs.get("unique_id")
        curve_master = kwargs.get("curve_master", "[]")
        curve_r = kwargs.get("curve_r", "[]")
        curve_g = kwargs.get("curve_g", "[]")
        curve_b = kwargs.get("curve_b", "[]")
        preset_name = kwargs.get("preset_name", "")
        save_preset_as = kwargs.get("save_preset_as", "")

        # Presets s'appliquent au master uniquement
        if preset_name.strip():
            from .curve_presets import load_preset

            spec_master = load_preset(preset_name.strip())
        else:
            spec_master = self._parse_spec(curve_master)

        spec_r = self._parse_spec(curve_r)
        spec_g = self._parse_spec(curve_g)
        spec_b = self._parse_spec(curve_b)

        lut_master = build_lut(spec_master, lut_size)
        lut_r = build_lut(spec_r, lut_size)
        lut_g = build_lut(spec_g, lut_size)
        lut_b = build_lut(spec_b, lut_size)

        if preset_name.strip():
            from .curve_presets import save_preset

            save_preset(spec_master, preset_name.strip())

        img = image.squeeze(0)

        # Master en premier, puis canaux individuels
        if not self._is_identity(lut_master):
            img = apply_lut_torch(img, lut_master)
        if not self._is_identity(lut_r):
            img = self._apply_lut_channel(img, lut_r, channel=0)
        if not self._is_identity(lut_g):
            img = self._apply_lut_channel(img, lut_g, channel=1)
        if not self._is_identity(lut_b):
            img = self._apply_lut_channel(img, lut_b, channel=2)

        if node_id:
            self._send_histogram_to_ui(img, node_id)

        return (img.unsqueeze(0),)

    # --- Helpers ---

    def _apply_lut_channel(
        self, image: torch.Tensor, lut: np.ndarray, channel: int
    ) -> torch.Tensor:
        """Applique une LUT sur un seul canal, sans toucher aux autres."""
        lut_t = torch.from_numpy(lut).float().to(image.device)
        result = image.clone()
        indices = (image[..., channel] * (len(lut) - 1)).long().clamp(0, len(lut) - 1)
        result[..., channel] = lut_t[indices]
        return result

    def _is_identity(self, lut: np.ndarray) -> bool:
        """Vérifie si la LUT est une identité — skip si aucun effet."""
        return np.allclose(lut, np.linspace(0, 1, len(lut)), atol=1e-4)


NODE_CLASS_MAPPINGS = {"ToneCurveNode": ToneCurveNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ToneCurveNode": "Tone Curve"}
