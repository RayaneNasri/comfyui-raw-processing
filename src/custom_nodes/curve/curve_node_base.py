from __future__ import annotations

import json
from abc import ABC, abstractmethod

import numpy as np
import torch
import server  # type: ignore

from algorithms.curve.curve_engine import build_lut
from algorithms.curve.curve_spec import CurveSpec

from .curve_presets import load_preset, save_preset


class CurveNodeBase(ABC):
    """Abstract ComfyUI node for all 1-D curve operations.

    Subclasses must implement:
      _default_spec(**kwargs) -> CurveSpec
          Return the initial curve when no control points have been drawn.
          Extra **kwargs come from the node's INPUT_TYPES (e.g. gamma_preset).

      _apply_lut(image, lut) -> torch.Tensor
          Apply the 1-D LUT to the image.  Receives a (H, W, C) tensor.

    Subclasses may extend INPUT_TYPES by calling super().INPUT_TYPES() and
    adding to the returned dict.
    """

    CATEGORY = "image/processing/curve-manipulation"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"

    WEB_DIRECTORY = "./js"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "curve_points": ("CURVE", {"default": "[]"}),
                "lut_size": (
                    "INT",
                    {"default": 256, "min": 64, "max": 4096, "step": 64},
                ),
            },
            "optional": {
                "preset_name": ("STRING", {"default": ""}),
                "save_preset_as": ("STRING", {"default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # trick to force ComfyUI to re-render the node's UI when the preset list changes
        return float("nan")

    # Abstract interface
    @abstractmethod
    def _default_spec(self, **kwargs) -> CurveSpec:
        """Return the CurveSpec to use when no control points are provided."""

    @abstractmethod
    def _apply_lut(self, image: torch.Tensor, lut: np.ndarray) -> torch.Tensor:
        """Apply the LUT to the (H, W, C) image tensor."""

    def _parse_spec(self, curve_points: str, **kwargs) -> CurveSpec:
        try:
            data = json.loads(curve_points) if curve_points.strip() else []
        except (json.JSONDecodeError, AttributeError):
            data = []

        if not data:
            return self._default_spec(**kwargs)

        if isinstance(data, list):
            return CurveSpec(points=[(float(x), float(y)) for x, y in data])

        return CurveSpec.from_json(data)

    @abstractmethod
    def _compute_histogram_data(self, image_tensor: torch.Tensor) -> list:
        """Calcule les données de l'histogramme.
        Par défaut : calcule la luminance globale (Rec. 709).
        Peut être réécrite dans les sous-classes.
        """

    def _send_histogram_to_ui(self, image_tensor: torch.Tensor, node_id: str):
        """Calcule l'histogramme de luminance et l'envoie au Front-End via WebSockets."""
        try:
            img_np = image_tensor.cpu().numpy()

            if img_np.ndim == 4:
                img_np = img_np[0]

            if img_np.shape[-1] >= 3:
                luma = (
                    0.2126 * img_np[..., 0]
                    + 0.7152 * img_np[..., 1]
                    + 0.0722 * img_np[..., 2]
                )
            else:
                luma = img_np[..., 0]

            counts, _ = np.histogram(luma, bins=256, range=(0.0, 1.0))
            hist_list = counts.tolist()

            # Envoi via l'API WebSocket de ComfyUI
            # self.id est fourni dynamiquement par l'exécuteur de prompt de ComfyUI

            server.PromptServer.instance.send_sync(
                "artishow-update-histogram", {"node_id": node_id, "hist": hist_list}
            )
        except Exception as e:
            print(f"Échec du calcul de l'histogramme: {e}")

    def execute(self, image: torch.Tensor, lut_size: int, **kwargs) -> tuple:
        node_id = kwargs.get("unique_id")
        curve_points = kwargs.get("curve_points", "[]")
        preset_name = kwargs.get("preset_name", "")
        save_preset_as = kwargs.get("save_preset_as", "")

        if preset_name.strip():
            spec = load_preset(preset_name.strip())
        else:
            spec = self._parse_spec(curve_points, **kwargs)

        if save_preset_as.strip():
            save_preset(spec, save_preset_as.strip())

        if node_id:
            self._send_histogram_to_ui(image, node_id)

        lut = build_lut(spec, lut_size)
        img = image.squeeze(0)
        result = self._apply_lut(img, lut)
        return (result.unsqueeze(0),)
