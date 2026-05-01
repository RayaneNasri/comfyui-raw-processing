from __future__ import annotations

import json
from abc import ABC, abstractmethod

import numpy as np
import torch

from .curve_engine import apply_lut_torch, build_lut
from .curve_presets import load_preset, save_preset
from .curve_spec import CurveSpec


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

    CATEGORY = "image/curves"
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


    # Shared logic
    def _parse_spec(self, curve_points: str, **kwargs) -> CurveSpec:
        try:
            data = json.loads(curve_points) if curve_points.strip() else []
        except (json.JSONDecodeError, AttributeError):
            data = []

        if not data:
            return self._default_spec(**kwargs)

        if isinstance(data, list):
            return CurveSpec(points=[tuple(p) for p in data])

        return CurveSpec.from_json(data)

    def execute(
        self,
        image: torch.Tensor,
        curve_points: str,
        lut_size: int,
        preset_name: str = "",
        save_preset_as: str = "",
        **kwargs,
    ) -> tuple:
        if preset_name.strip():
            spec = load_preset(preset_name.strip())
        else:
            spec = self._parse_spec(curve_points, **kwargs)

        if save_preset_as.strip():
            save_preset(spec, save_preset_as.strip())

        lut = build_lut(spec, lut_size)
        img = image.squeeze(0)
        result = self._apply_lut(img, lut)
        return (result.unsqueeze(0),)
