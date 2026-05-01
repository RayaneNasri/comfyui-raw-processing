from __future__ import annotations

import numpy as np
import torch

from .curve_engine import apply_lut_torch
from .curve_node_base import CurveNodeBase
from .curve_spec import CurveSpec


class GammaCurveNode(CurveNodeBase):
    """Gamma correction expressed as an editable curve.

    When no control points have been drawn, the default curve is computed
    from gamma_preset so the node behaves like a standard power-law gamma.
    Once the user draws custom points they take over.
    """

    CATEGORY = "image/curves"

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types["required"]["gamma_preset"] = (
            "FLOAT",
            {"default": 2.2, "min": 0.1, "max": 10.0, "step": 0.1},
        )
        return types

    def _default_spec(self, gamma_preset: float = 2.2, **kwargs) -> CurveSpec:
        return CurveSpec.gamma(gamma=gamma_preset)

    def _apply_lut(self, image: torch.Tensor, lut: np.ndarray) -> torch.Tensor:
        return apply_lut_torch(image, lut)


NODE_CLASS_MAPPINGS = {"GammaCurveNode": GammaCurveNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GammaCurveNode": "Gamma Curve"}
