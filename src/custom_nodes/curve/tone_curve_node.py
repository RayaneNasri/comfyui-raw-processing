from __future__ import annotations

import numpy as np
import torch

from .curve_engine import apply_lut_torch
from .curve_node_base import CurveNodeBase
from .curve_spec import CurveSpec


class ToneCurveNode(CurveNodeBase):
    """Tone curve applied uniformly to all channels (master RGB curve).

    The same LUT is broadcast across R, G and B, making it suitable for
    luminance-style adjustments such as lift/gamma/gain or S-curve contrast.
    """

    CATEGORY = "image/curves"

    def _default_spec(self, **kwargs) -> CurveSpec:
        return CurveSpec.identity(n_points=5)

    def _apply_lut(self, image: torch.Tensor, lut: np.ndarray) -> torch.Tensor:
        return apply_lut_torch(image, lut)


NODE_CLASS_MAPPINGS = {"ToneCurveNode": ToneCurveNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ToneCurveNode": "Tone Curve"}
