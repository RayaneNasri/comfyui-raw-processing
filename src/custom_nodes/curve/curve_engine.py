from __future__ import annotations

import numpy as np
import torch

from .curve_spec import CurveSpec


def _polynomial_lut(xs: np.ndarray, ys: np.ndarray, lut_size: int) -> np.ndarray:
    """Polynomial interpolation via the barycentric Lagrange formula"""
    n = len(xs)
    if n < 2:
        raise ValueError("Polynomial interpolation needs at least 2 control points")
    if np.unique(xs).size != n:
        raise ValueError(
            "Control point x values must be unique for polynomial interpolation"
        )

    if n == 2:
        lut_x = np.linspace(xs[0], xs[-1], lut_size)
        width = max(xs[1] - xs[0], 1e-10)
        t = np.clip((lut_x - xs[0]) / width, 0.0, 1.0)
        return np.clip(ys[0] + (ys[1] - ys[0]) * t, 0.0, 1.0).astype(np.float32)

    # stable barycentric weights only the relative scale matters
    log_weights = np.empty(n, dtype=np.float64)
    signs = np.empty(n, dtype=np.float64)
    for index in range(n):
        deltas = xs[index] - np.delete(xs, index)
        signs[index] = np.prod(np.sign(deltas))
        log_weights[index] = -np.sum(np.log(np.abs(deltas)))

    log_weights -= np.max(log_weights)
    weights = signs * np.exp(log_weights)

    lut_x = np.linspace(xs[0], xs[-1], lut_size)
    diff = lut_x[:, None] - xs[None, :]
    exact = np.isclose(diff, 0.0, atol=1e-12, rtol=0.0)

    safe_diff = np.where(exact, 1.0, diff)
    numerator = np.sum((weights * ys) / safe_diff, axis=1)
    denominator = np.sum(weights / safe_diff, axis=1)
    lut_y = numerator / denominator

    if np.any(exact):
        exact_rows = np.where(np.any(exact, axis=1))[0]
        exact_cols = np.argmax(exact[exact_rows], axis=1)
        lut_y[exact_rows] = ys[exact_cols]

    return np.clip(lut_y, 0.0, 1.0).astype(np.float32)


def build_lut(spec: CurveSpec, lut_size: int = 256) -> np.ndarray:
    """Build a float32 LUT of length lut_size from a CurveSpec."""
    if len(spec.points) < 2:
        raise ValueError("CurveSpec needs at least 2 control points")

    pts = sorted(spec.points, key=lambda p: p[0])
    dom_span = max(spec.domain_max - spec.domain_min, 1e-10)
    rng_span = max(spec.range_max - spec.range_min, 1e-10)

    xs = np.clip([(p[0] - spec.domain_min) / dom_span for p in pts], 0.0, 1.0)
    ys = np.clip([(p[1] - spec.range_min) / rng_span for p in pts], 0.0, 1.0)

    return _polynomial_lut(
        np.array(xs, dtype=np.float64),
        np.array(ys, dtype=np.float64),
        lut_size,
    )


def apply_lut_numpy(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a float32 LUT to a float32 image array (H, W, C), values in [0,1]."""
    n = len(lut) - 1
    idx = np.clip(image * n, 0.0, n)
    lo = idx.astype(np.int32)
    hi = np.clip(lo + 1, 0, n)
    frac = (idx - lo).astype(np.float32)
    return (lut[lo] * (1.0 - frac) + lut[hi] * frac).astype(np.float32)


def apply_lut_torch(image: torch.Tensor, lut: np.ndarray) -> torch.Tensor:
    """Apply a float32 LUT to a float32 torch tensor (H, W, C), values in [0,1]."""
    lut_t = torch.from_numpy(lut).float().to(image.device)
    n = len(lut_t) - 1
    idx = (image * n).clamp(0.0, float(n))
    lo = idx.long().clamp(0, n)
    hi = (lo + 1).clamp(0, n)
    frac = idx - lo.float()
    return (lut_t[lo] * (1.0 - frac) + lut_t[hi] * frac).clamp(0.0, 1.0)
