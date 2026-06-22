from __future__ import annotations

import numpy as np
import torch
import numpy as np

from .curve_spec import CurveSpec


def _bezier_lut(xs: np.ndarray, ys: np.ndarray, lut_size: int) -> np.ndarray:
    """
    Génère une LUT en interpolant les points via des segments de Bézier cubiques,
    aligné exactement sur la logique Catmull-Rom du Front-End.
    """

    n = len(xs)
    if n < 2:
        raise ValueError("L'interpolation nécessite au moins 2 points.")

    # 1. Calcul des points de contrôle (Handles) pour chaque segment
    # On reproduit exactement le comportement JS avec les bornes Max/Min
    cp1x = np.zeros(n - 1)
    cp1y = np.zeros(n - 1)
    cp2x = np.zeros(n - 1)
    cp2y = np.zeros(n - 1)

    for i in range(n - 1):
        # Indices avec clamping comme en JS
        p0_idx = max(i - 1, 0)
        p1_idx = i
        p2_idx = i + 1
        p3_idx = min(i + 2, n - 1)

        # Calcul des tangentes (facteur 1/6)
        cp1x[i] = xs[p1_idx] + (xs[p2_idx] - xs[p0_idx]) / 6.0
        cp1y[i] = ys[p1_idx] + (ys[p2_idx] - ys[p0_idx]) / 6.0

        cp2x[i] = xs[p2_idx] - (xs[p3_idx] - xs[p1_idx]) / 6.0
        cp2y[i] = ys[p2_idx] - (ys[p3_idx] - ys[p1_idx]) / 6.0

    # 2. Génération de la Look-Up Table (LUT)
    # On crée un axe X régulier entre le premier et le dernier point
    lut_x = np.linspace(xs[0], xs[-1], lut_size, dtype=np.float32)
    lut_y = np.zeros(lut_size, dtype=np.float32)

    # Pour chaque valeur de lut_x, on cherche dans quel segment elle se trouve
    for j, x in enumerate(lut_x):
        if x <= xs[0]:
            lut_y[j] = ys[0]
            continue
        if x >= xs[-1]:
            lut_y[j] = ys[-1]
            continue

        # Trouver l'indice du segment
        # On cherche le premier point d'ancrage qui est juste avant 'x'
        i = np.searchsorted(xs, x) - 1
        i = max(0, min(i, n - 2))

        # Résoudre l'équation de Bézier cubique pour trouver 't' par rapport à X
        # P(t) = (1-t)³*P1 + 3(1-t)²t*CP1 + 3(1-t)t²*CP2 + t³*P2
        # Comme l'axe X est généralement monotone, une approximation ou une recherche
        # locale de t suffit. Ici, on utilise une interpolation linéaire locale pour t de départ,
        # suivie d'une approximation par dichotomie (très rapide).
        t_min, t_max = 0.0, 1.0
        p1_x, cp1_x, cp2_x, p2_x = xs[i], cp1x[i], cp2x[i], xs[i + 1]

        for _ in range(8):  # 8 itérations suffisent pour une précision de pixel (1/256)
            t_mid = (t_min + t_max) / 2.0
            mt = 1.0 - t_mid
            x_est = (
                (mt**3) * p1_x
                + 3 * (mt**2) * t_mid * cp1_x
                + 3 * mt * (t_mid**2) * cp2_x
                + (t_mid**3) * p2_x
            )

            if x_est < x:
                t_min = t_mid
            else:
                t_max = t_mid

        t = (t_min + t_max) / 2.0

        # Calculer le Y correspondant avec le 't' trouvé
        mt = 1.0 - t
        p1_y, cp1_y, cp2_y, p2_y = ys[i], cp1y[i], cp2y[i], ys[i + 1]
        y_val = (
            (mt**3) * p1_y
            + 3 * (mt**2) * t * cp1_y
            + 3 * mt * (t**2) * cp2_y
            + (t**3) * p2_y
        )

        lut_y[j] = np.clip(y_val, 0.0, 1.0)

    return lut_y


def build_lut(spec: CurveSpec, lut_size: int = 256) -> np.ndarray:
    """Build a float32 LUT of length lut_size from a CurveSpec."""

    if len(spec.points) < 2:
        raise ValueError("CurveSpec needs at least 2 control points")

    pts = sorted(spec.points, key=lambda p: p[0])
    dom_span = max(spec.domain_max - spec.domain_min, 1e-10)
    rng_span = max(spec.range_max - spec.range_min, 1e-10)

    xs = np.clip([(p[0] - spec.domain_min) / dom_span for p in pts], 0.0, 1.0)
    ys = np.clip([(p[1] - spec.range_min) / rng_span for p in pts], 0.0, 1.0)

    return _bezier_lut(
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
