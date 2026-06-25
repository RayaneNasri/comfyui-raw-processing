"""
ComfyUI nodes for the HDR+ pipeline.

Split into two nodes to support an external demosaicing node in between:
  1. HDRPlusAlignMerge  — burst alignment + Wiener merging → merged Bayer tensor
  2. HDRPlusFinish      — tone mapping + gamma + sharpening → final RGB image

Expected wiring:
  [BatchReadRawSensorNode] ──► [HDRPlusAlignMerge] ──► [your demosaic node] ──► [HDRPlusFinish]

Loader output types and what this node receives:
  raw_imgs     → IMAGES        : Tensor (B, H, W, 1)  float32  raw Bayer values
  black_levels → BLACK_LEVEL   : Tensor (B, 4)         float32  per-channel black level
  white_levels → WHITE_LEVEL   : Tensor (B, 1)         float32  sensor white level
  exif_tags    → EXIF_TAGS     : list[dict]  keys: 'noise_profile' (list[float] | None),
                                                        'iso'           (float | None)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Lazy imports so ComfyUI can register the node list without the full hdrplus
# package being importable yet (errors surface at execution time instead).
# ---------------------------------------------------------------------------
from algorithms.hdr.alignment import select_reference, align_burst
from algorithms.hdr.merging import merge_burst, validate_consistent_cfa_pattern
from algorithms.hdr.finishing import finish

logger = logging.getLogger(__name__)


# ===========================================================================
# EXIF translation
# ===========================================================================


def _translate_exif_tags(loader_tags: dict) -> dict:
    """
    Translate the clean dict produced by BatchReadRawSensorNode._extract_exif_tags()
    into the format expected by merging.get_noise_params().

    Loader dict keys:
        'noise_profile' : list[float] | None   — parsed DNG NoiseProfile values
        'iso'           : float | None          — ISO speed rating

    get_noise_params() looks for:
        'Image Tag 0xC761'          — with a .values attribute (list of 2-tuples or floats)
        'Image ISOSpeedRatings'     — with a .values[0] attribute
    """

    class _FakeTag:
        """Minimal stand-in for an exifread IfdTag so get_noise_params() can call .values."""

        def __init__(self, values):
            self.values = values

    translated: dict[str, Any] = {}

    noise_profile = loader_tags.get("noise_profile")
    if noise_profile is not None:
        # get_noise_params expects values as a list of 2-element sequences (numerator, denominator)
        # when len == 2 it reads float(values[0]) and float(values[1]) directly;
        # when len == 6 it checks equality of alternating elements.
        # Our loader already stored them as plain floats, so wrap each in a 1-tuple so
        # float(v[0]) still works.
        translated["Image Tag 0xC761"] = _FakeTag([[v] for v in noise_profile])

    iso = loader_tags.get("iso")
    if iso is not None:
        translated["Image ISOSpeedRatings"] = _FakeTag([int(iso)])

    return translated


# ===========================================================================
# Param / options builders
# ===========================================================================


def _build_alignment_params(
    factors_str: str,
    tile_sizes_str: str,
    search_radia_str: str,
    distances_str: str,
    subpixels_str: str,
    bayer_mode: bool,
) -> dict[str, Any]:
    """Convert user-facing comma-separated string inputs into the nested dict
    that align_burst / align_hdrplus expect under params["tuning"].
    Lists are defined fine-to-coarse (index 0 = finest level)."""

    def _ints(s: str) -> list[int]:
        return [int(x.strip()) for x in s.split(",")]

    def _strs(s: str) -> list[str]:
        return [x.strip() for x in s.split(",")]

    def _bools(s: str) -> list[bool]:
        m = {"true": True, "false": False, "1": True, "0": False}
        return [m[x.strip().lower()] for x in s.split(",")]

    return {
        "mode": "bayer" if bayer_mode else "gray",
        "tuning": {
            "factors": _ints(factors_str),
            "tileSizes": _ints(tile_sizes_str),
            "searchRadia": _ints(search_radia_str),
            "distances": _strs(distances_str),
            "subpixels": _bools(subpixels_str),
        },
    }


def _build_merging_params(
    patch_size: int,
    method: str,
    noise_curve: str,
    lambda_s: float,
    lambda_r: float,
) -> dict[str, Any]:
    return {
        "patchSize": patch_size,
        "method": method,
        # When 'manual', pass a tuple so get_noise_params() takes the isinstance(tuple) branch
        "noiseCurve": (lambda_s, lambda_r) if noise_curve == "manual" else noise_curve,
    }


def _build_finishing_params(
    ltm_gain: float,
    gtm_contrast: float,
    sharpen_amounts: str,
    sharpen_sigmas: str,
    sharpen_thresholds: str,
) -> dict[str, Any]:
    def _floats(s: str) -> list[float]:
        return [float(x.strip()) for x in s.split(",")]

    return {
        "ltmGain": ltm_gain,
        "gtmContrast": gtm_contrast,
        "tuning": {
            "sharpenAmount": _floats(sharpen_amounts),
            "sharpenSigma": _floats(sharpen_sigmas),
            "sharpenThreshold": _floats(sharpen_thresholds),
        },
    }


def _build_options(
    reference_index: int,
    temporal_factor: float,
    spatial_factor: float,
    ltm_gain: float,
    gtm_contrast: float,
) -> dict[str, Any]:
    return {
        "mode": "full",
        "referenceIndex": reference_index,
        "temporalFactor": temporal_factor,
        "spatialFactor": spatial_factor,
        "ltmGain": ltm_gain if ltm_gain > 0 else -1,
        "gtmContrast": gtm_contrast,
    }


# ===========================================================================
# Burst tensor normalisation
# ===========================================================================


def _unpack_burst(burst_images) -> list[torch.Tensor]:
    """
    Accept whatever shape the loader produces and return a list of 2-D (H, W)
    float32 tensors, one per frame.

    Handled shapes
    --------------
    (B, H, W, 1)  ← BatchReadRawSensorNode output  [primary path]
    (B, H, W)     ← stacked Bayer without channel dim
    (H, W, 1)     ← single Bayer image with channel dim
    (H, W)        ← single Bayer image
    (B, H, W, C≥3) ← RGB/RGBA — converted to luminance with a warning
    list[Tensor]  ← already unpacked; each element normalised to 2-D
    """
    if isinstance(burst_images, torch.Tensor):
        t = burst_images

        # Promote single images to a batch
        if t.ndim == 2:  # (H, W)
            t = t.unsqueeze(0)  # → (1, H, W)
        elif t.ndim == 3 and t.shape[-1] in (1, 3, 4):  # (H, W, C) single image
            t = t.unsqueeze(0)  # → (1, H, W, C)

        if t.ndim == 3:  # (B, H, W)
            return [t[i] for i in range(t.shape[0])]

        if t.ndim == 4:
            C = t.shape[-1]
            if C == 1:  # (B, H, W, 1) ← our case
                return [t[i, :, :, 0] for i in range(t.shape[0])]
            else:  # (B, H, W, 3/4) RGB
                logger.warning(
                    "HDRPlusAlignMerge: received %d-channel images — converting to "
                    "luminance. Supply raw Bayer images for best results.",
                    C,
                )
                w = t.new_tensor([0.299, 0.587, 0.114])
                luma = (t[..., :3] * w).sum(dim=-1)  # (B, H, W)
                return [luma[i] for i in range(luma.shape[0])]

        raise ValueError(
            f"HDRPlusAlignMerge: unsupported burst tensor shape {burst_images.shape}."
        )

    # list[Tensor] path
    images = []
    for img in burst_images:
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        elif img.ndim == 3:
            w = img.new_tensor([0.299, 0.587, 0.114])
            img = (img[..., :3] * w).sum(dim=-1)
        images.append(img)
    return images


# ===========================================================================
# Node 1 – HDRPlus Align & Merge
# ===========================================================================


class HDRPlusAlignMerge:
    """
    HDR+ Burst Alignment + Merging.

    Wire directly to BatchReadRawSensorNode:
        raw_imgs     → burst_images
        black_levels → black_levels
        white_levels → white_levels
        exif_tags    → exif_tags

    Outputs the merged raw Bayer tensor (H, W) ready for your demosaic node.
    """

    CATEGORY = "HDR+"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "PATTERN")
    RETURN_NAMES = ("merged_bayer", "cfa_pattern")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ── From BatchReadRawSensorNode ───────────────────────────
                "burst_images": ("IMAGES",),  # (B, H, W, 1) float32
                "cfa_patterns": ("PATTERN",),  # (B, 2, 2) — one CFA pattern per frame
                "black_levels": ("BLACK_LEVEL",),  # (B, 4)        float32
                "white_levels": ("WHITE_LEVEL",),  # (B, 1)        float32
                "exif_tags": ("EXIF_TAGS",),  # list[dict]
                # ── Reference selection ───────────────────────────────────
                "reference_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 31,
                        "step": 1,
                        "tooltip": "Index of the reference frame in the burst (0 = first)",
                    },
                ),
                # ── Alignment pyramid (fine-to-coarse, 4 values each) ─────
                "bayer_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Keep True for raw Bayer input (recommended).",
                    },
                ),
                "align_factors": (
                    "STRING",
                    {
                        "default": "1,2,4,4",
                        "tooltip": "Pyramid downsampling factors, fine-to-coarse",
                    },
                ),
                "align_tile_sizes": (
                    "STRING",
                    {
                        "default": "16,16,16,8",
                        "tooltip": "Tile sizes at each pyramid level, fine-to-coarse",
                    },
                ),
                "align_search_radia": (
                    "STRING",
                    {
                        "default": "1,4,4,4",
                        "tooltip": "Search radii at each pyramid level, fine-to-coarse",
                    },
                ),
                "align_distances": (
                    "STRING",
                    {
                        "default": "L1,L2,L2,L2",
                        "tooltip": "Distance metric per level: L1 or L2, fine-to-coarse",
                    },
                ),
                "align_subpixels": (
                    "STRING",
                    {
                        "default": "False,True,True,True",
                        "tooltip": "Sub-pixel alignment per level, fine-to-coarse",
                    },
                ),
                # ── Merging ───────────────────────────────────────────────
                "merge_patch_size": (
                    "INT",
                    {
                        "default": 16,
                        "min": 4,
                        "max": 64,
                        "step": 4,
                        "tooltip": "Tile size used during merging",
                    },
                ),
                "merge_method": (
                    ["DFTWiener", "keepAlternate", "pairedAverage"],
                    {
                        "default": "DFTWiener",
                    },
                ),
                "noise_curve": (
                    ["exifNoiseProfile", "exifISO", "manual"],
                    {
                        "default": "exifNoiseProfile",
                        "tooltip": (
                            "exifNoiseProfile: uses DNG NoiseProfile tag (best). "
                            "exifISO: estimates from ISO. "
                            "manual: uses the lambda_s / lambda_r values below."
                        ),
                    },
                ),
                "temporal_factor": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 64.0,
                        "step": 0.5,
                        "tooltip": "Wiener temporal denoising strength (0 = off)",
                    },
                ),
                "spatial_factor": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 64.0,
                        "step": 0.5,
                        "tooltip": "Wiener spatial denoising strength (0 = off)",
                    },
                ),
            },
            "optional": {
                # Only used when noise_curve = 'manual'
                "lambda_s": (
                    "FLOAT",
                    {
                        "default": 3.24e-4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 1e-6,
                        "tooltip": "Shot-noise coefficient (manual noise curve only)",
                    },
                ),
                "lambda_r": (
                    "FLOAT",
                    {
                        "default": 4.3e-6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 1e-8,
                        "tooltip": "Read-noise coefficient (manual noise curve only)",
                    },
                ),
            },
        }

    def run(
        self,
        burst_images,
        cfa_patterns: torch.Tensor,
        black_levels: torch.Tensor,
        white_levels: torch.Tensor,
        exif_tags: list[dict],
        reference_index: int,
        bayer_mode: bool,
        align_factors: str,
        align_tile_sizes: str,
        align_search_radia: str,
        align_distances: str,
        align_subpixels: str,
        merge_patch_size: int,
        merge_method: str,
        noise_curve: str,
        temporal_factor: float,
        spatial_factor: float,
        lambda_s: float = 3.24e-4,
        lambda_r: float = 4.3e-6,
    ):
        if align_burst is None:
            raise RuntimeError(
                "HDR+ algorithm package not found. "
                "Ensure algorithms/hdr/ is on the Python path."
            )

        # ── Validate the burst agrees on a single CFA layout ──────────────
        # Alignment and merging assume one consistent sensor layout across the
        # whole burst; this is true for any normal burst from one camera, so a
        # mismatch usually signals a loader bug or a mixed-source burst.
        validate_consistent_cfa_pattern(cfa_patterns)

        # ── Unpack burst into list[Tensor(H, W)] ─────────────────────────
        images = _unpack_burst(burst_images)
        n = len(images)
        if n < 2:
            logger.warning(
                "HDRPlusAlignMerge: only %d image in burst — merging is a no-op.", n
            )

        # ── Sensor levels: pick reference frame's values ──────────────────
        # black_levels : (B, 4) — use reference frame row
        ref_idx_clamped = min(reference_index, n - 1)
        black_level_tensor = black_levels[ref_idx_clamped]  # (4,) float32
        white_level_scalar = white_levels[ref_idx_clamped, 0].item()  # scalar float

        # ── EXIF tags: translate loader dict → get_noise_params() format ──
        ref_tags_raw = exif_tags[ref_idx_clamped] if exif_tags else {}
        translated_tags = _translate_exif_tags(ref_tags_raw)

        if not translated_tags and noise_curve != "manual":
            logger.warning(
                "HDRPlusAlignMerge: no noise profile or ISO found in EXIF tags for "
                "frame %d. Falling back to ISO-100 baseline.",
                ref_idx_clamped,
            )

        # ── Build param / options dicts ───────────────────────────────────
        alignment_params = _build_alignment_params(
            align_factors,
            align_tile_sizes,
            align_search_radia,
            align_distances,
            align_subpixels,
            bayer_mode,
        )
        merging_params = _build_merging_params(
            merge_patch_size,
            merge_method,
            noise_curve,
            lambda_s,
            lambda_r,
        )
        options = _build_options(
            reference_index=reference_index,
            temporal_factor=temporal_factor,
            spatial_factor=spatial_factor,
            ltm_gain=-1,
            gtm_contrast=0.0,
        )

        # ── Alignment ─────────────────────────────────────────────────────
        ref_idx = select_reference(images, options)
        logger.debug("HDRPlusAlignMerge: reference index = %d", ref_idx)

        aligned_tiles, padding = align_burst(images, ref_idx, alignment_params, options)
        logger.debug("HDRPlusAlignMerge: alignment done, padding=%s", padding)

        # The merged Bayer output inherits the reference frame's CFA layout —
        # alignment/merging only shift and denoise content, they never change
        # the underlying mosaic pattern.
        ref_cfa_pattern = cfa_patterns[ref_idx]

        # ── Merging ───────────────────────────────────────────────────────
        merged_bayer = merge_burst(
            images,
            ref_idx,
            aligned_tiles,
            padding,
            translated_tags,
            black_level_tensor,
            white_level_scalar,
            merging_params,
            options,
            ref_cfa_pattern,
        )
        logger.debug(
            "HDRPlusAlignMerge: merged shape=%s  dtype=%s  range=[%.1f, %.1f]",
            merged_bayer.shape,
            merged_bayer.dtype,
            merged_bayer.min().item(),
            merged_bayer.max().item(),
        )

        return (merged_bayer, ref_cfa_pattern)


# ===========================================================================
# Node 2 – HDRPlus Finish
# ===========================================================================


class HDRPlusFinish:
    """
    HDR+ Finishing: Local Tone Mapping → Global Contrast → sRGB Gamma → Sharpening.

    Input : demosaiced RGB tensor from your demosaic node  (H, W, 3) or (1, H, W, 3)
    Output: display-ready IMAGE in ComfyUI standard format (1, H, W, 3) float32 [0, 1]
    """

    CATEGORY = "HDR+"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_image": ("IMAGE",),
                "ltm_gain": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 64.0,
                        "step": 0.5,
                        "tooltip": "-1 = auto-estimate, 0 = disable LTM, >0 = fixed gain",
                    },
                ),
                "gtm_contrast": (
                    "FLOAT",
                    {
                        "default": 0.075,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.005,
                        "tooltip": "S-curve contrast strength (0 = off)",
                    },
                ),
                "sharpen_amounts": (
                    "STRING",
                    {
                        "default": "1.0,0.5,0.5",
                        "tooltip": "Unsharp mask amounts for the 3 Gaussian scales",
                    },
                ),
                "sharpen_sigmas": (
                    "STRING",
                    {
                        "default": "1.0,2.0,4.0",
                        "tooltip": "Gaussian sigmas for the 3 sharpening scales",
                    },
                ),
                "sharpen_thresholds": (
                    "STRING",
                    {
                        "default": "0.02,0.04,0.06",
                        "tooltip": "Edge thresholds: only sharpen above these values",
                    },
                ),
            },
        }

    def run(
        self,
        rgb_image: torch.Tensor,
        ltm_gain: float,
        gtm_contrast: float,
        sharpen_amounts: str,
        sharpen_sigmas: str,
        sharpen_thresholds: str,
    ):
        if finish is None:
            raise RuntimeError(
                "HDR+ algorithm package not found. "
                "Ensure algorithms/hdr/ is on the Python path."
            )

        # Handle ComfyUI batch dim (1, H, W, 3) → (H, W, 3)
        squeeze = False
        if rgb_image.ndim == 4:
            if rgb_image.shape[0] == 1:
                rgb_image = rgb_image.squeeze(0)
                squeeze = True
            else:
                raise ValueError(
                    f"HDRPlusFinish processes one image at a time, got batch {rgb_image.shape[0]}."
                )

        if not rgb_image.is_floating_point():
            rgb_image = rgb_image.to(torch.float32) / 65535.0

        finishing_params = _build_finishing_params(
            ltm_gain,
            gtm_contrast,
            sharpen_amounts,
            sharpen_sigmas,
            sharpen_thresholds,
        )
        options = _build_options(
            reference_index=0,
            temporal_factor=8.0,
            spatial_factor=0.0,
            ltm_gain=ltm_gain,
            gtm_contrast=gtm_contrast,
        )

        result = finish(rgb_image, finishing_params, options)
        logger.debug(
            "HDRPlusFinish: output shape=%s  range=[%.3f, %.3f]",
            result.shape,
            result.min().item(),
            result.max().item(),
        )

        # ComfyUI expects (1, H, W, 3)
        if squeeze:
            result = result.unsqueeze(0)

        return (result,)


# ===========================================================================
# Node registry
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "HDRPlusAlignMerge": HDRPlusAlignMerge,
    "HDRPlusFinish": HDRPlusFinish,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HDRPlusAlignMerge": "HDR+ Align & Merge",
    "HDRPlusFinish": "HDR+ Finish",
}
