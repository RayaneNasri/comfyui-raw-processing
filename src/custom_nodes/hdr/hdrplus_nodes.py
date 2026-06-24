import torch
import logging

# Adjust import path based on your exact extension directory structure
from algorithms.hdr._bridge import (
    process_hdrplus_burst,
    apply_global_tone_mapping,
    apply_local_tone_mapping,
)

logger = logging.getLogger(__name__)


class HDRPlusFusionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_imgs": ("IMAGE",),
                "cfa_patterns": ("PATTERN",),
                "black_levels": ("BLACK_LEVEL",),
                "white_levels": ("WHITE_LEVEL",),
                "wb_gains": ("WB_GAINS",),
                "exif_tags": ("EXIF_TAGS",),
                "reference_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "temporal_factor": (
                    "FLOAT",
                    {
                        "default": 75.0,
                        "min": 0.0,
                        "max": 200.0,
                        "step": 1.0,
                        "tooltip": "Ghosting vs Denoising tradeoff",
                    },
                ),
                "spatial_factor": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Wiener filter spatial noise reduction",
                    },
                ),
            },
        }

    CATEGORY = "image/HDR"
    # Assurez-vous que RETURN_TYPES correspond aux noms de types de votre écosystème
    RETURN_TYPES = (
        "IMAGE",
        "PATTERN",
        "BLACK_LEVEL",
        "WHITE_LEVEL",
        "WB_GAIN",
        "EXIF_TAGS",
    )
    RETURN_NAMES = (
        "merged_raw",
        "cfa_pattern",
        "black_level",
        "white_level",
        "wb_gains",
        "exif_tags",
    )
    FUNCTION = "execute"

    def execute(
        self,
        raw_imgs,
        cfa_patterns,
        black_levels,
        white_levels,
        wb_gains,
        exif_tags,
        reference_index,
        temporal_factor,
        spatial_factor,
    ):
        B = raw_imgs.shape[0]
        if reference_index >= B:
            logger.warning(
                f"Reference index {reference_index} out of bounds. Defaulting to 0."
            )
            reference_index = 0

        # Run the Numba CPU HDR+ algorithm
        merged_raw = process_hdrplus_burst(
            raw_imgs=raw_imgs,
            black_levels=black_levels,
            white_levels=white_levels,
            exif_tags=exif_tags,
            ref_idx=reference_index,
            temporal_factor=temporal_factor,
            spatial_factor=spatial_factor,
        )

        # Pass along the metadata of the reference image
        out_pattern = cfa_patterns[reference_index].unsqueeze(0)
        out_bl = black_levels[reference_index].unsqueeze(0)
        out_wl = white_levels[reference_index].unsqueeze(0)

        # <-- CORRECTION ICI : On extrait le wb_gains de l'image de référence au lieu de renvoyer None
        out_wb = wb_gains[reference_index].unsqueeze(0)

        return (
            merged_raw,
            out_pattern,
            out_bl,
            out_wl,
            out_wb,
            [exif_tags[reference_index]],
        )


class HDRPlusToneMappingNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                ),  # Expects normalized RGB (0-1) after demosaicing and WB
                "apply_local_tm": (
                    "BOOLEAN",
                    {"default": True, "label": "Local Tone Mapping (Mertens Fusion)"},
                ),
                "ltm_gain": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "tooltip": "0.0 means Auto Gain",
                    },
                ),
                "apply_global_tm": (
                    "BOOLEAN",
                    {"default": True, "label": "Global Tone Mapping (S-Curve)"},
                ),
                "gtm_contrast": (
                    "FLOAT",
                    {"default": 0.075, "min": 0.0, "max": 0.5, "step": 0.005},
                ),
            },
        }

    CATEGORY = "image/HDR"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"

    def execute(self, image, apply_local_tm, ltm_gain, apply_global_tm, gtm_contrast):
        B = image.shape[0]
        out_images = []

        for i in range(B):
            img_slice = image[i : i + 1]  # Keep batch dimension [1, H, W, 3]

            if apply_local_tm:
                img_slice = apply_local_tone_mapping(img_slice, ltm_gain)

            if apply_global_tm:
                img_slice = apply_global_tone_mapping(img_slice, gtm_contrast)

            out_images.append(img_slice)

        return (torch.cat(out_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "HDRPlusFusionNode": HDRPlusFusionNode,
    "HDRPlusToneMappingNode": HDRPlusToneMappingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HDRPlusFusionNode": "HDR+ Burst Fusion",
    "HDRPlusToneMappingNode": "HDR+ Tone Mapping",
}
