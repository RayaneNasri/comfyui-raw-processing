import logging
from typing import Any, Callable, Optional
import torch

# Assuming these are imported from your modernized translation modules
from .alignment import select_reference, align_burst
from .merging import merge_burst
from .finishing import finish

logger = logging.getLogger(__name__)


def hdrplus_pipeline(
    burst_images: list[torch.Tensor],
    tags: dict[str, Any],
    black_level: list[int] | tuple[int, ...] | torch.Tensor,
    white_level: int | float,
    params: dict[str, Any],
    options: dict[str, Any],
    demosaic_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
) -> torch.Tensor:
    """
    This function encompasses the whole HDR+ pipeline, fully modernized for PyTorch.
    File I/O and metadata extraction should be handled prior to calling this function.
    
    Args:
        burst_images: A list of 2D Bayer tensors or a single 3D tensor (B, H, W).
        tags: Dictionary of EXIF metadata (needed for noise curve estimation).
        black_level: Per-channel black level of the camera sensor.
        white_level: Maximum white level of the camera sensor.
        params: Dictionary containing algorithm hyperparameters.
        options: Dictionary containing processing options.
        demosaic_fn: A custom function/module that converts a 2D Bayer tensor to a 3D RGB tensor.
        
    Returns:
        The final processed RGB image tensor [0.0, 1.0], or the Bayer tensor if no demosaic_fn is provided.
    """
    
    # 1. Reference Image Selection
    # (Assuming select_reference was modernized to accept a list of tensors)
    ref_idx = select_reference(burst_images, options)
    logger.debug(f"Selected reference image index: {ref_idx}")

    # 2. Burst Alignment / Registration
    aligned_tiles, padding = align_burst(
        burst_images, 
        ref_idx, 
        params.get('alignment', {}), 
        options
    )
    logger.debug("Burst alignment complete.")

    # 3. Burst Merging
    merged_bayer = merge_burst(
        burst_images, 
        ref_idx, 
        aligned_tiles, 
        padding, 
        tags, 
        black_level, 
        white_level, 
        params.get('merging', {}), 
        options
    )
    logger.debug("Burst merging complete.")

    # 4. Demosaicing (Bridging the gap between Bayer and RGB)
    if demosaic_fn is not None:
        merged_rgb = demosaic_fn(merged_bayer)
    else:
        logger.warning("No demosaicing function provided. Pipeline stopping early and returning raw merged Bayer tensor.")
        return merged_bayer

    # 5. Finishing (Tone Mapping, Contrast, Gamma, Sharpening)
    final_image = finish(
        merged_rgb, 
        params.get('finishing', {}), 
        options
    )
    logger.debug("Finishing steps complete.")

    return final_image