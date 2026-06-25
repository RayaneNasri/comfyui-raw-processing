import logging
from pathlib import Path
from typing import Any
import shutil
import cv2
import torch
import torch.nn.functional as F
import rawpy
import numpy as np

from .utils import (
    downsample,
    get_tiles,
    get_aligned_tiles,
    compute_tiles_distance_l1,
    compute_distance,
    sub_pixel_minimum,
)
# from ..visualization.vis import addMotionField

logger = logging.getLogger(__name__)


def select_reference(image_list: list[torch.Tensor], options: dict[str, Any]) -> int:
    """Selects the reference image inside a burst (Section 3.1 of the article)."""

    mode = options.get("mode")
    ref_idx_opt = options.get("referenceIndex", -1)

    if mode in ("full", "align"):
        # If user explicitly provided a valid index, use it.
        if isinstance(ref_idx_opt, int) and 0 <= ref_idx_opt < len(image_list):
            ref_idx = ref_idx_opt
        else:
            # TODO: Implement better reference selection logic here.
            if ref_idx_opt != -1:
                logger.warning(
                    "Incorrect reference image index provided. Defaulting to the first image."
                )
            ref_idx = 0
    else:
        # Defaulting to the first image for other modes
        ref_idx = 0

    return ref_idx


def align_burst(
    images: list[torch.Tensor] | torch.Tensor,
    ref_idx: int,
    params: dict[str, Any],
    options: dict[str, Any],
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Estimate motion between the reference and other images of the burst, and return a set of aligned tiles."""

    h, w = images[ref_idx].shape[-2:]

    if params.get("mode") == "bayer":
        tile_size = 2 * params["tuning"]["tileSizes"][0]
    else:
        tile_size = params["tuning"]["tileSizes"][0]

    # Calculate padding to ensure tiles cleanly fit the image
    padding_patches_height = (tile_size - h % tile_size) * (h % tile_size != 0)
    padding_patches_width = (tile_size - w % tile_size) * (w % tile_size != 0)

    # Additional overlap padding
    padding_overlap_height = padding_overlap_width = tile_size // 2

    padding_top = padding_overlap_height
    padding_bottom = padding_overlap_height + padding_patches_height
    padding_left = padding_overlap_width
    padding_right = padding_overlap_width + padding_patches_width

    pad_tuple = (padding_left, padding_right, padding_top, padding_bottom)

    # Pad all images
    images_padded = []
    for im in images:
        orig_ndim = im.ndim
        if orig_ndim == 2:
            im = im.unsqueeze(0).unsqueeze(0)
        elif orig_ndim == 3:
            im = im.unsqueeze(0)

        needs_cast = not im.is_floating_point()
        if needs_cast:
            padded = F.pad(im.to(torch.float32), pad_tuple, mode="reflect").to(im.dtype)
        else:
            padded = F.pad(im, pad_tuple, mode="reflect")

        if orig_ndim == 2:
            padded = padded.squeeze(0).squeeze(0)
        elif orig_ndim == 3:
            padded = padded.squeeze(0)

        images_padded.append(padded)

    im_ref = images_padded[ref_idx]
    alternate_images = [img for i, img in enumerate(images_padded) if i != ref_idx]

    # Call the alignment function
    motion_vectors, aligned_tiles = align_hdrplus(
        im_ref, alternate_images, params, options
    )
    padding = (padding_top, padding_bottom, padding_left, padding_right)

    return aligned_tiles, padding


def align_hdrplus(
    reference_image: torch.Tensor,
    alternate_images: list[torch.Tensor],
    params: dict[str, Any],
    options: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implements the coarse-to-fine alignment on 4-level gaussian pyramids
    as defined in Algorithm 1 of Section 3 of the IPOL article.
    """
    tuning = params["tuning"]
    factors = tuning["factors"]
    tile_sizes = tuning["tileSizes"]
    distances = tuning["distances"]
    search_radia = tuning["searchRadia"]
    subpixels = tuning["subpixels"]

    # factors, tile_sizes, distances, search_radia and subpixels are described fine-to-coarse
    upsampling_factors = factors[1:] + [1]
    previous_tile_sizes = tile_sizes[1:] + [None]

    is_bayer = params.get("mode") == "bayer"

    if is_bayer:
        # If dealing with raw images, 2x2 bayer pixels block are averaged
        # (motion can then only be multiples of 2 pixels in original image size)
        im_ref = downsample(reference_image, kernel="bayer")
        tile_size = 2 * tile_sizes[0]
    else:
        im_ref = reference_image
        tile_size = tile_sizes[0]

    # Extract reference tiles (overlap by half in each spatial dimension)
    ref_tiles = get_tiles(reference_image, tile_size, tile_size // 2)

    # Collect tensors in lists, then stack at the end (PyTorch best practice)
    aligned_tiles_list = [ref_tiles]
    motion_vectors_list = []

    # construct 4-level coarse-to fine pyramid of the reference
    reference_pyramid = hdrplus_pyramid(im_ref, factors)

    # Align each alternate image to the reference image
    for i, alternate_image in enumerate(alternate_images):
        if is_bayer:
            # downsample bayer to grayscale
            im_alt = downsample(alternate_image, kernel="bayer")
        else:
            im_alt = alternate_image

        # 4-level coarse-to fine pyramid of alternate image
        alternate_pyramid = hdrplus_pyramid(im_alt, factors)

        # successively align from coarsest to finest level of the pyramid
        alignments = None
        for lv in range(len(reference_pyramid)):
            # Index logic from original: go backwards
            idx = -lv - 1
            alignments = align_on_a_level(
                reference_pyramid[lv],
                alternate_pyramid[lv],
                upsampling_factors[idx],
                tile_sizes[idx],
                previous_tile_sizes[idx],
                search_radia[idx],
                distances[idx],
                subpixels[idx],
                alignments,
            )

        assert alignments is not None, (
            "Alignments must be computed by the pyramid loop."
        )
        # use alignment vectors to get the tiles of alternateImage matching with those of the reference image
        if is_bayer:
            # estimated motion must be upsampled by a factor of 2 to go back to original image size
            alignments = upsample_alignments(
                reference_pyramid_level=torch.empty(0),
                alternate_pyramid_level=torch.empty(0),
                previous_alignments=alignments,
                upsampling_factor=2,
                tile_size=tile_size,
                previous_tile_size=tile_sizes[0],
                method=None,
            )

        aligned_alt_tiles = get_aligned_tiles(alternate_image, tile_size, alignments)

        motion_vectors_list.append(alignments)
        aligned_tiles_list.append(aligned_alt_tiles)

    # Stack lists into final batched Tensors
    final_aligned_tiles = torch.stack(aligned_tiles_list, dim=0)

    # Handle the edge case where no alternate images were passed
    if motion_vectors_list:
        final_motion_vectors = torch.stack(motion_vectors_list, dim=0)
    else:
        h, w = ref_tiles.shape[0], ref_tiles.shape[1]
        final_motion_vectors = torch.empty((0, h, w, 2), device=reference_image.device)

    return final_motion_vectors, final_aligned_tiles


def hdrplus_pyramid(
    image: torch.Tensor, factors: list[int] | None = None, kernel: str = "gaussian"
) -> list[torch.Tensor]:
    """
    Construct 4-level coarse-to-fine gaussian pyramid as described
    in the HDR+ paper and its supplement (Section 3.2 of the IPOL article).

    Args:
        image: input tensor (expected to be a grayscale image downsampled from a Bayer raw image)
        factors: downsampling factors (fine-to-coarse). Defaults to [1, 2, 4, 4].
        kernel: convolution kernel to apply before downsampling. Defaults to 'gaussian'.

    Returns:
        List of tensors representing the pyramid levels ordered coarse-to-fine.
    """
    if factors is None:
        factors = [1, 2, 4, 4]

    # Start with the finest level computed from the input
    pyramid_levels: list[torch.Tensor] = [
        downsample(image, kernel=kernel, factor=factors[0])
    ]

    # Subsequent pyramid levels are successively created
    # with convolution by a kernel followed by downsampling
    for factor in factors[1:]:
        pyramid_levels.append(
            downsample(pyramid_levels[-1], kernel=kernel, factor=factor)
        )

    # Reverse the pyramid in-place to get it coarse-to-fine
    pyramid_levels.reverse()

    return pyramid_levels


def upsample_alignments(
    reference_pyramid_level: torch.Tensor,
    alternate_pyramid_level: torch.Tensor,
    previous_alignments: torch.Tensor,
    upsampling_factor: int,
    tile_size: int,
    previous_tile_size: int,
    method: str | None = "hdrplus",
) -> torch.Tensor:
    """Upsample alignments to adapt them to the next pyramid level (Section 3.2 of the IPOL article)."""

    device = previous_alignments.device
    dtype = previous_alignments.dtype

    # As resolution is multiplied, so are alignment vector values
    previous_alignments = previous_alignments * upsampling_factor

    # Different resolution upsampling factors and tile sizes lead to different vector repetitions
    repeat_factor = upsampling_factor // (tile_size // previous_tile_size)

    # Upsampled_alignments shape can be less than expected if previous alignments
    # could not be computed over the whole image.
    # PyTorch's repeat_interleave maps to NumPy's repeat(axis=x).
    upsampled_alignments = previous_alignments.repeat_interleave(
        repeat_factor, dim=0
    ).repeat_interleave(repeat_factor, dim=1)

    # If the method is not defined, no need to go further in the upsampling
    if method is None:
        return upsampled_alignments

    h, w = upsampled_alignments.shape[0], upsampled_alignments.shape[1]

    # --- HDR+ method ---
    # Take as candidates the alignments for the 3 nearest coarse-scale tiles,
    # the nearest tile plus the next-nearest neighbor tiles in each dimension

    # Pad alignments by mirroring to avoid nearest neighbor tile problems on edges.
    # F.pad works on the last dims, so we permute (H, W, 2) to (2, H, W).
    prev_permuted = previous_alignments.permute(2, 0, 1)

    # PyTorch's 'replicate' mode requires 4D floating point tensors for spatial 2D padding
    is_float = prev_permuted.is_floating_point()
    tensor_to_pad = prev_permuted.unsqueeze(0)
    if not is_float:
        tensor_to_pad = tensor_to_pad.float()

    padded_permuted = F.pad(tensor_to_pad, (1, 1, 1, 1), mode="replicate").squeeze(0)

    if not is_float:
        padded_permuted = padded_permuted.to(dtype)

    padded_prev_alignments = padded_permuted.permute(1, 2, 0)  # Back to (H+2, W+2, 2)

    # Build the elemental tile to be repeated
    tile = torch.empty(
        (repeat_factor, repeat_factor, 2, 2), dtype=torch.long, device=device
    )
    rf2 = repeat_factor // 2

    tile[:rf2, :rf2] = torch.tensor([[-1, 0], [0, -1]], device=device)
    tile[:rf2, rf2:] = torch.tensor([[-1, 0], [0, 1]], device=device)
    tile[rf2:, :rf2] = torch.tensor([[1, 0], [0, -1]], device=device)
    tile[rf2:, rf2:] = torch.tensor([[1, 0], [0, 1]], device=device)

    # Repeat the elemental tile into the full mask of shape (H, W, 2, 2)
    neighbors_mask = tile.repeat(h // repeat_factor, w // repeat_factor, 1, 1)

    # Compute base indices as 2D grids (H, W)
    base_i = 2 + (torch.arange(h, device=device) // repeat_factor).view(h, 1).expand(
        h, w
    )
    base_j = 2 + (torch.arange(w, device=device) // repeat_factor).view(1, w).expand(
        h, w
    )

    # Calculate actual indices by adding the mask offsets and clamping
    max_h, max_w = (
        padded_prev_alignments.shape[0] - 1,
        padded_prev_alignments.shape[1] - 1,
    )

    ti1 = torch.clamp(base_i + neighbors_mask[:, :, 0, 0], 0, max_h)
    tj1 = torch.clamp(base_j + neighbors_mask[:, :, 0, 1], 0, max_w)
    ti2 = torch.clamp(base_i + neighbors_mask[:, :, 1, 0], 0, max_h)
    tj2 = torch.clamp(base_j + neighbors_mask[:, :, 1, 1], 0, max_w)

    # Extract the previously estimated motion vectors associated with neighbors
    ppa1 = padded_prev_alignments[ti1, tj1]  # Shape: (h, w, 2)
    ppa2 = padded_prev_alignments[ti2, tj2]  # Shape: (h, w, 2)

    # Get all possible tiles in the reference and alternate pyramid level
    ref_tiles = get_tiles(reference_pyramid_level, tile_size, 1)
    alt_tiles = get_tiles(alternate_pyramid_level, tile_size, 1)

    # Compute the L1 distance (outputs shape (h, w))
    d0 = compute_tiles_distance_l1(ref_tiles, alt_tiles, upsampled_alignments)
    d1 = compute_tiles_distance_l1(ref_tiles, alt_tiles, ppa1)
    d2 = compute_tiles_distance_l1(ref_tiles, alt_tiles, ppa2)

    # Stack distances to find the best candidate per pixel
    distances = torch.stack([d0, d1, d2], dim=0)  # Shape: (3, h, w)
    min_indices = torch.argmin(distances, dim=0)  # Shape: (h, w)

    # Stack candidates and gather the best ones
    candidates = torch.stack(
        [upsampled_alignments, ppa1, ppa2], dim=0
    )  # Shape: (3, h, w, 2)

    # Expand min_indices to match the last dimension (2) so we can use torch.gather
    min_indices_expanded = min_indices.unsqueeze(0).unsqueeze(-1).expand(1, h, w, 2)
    selected_alignments = torch.gather(candidates, 0, min_indices_expanded).squeeze(0)

    # Check if padding is required due to uncomputed bounds
    expected_h = reference_pyramid_level.shape[0] // (tile_size // 2) - 1
    expected_w = reference_pyramid_level.shape[1] // (tile_size // 2) - 1

    if h < expected_h or w < expected_w:
        # tiles where no alignment was computed will have an estimated motion of 0 pixels
        new_alignments = torch.zeros(
            (expected_h, expected_w, 2), dtype=dtype, device=device
        )
        new_alignments[:h, :w] = selected_alignments
    else:
        new_alignments = selected_alignments

    return new_alignments


def align_on_a_level(
    reference_pyramid_level: torch.Tensor,
    alternate_pyramid_level: torch.Tensor,
    upsampling_factor: int = 1,
    tile_size: int = 16,
    previous_tile_size: int | None = None,
    search_radius: int = 4,
    distance: str = "L2",
    subpixel: bool = True,
    previous_alignments: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Motion estimation performed at a single Gaussian pyramid level.
    """
    device = reference_pyramid_level.device

    # Distances shall be computed over float32 values.
    # By casting the input images here, we save casting much more data afterwards.
    if not reference_pyramid_level.is_floating_point():
        reference_pyramid_level = reference_pyramid_level.to(torch.float32)
    if not alternate_pyramid_level.is_floating_point():
        alternate_pyramid_level = alternate_pyramid_level.to(torch.float32)

    # Extract the tiles of the reference image overlapped by half in each spatial dimension
    ref_tiles = get_tiles(reference_pyramid_level, tile_size, step=tile_size // 2)

    # Upsample the previous alignments for initialization
    if previous_alignments is None:
        # no initial offset, search windows centered around reference tiles
        upsampled_alignments = torch.zeros(
            (ref_tiles.shape[0], ref_tiles.shape[1], 2),
            dtype=torch.float32,
            device=device,
        )
    else:
        # use the upsampled previous alignments as initial guesses
        assert previous_tile_size is not None, (
            "previous_tile_size must be an int when upsampling alignments"
        )
        upsampled_alignments = upsample_alignments(
            reference_pyramid_level,
            alternate_pyramid_level,
            previous_alignments,
            upsampling_factor,
            tile_size,
            previous_tile_size,
        )

    h, w = upsampled_alignments.shape[0], upsampled_alignments.shape[1]
    sr_dim = 2 * search_radius + 1

    # the initial offsets / alignment guesses [u0, v0] correspond to the upsampled previous alignments
    u0 = torch.round(upsampled_alignments[:, :, 0]).long()
    v0 = torch.round(upsampled_alignments[:, :, 1]).long()

    # Get all possible square search areas in the alternate image.
    # Pad the original image with a large constant (simulating infinite bounds).
    pad_tuple = (search_radius, search_radius, search_radius, search_radius)
    # F.pad requires (N, C, H, W) or (C, H, W) for 2D spatial padding
    alt_unsq = alternate_pyramid_level.unsqueeze(0).unsqueeze(0)
    padded_alt = (
        F.pad(alt_unsq, pad_tuple, mode="constant", value=65535.0).squeeze(0).squeeze(0)
    )

    # each area has a side of length (tileSize + 2 * searchRadius)
    search_window = tile_size + 2 * search_radius
    search_areas = get_tiles(padded_alt, window_size=search_window, step=1)

    # Only keep those corresponding to the area around the reference tile location + [u0, v0]
    base_i = (torch.arange(h, device=device) * (tile_size // 2)).view(h, 1).expand(h, w)
    base_j = (torch.arange(w, device=device) * (tile_size // 2)).view(1, w).expand(h, w)

    max_i = search_areas.shape[0] - 1
    max_j = search_areas.shape[1] - 1

    ind_i = torch.clamp(base_i + u0, 0, max_i)
    ind_j = torch.clamp(base_j + v0, 0, max_j)

    # PyTorch advanced indexing cleanly extracts the needed (h, w, window, window) tensor
    extracted_search_areas = search_areas[ind_i, ind_j]

    # The upsampled alignment grid can be larger than the ref tile grid when image
    # dimensions are not exact multiples of the tile stride.  Clamp h and w to the
    # ref tile grid size and trim every dependent tensor so shapes stay consistent.
    h = min(h, ref_tiles.shape[0])
    w = min(w, ref_tiles.shape[1])

    extracted_search_areas = extracted_search_areas[:h, :w]
    u0 = u0[:h, :w]
    v0 = v0[:h, :w]

    # compute_distance flattens distances internally returning shape (h*w, sr_dim*sr_dim)
    distances = compute_distance(
        ref_tiles[:h, :w], extracted_search_areas, distance_type=distance
    )

    # Get the indexes (within the search area) of alternate tiles of minimum distance wrt the reference
    min_idx = torch.argmin(distances, dim=1)  # Flattened indices

    # Unravel 1D indices to 2D indices (equivalent to np.unravel_index)
    level_offsets_i = (min_idx // sr_dim).float()
    level_offsets_j = (min_idx % sr_dim).float()
    level_offsets = torch.stack(
        [level_offsets_i, level_offsets_j], dim=1
    )  # Shape: (h*w, 2)

    if subpixel:
        subpix_offsets = torch.zeros_like(level_offsets)
        # only work on distance minimums that actually feature a 3x3 = 9 pixel neighborhood
        valid_mask = (min_idx >= 4) & (min_idx < distances.shape[1] - 4)

        if valid_mask.any():
            # Find the optimal offset only for valid positions
            subpix_offsets[valid_mask] = sub_pixel_minimum(
                distances[valid_mask], min_idx[valid_mask]
            )

        # Update the offsets
        level_offsets += subpix_offsets

    level_offsets = level_offsets.view(h, w, 2)

    # final values: initial guess [u0, v0] + current motion estimation that is actually between - and + searchRadius
    level_offsets[..., 0] = u0.float() + level_offsets[..., 0] - search_radius
    level_offsets[..., 1] = v0.float() + level_offsets[..., 1] - search_radius

    return level_offsets
