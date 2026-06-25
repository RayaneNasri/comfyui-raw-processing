import torch
import logging
from typing import Any

logger = logging.getLogger(__name__)


# Standard libraw/rawpy color index -> name mapping (color_desc is usually b"RGBG").
# raw_pattern is a 2x2 array of these indices giving the color at each of the
# 4 sub-pixel positions (row, col) within one CFA block.
_DEFAULT_COLOR_DESC = "RGBG"


def cfa_pattern_to_bayer_offsets(
    cfa_pattern: "torch.Tensor | list | tuple",
    color_desc: str = _DEFAULT_COLOR_DESC,
) -> list[tuple[int, int]]:
    """
    Convert a 2x2 CFA pattern (as produced by rawpy's `raw_pattern`, indexed via
    `color_desc`) into the [(row_offset, col_offset), ...] list expected by
    merge_hdrplus, ordered as [R, Gr, Gb, B] — i.e. the (0,0), (1,0), (0,1), (1,1)
    Bayer quadrants reinterpreted according to the *actual* sensor layout instead
    of assuming RGGB.

    Args:
        cfa_pattern: 2x2 array-like of small ints (0..3), one per sub-pixel
            position, indexing into `color_desc`. This matches rawpy's
            `raw_pattern` convention used by `read_raw_sensor_data`.
        color_desc: string mapping each index in `cfa_pattern` to a color
            letter. Defaults to rawpy's typical "RGBG" (two green channels,
            Gr and Gb, kept distinct).

    Returns:
        List of 4 (row, col) offsets in [R, Gr, Gb, B] order, matching the
        channel order merge_hdrplus iterates over.
    """
    if isinstance(cfa_pattern, torch.Tensor):
        pattern = cfa_pattern.detach().cpu().tolist()
    else:
        pattern = cfa_pattern

    # pattern[row][col] -> color index into color_desc
    positions: dict[str, tuple[int, int]] = {}
    green_positions: list[tuple[int, int]] = []

    for row in range(2):
        for col in range(2):
            color_idx = int(pattern[row][col])
            color = color_desc[color_idx]
            if color == "G":
                green_positions.append((row, col))
            else:
                positions[color] = (row, col)

    if "R" not in positions or "B" not in positions or len(green_positions) != 2:
        raise ValueError(
            f"Unsupported or malformed CFA pattern {pattern} with color_desc "
            f"'{color_desc}'. Expected exactly one R, one B, and two G positions."
        )

    # Disambiguate Gr (green sharing a column with R) vs Gb (green sharing a
    # column with B) — matches the convention of the original hard-coded RGGB
    # offsets [(0,0)->R, (1,0)->Gr, (0,1)->Gb, (1,1)->B], where (1,0) shares
    # its column with R at (0,0), and (0,1) shares its column with B at (1,1).
    r_col = positions["R"][1]
    gr = next((p for p in green_positions if p[1] == r_col), green_positions[0])
    gb = next((p for p in green_positions if p != gr), green_positions[1])

    return [positions["R"], gr, gb, positions["B"]]


def validate_consistent_cfa_pattern(
    cfa_patterns: "torch.Tensor | list",
) -> None:
    """
    Raise a clear error if frames in the burst disagree on their CFA pattern.
    Alignment and merging assume a single, consistent Bayer layout across the
    whole burst (standard for a burst captured by one camera); a mismatch
    usually means corrupted metadata or frames from different sources.
    """
    if isinstance(cfa_patterns, torch.Tensor):
        if cfa_patterns.ndim < 2:
            return  # single pattern, nothing to compare
        first = cfa_patterns[0]
        mismatched = [
            i
            for i in range(1, cfa_patterns.shape[0])
            if not torch.equal(cfa_patterns[i], first)
        ]
    else:
        first = cfa_patterns[0]
        to_list = lambda p: p.tolist() if isinstance(p, torch.Tensor) else p
        first_list = to_list(first)
        mismatched = [
            i
            for i in range(1, len(cfa_patterns))
            if to_list(cfa_patterns[i]) != first_list
        ]

    if mismatched:
        raise ValueError(
            "Burst frames disagree on CFA pattern at index(es) "
            f"{mismatched}. All frames in a burst must share the same sensor "
            "layout — check the loader output or remove mismatched frames."
        )


def merge_burst(
    images: list[torch.Tensor] | torch.Tensor,
    reference_index: int,
    aligned_tiles: torch.Tensor,
    padding: tuple[int, int, int, int],
    tags: dict[str, Any],
    black_level: list[int] | tuple[int, ...] | torch.Tensor,
    white_level: float | int,
    params: dict[str, Any],
    options: dict[str, Any],
    cfa_pattern: "torch.Tensor | list | tuple | None" = None,
) -> torch.Tensor:
    """
    Merges previously aligned tiles of a burst, and returns a single temporally denoised image.
    Stripped of all file I/O, raw demosaicking, and intermediate visual dumps.

    Args:
        cfa_pattern: 2x2 CFA pattern of the burst (rawpy `raw_pattern` convention).
            If None, defaults to standard RGGB offsets for backward compatibility.
    """

    # Extract the reference image tensor
    reference_image = images[reference_index]

    # The core HDR+ tile-based merging algorithm
    merged_image = merge_hdrplus(
        reference_image,
        aligned_tiles,
        padding,
        tags,
        black_level,
        white_level,
        params.get("tuning", {}),
        options,
        cfa_pattern,
    )

    return merged_image


def centered_cosine_window(x: torch.Tensor, window_size: int = 16) -> torch.Tensor:
    """
    1D version of the modified raised cosine window (Section 4.4 of the IPOL article).
    It is centered and nonzero at x=0 and x=windowSize-1.
    """
    return 0.5 - 0.5 * torch.cos(2.0 * torch.pi * (x + 0.5) / window_size)


def cosine_window_2d_patches(patches: torch.Tensor) -> torch.Tensor:
    """
    Apply a 2D version of the modified raised cosine window
    to a set of overlapped patches to avoid discontinuities and edge artifacts.
    """
    assert patches.ndim == 4, "not a 2D array of image patches"

    window_size = patches.shape[-1]  # Assumes patches are square
    device = patches.device

    # Ensure our math is done in floating point, even if patches are ints
    dtype = patches.dtype if patches.is_floating_point() else torch.float32

    # Create 1D coordinate tensor
    x = torch.arange(window_size, device=device, dtype=dtype)

    # Compute the attenuation window on 1 patch dimension
    line_weights = centered_cosine_window(x, window_size).unsqueeze(1)
    column_weights = line_weights.T

    # The 2D window is the product of the 1D window in both patch dimensions
    window = line_weights * column_weights

    # Apply the attenuation cosine weighting to all patches
    # PyTorch broadcasts the (W, W) window across the (H_tiles, W_tiles) batch dimensions automatically
    return patches * window


def cat_2d_patches(patches: torch.Tensor) -> torch.Tensor:
    """
    Stitches a 2D grid of 2D patches into a single continuous 2D tensor.
    Replaces the double np.concatenate trick.
    """
    assert patches.ndim == 4, "not a 2D array of 2D arrays"

    h_tiles, w_tiles, h_p, w_p = patches.shape
    # .contiguous() ensures the memory layout matches the new shape before viewing.
    return patches.permute(0, 2, 1, 3).contiguous().view(h_tiles * h_p, w_tiles * w_p)


def depatchify_overlap(patches: torch.Tensor) -> torch.Tensor:
    """
    Recreates a single image out of a 2D arrangement
    of patches overlapped by half in each dimension.
    """
    assert patches.ndim == 4, "not a 2D array of 2D patches"

    patch_size = patches.shape[-1]
    dp = patch_size // 2

    assert patch_size == patches.shape[-2] and patch_size % 2 == 0, (
        "function only supports square patches of even size"
    )

    # separate the different groups of overlapped patches
    patch_set_00 = patches[0::2, 0::2]  # original decomposition
    patch_set_01 = patches[0::2, 1::2]  # straddled by patch_size/2 in horizontal axis
    patch_set_10 = patches[1::2, 0::2]  # straddled by patch_size/2 in vertical axis
    patch_set_11 = patches[1::2, 1::2]  # straddled by patch_size/2 half in both axes

    # recreate sub-images from the different patch groups
    im_set_00 = cat_2d_patches(patch_set_00)
    im_set_01 = cat_2d_patches(patch_set_01)
    im_set_10 = cat_2d_patches(patch_set_10)
    im_set_11 = cat_2d_patches(patch_set_11)

    # reconstruct final image by correctly adding sub-images
    h_out = (patches.shape[0] + 1) * dp
    w_out = (patches.shape[1] + 1) * dp

    reconstructed_image = torch.zeros(
        (h_out, w_out), dtype=im_set_00.dtype, device=patches.device
    )

    reconstructed_image[0 : im_set_00.shape[0], 0 : im_set_00.shape[1]] = im_set_00
    reconstructed_image[0 : im_set_01.shape[0], dp : im_set_01.shape[1] + dp] += (
        im_set_01
    )
    reconstructed_image[dp : im_set_10.shape[0] + dp, 0 : im_set_10.shape[1]] += (
        im_set_10
    )
    reconstructed_image[dp : im_set_11.shape[0] + dp, dp : im_set_11.shape[1] + dp] += (
        im_set_11
    )

    return reconstructed_image


def patches_rms(patches: torch.Tensor) -> torch.Tensor:
    """
    Computes the Root-Mean-Square of a set of patches/tiles.
    """
    assert patches.ndim >= 3, "not an nD array of patches"

    # PyTorch natively computes mean over multiple dimensions simultaneously.
    # No need to flatten the patches!
    return torch.sqrt(torch.mean(patches**2, dim=(-1, -2)))


def merge_wiener_dft_patches(
    ref_patches_fft: torch.Tensor,
    alt_patches_fft: torch.Tensor,
    noise_variance: torch.Tensor | float,
    temporal_factor: float = 8.0,
) -> torch.Tensor:
    """
    Temporally denoise a pair of sets of 2D Frequency-domain patches
    Using a variant of the Wiener filter (Section 4.2 of the IPOL article).
    """
    patch_size = ref_patches_fft.shape[-1]

    # scale the noise variance to match the scale of dSq
    # 1 / 4**2 becomes 1.0 / 16.0
    noise_scaling = (patch_size**2) * (1.0 / 16.0) * 2.0 * temporal_factor
    scaled_variance = noise_scaling * noise_variance

    # --- Native PyTorch Wiener Filter Logic ---
    # Keep the difference in memory for later use
    diff = ref_patches_fft - alt_patches_fft

    # Compute the squared absolute difference
    # PyTorch's .abs() on a complex tensor naturally computes the magnitude
    dist = diff.abs() ** 2

    # Derive a shrinkage operator (Wiener filter variant)
    # A is automatically real-valued, matching the original logic perfectly
    A = dist / (dist + scaled_variance)

    # The merge result is an interpolation of reference and alternate patches guided by operator A
    return alt_patches_fft + A * diff


def temporal_denoise_pair_patches(
    reference_image_patches: torch.Tensor,
    alternate_image_patches: torch.Tensor,
    noise_variance: torch.Tensor | float,
    temporal_factor: float = 8.0,
    method: str = "DFTWiener",
) -> torch.Tensor:
    """
    Temporally denoise two sets of 2D patches, and return a single set of denoised patches.
    """
    if method == "keepAlternate":
        # only return alternate image patches (e.g. to later average them)
        merged_image_patches = alternate_image_patches

    elif method == "pairedAverage":
        merged_image_patches = 0.5 * (reference_image_patches + alternate_image_patches)

    elif method == "DFTWiener":
        # reference_image_patches already contains 2D DFT of patches
        # Apply 2D FFT to the spatial dimensions (-2, -1) of the alternate patches
        alt_patches_fft = torch.fft.fft2(
            alternate_image_patches.to(torch.float32), dim=(-2, -1)
        )

        merged_image_patches = merge_wiener_dft_patches(
            reference_image_patches, alt_patches_fft, noise_variance, temporal_factor
        )
    else:
        raise ValueError(f"Unknown temporal denoising method: {method}")

    return merged_image_patches


def spatial_denoise_patches(
    patches_fft: torch.Tensor,
    noise_variance: torch.Tensor | float,
    spatial_factor: float,
) -> torch.Tensor:
    """
    Spatially denoise a set of 2D Frequency-domain patches
    Using a variant of the Wiener filter (Section 4.3 of the IPOL article).
    """
    assert patches_fft.ndim == 4, "not a 2D array of 2D patches"

    # Assumes square patches
    patch_size = patches_fft.shape[-1]
    device = patches_fft.device

    # Create a patch of distance of spatial frequency module with respect to the origin.
    # PyTorch broadcasting natively handles the Nx1 and 1xN grid generation.
    grid_1d = torch.arange(patch_size, dtype=torch.float32, device=device) - (
        patch_size / 2.0
    )

    row_distances = grid_1d.unsqueeze(1)  # Shape: (patch_size, 1)
    col_distances = grid_1d.unsqueeze(0)  # Shape: (1, patch_size)

    distance_patch = torch.sqrt(row_distances**2 + col_distances**2)

    # Shift the zero-frequency component to the corners to match the FFT layout
    dist_patch_shift = torch.fft.ifftshift(distance_patch)

    # Scale the noise variance to match the scale of patches_fft.abs()**2
    # 1 / 4**2 becomes 1.0 / 16.0
    noise_scaling = (patch_size**2) * (1.0 / 16.0) * spatial_factor

    wiener_coeff = dist_patch_shift * noise_scaling * noise_variance

    # --- Native PyTorch Wiener Filtering (Replaces Numba helper) ---
    # Compute the squared magnitude of the complex patches
    dist = patches_fft.abs() ** 2

    # Wiener filtering
    return patches_fft * dist / (dist + wiener_coeff)


def get_noise_params(
    tags: dict[str, Any],
    black_level: list[int] | tuple[int, ...] | torch.Tensor,
    white_level: int | float,
    params: dict[str, Any],
    options: dict[str, Any],
) -> tuple[float, float]:
    """
    Retrieve noise curve parameters either from EXIF metadata, ISO and baseline values,
    or from an input tuple (Section 4.1 of the IPOL article).
    """
    if isinstance(params.get("noiseCurve"), tuple):
        lambda_s, lambda_r = params["noiseCurve"]
    else:
        noise_curve = params.get("noiseCurve")

        if noise_curve == "exifNoiseProfile":
            logger.info(
                "Looking for noise curve parameters in the NoiseProfile EXIF tag"
            )

            if "Image Tag 0xC761" in tags:
                # Typically a NumPy array or list from rawpy/exifread
                noise_profile = tags["Image Tag 0xC761"].values
                # Squeeze functionally removes single-dimensional entries
                if hasattr(noise_profile, "squeeze"):
                    noise_profile = noise_profile.squeeze()

                if len(noise_profile) == 2:
                    lambda_sn = float(noise_profile[0])
                    lambda_rn = float(noise_profile[1])
                else:  # if noiseProfile has one value per CFA color
                    assert len(noise_profile) == 6
                    assert noise_profile[0] == noise_profile[2] == noise_profile[4], (
                        "NoiseProfile tag differs per channel"
                    )
                    assert noise_profile[1] == noise_profile[3] == noise_profile[5], (
                        "NoiseProfile tag differs per channel"
                    )
                    lambda_sn = float(noise_profile[0])
                    lambda_rn = float(noise_profile[1])
            else:
                iso = 100
                if "Image ISOSpeedRatings" in tags:
                    iso = tags["Image ISOSpeedRatings"].values[0]
                elif "EXIF ISOSpeedRatings" in tags:
                    iso = tags["EXIF ISOSpeedRatings"].values[0]

                if iso == 0:
                    iso = 100  # some images can have incorrect ISO data

                logger.info(
                    f"NoiseProfile tag not found. Computing lambdaS and lambdaR from ISO({iso}) and baseline values"
                )

                # Average values normalized at ISO 100 from images with the NoiseProfile tag
                baseline_lambda_s, baseline_lambda_r = (
                    3.24 * 10 ** (-4),
                    4.3 * 10 ** (-6),
                )
                lambda_sn = (iso / 100.0) * baseline_lambda_s
                lambda_rn = (iso / 100.0) ** 2 * baseline_lambda_r

        elif noise_curve == "exifISO":
            iso = 100
            if "Image ISOSpeedRatings" in tags:
                iso = tags["Image ISOSpeedRatings"].values[0]
            elif "EXIF ISOSpeedRatings" in tags:
                iso = tags["EXIF ISOSpeedRatings"].values[0]

            if iso == 0:
                iso = 100  # some images can have incorrect ISO data

            logger.info(
                f"Computing lambdaS and lambdaR from ISO({iso}) and baseline values"
            )
            baseline_lambda_s, baseline_lambda_r = 3.24 * 10 ** (-4), 4.3 * 10 ** (-6)
            lambda_sn = (iso / 100.0) * baseline_lambda_s
            lambda_rn = (iso / 100.0) ** 2 * baseline_lambda_r

        else:
            raise ValueError(f"Unknown noiseCurve parameter: {noise_curve}")

        # Un-normalize: var(k*x) = k**2*var(x) = k**2*(lambdaSn*x+lambdaRn) = k*lambdaSn*(k*x) + k**2*lambdaRn
        if isinstance(black_level, torch.Tensor):
            b_l = torch.min(black_level).item()
        else:
            b_l = min(black_level)

        lambda_s = lambda_sn * (white_level - b_l)
        lambda_r = lambda_rn * (white_level - b_l) ** 2

    logger.debug(f"Noise curve parameters: lambdaS={lambda_s}, lambdaR={lambda_r}")
    return float(lambda_s), float(lambda_r)


def merge_channel_hdrplus(
    reference_channel: torch.Tensor,
    aligned_channel_tiles: torch.Tensor,
    lambda_s: float,
    lambda_r: float,
    params: dict[str, Any],
    options: dict[str, Any],
) -> torch.Tensor:
    """
    Perform per-channel, tile-based, pairwise temporal denoising
    as defined in Algorithm 2 of Section 4.2 of the IPOL article.
    """
    # aligned_channel_tiles shape assumption: (num_images, h_tiles, w_tiles, patch_size, patch_size)
    reference_channel_tiles = aligned_channel_tiles[0]

    # Initialize noise variance as a simple scalar/empty tensor placeholder
    noise_variance: torch.Tensor | float = 0.0

    method = params.get("method", "DFTWiener")
    temporal_factor = options.get("temporalFactor", 8.0)
    spatial_factor = options.get("spatialFactor", 0.0)

    if method == "DFTWiener":
        # Compute noise model of reference image (signal level is constant per patch)
        signal_level = patches_rms(reference_channel_tiles)

        # noiseVariance = lambdaS * signalLevel + lambdaR
        # We use .unsqueeze() instead of .repeat() to save memory.
        # PyTorch will broadcast this (h_tiles, w_tiles, 1, 1) tensor seamlessly.
        noise_variance = lambda_s * signal_level.unsqueeze(-1).unsqueeze(-1) + lambda_r

        # Compute 2D DFT of reference image tiles natively on the GPU
        reference_channel_tiles = torch.fft.fft2(
            reference_channel_tiles.to(torch.float32), dim=(-2, -1)
        )

    # Pairwise merging of reference image channel with itself = reference image channel
    merged_pairs_tiles_sum = reference_channel_tiles.clone()

    if temporal_factor == 0:
        # No temporal denoising
        merged_channel_tiles = merged_pairs_tiles_sum
    else:
        # Pairwise merging of all other images with reference image
        num_images = aligned_channel_tiles.shape[0]
        for i in range(1, num_images):
            merged_pairs_tiles_sum += temporal_denoise_pair_patches(
                reference_channel_tiles,
                aligned_channel_tiles[i],
                noise_variance,
                temporal_factor,
                method,
            )

        # Final merge = mean of all pairwise merges
        merged_channel_tiles = merged_pairs_tiles_sum / num_images

    if method == "DFTWiener":
        if spatial_factor > 0:
            num_images = aligned_channel_tiles.shape[0]
            # Update noise variance estimation after temporal denoising
            noise_variance = noise_variance / num_images

            # Perform fast spatial denoising using a Wiener filter
            merged_channel_tiles = spatial_denoise_patches(
                merged_channel_tiles, noise_variance, spatial_factor
            )

        # Apply Inverse 2D FFT natively and take the real component
        merged_channel_tiles = torch.fft.ifft2(merged_channel_tiles, dim=(-2, -1)).real

    # Apply a cosine window function to the overlapping patches to avoid edge artifacts
    merged_channel_tiles = cosine_window_2d_patches(merged_channel_tiles)

    # Reconstruct an image from the patches and return to the original data type
    merged_channel = depatchify_overlap(merged_channel_tiles).to(
        reference_channel.dtype
    )

    return merged_channel


def merge_hdrplus(
    reference_image: torch.Tensor,
    aligned_tiles: torch.Tensor,
    padding: tuple[int, int, int, int],
    tags: dict[str, Any],
    black_level: list[int] | tuple[int, ...] | torch.Tensor,
    white_level: int | float,
    params: dict[str, Any],
    options: dict[str, Any],
    cfa_pattern: "torch.Tensor | list | tuple | None" = None,
) -> torch.Tensor:
    """
    Implements the Fourier Tile-based Merging as described in Section 4 of the IPOL article.
    Processes each of the 4 Bayer channels (R, Gr, Gb, B) separately.

    Args:
        cfa_pattern: 2x2 CFA pattern of the sensor (rawpy `raw_pattern` convention,
            i.e. a 2x2 array of color indices into "RGBG"). If None, assumes the
            standard RGGB layout: [(0,0)->R, (1,0)->Gr, (0,1)->Gb, (1,1)->B].
    """
    device = reference_image.device
    dtype = reference_image.dtype

    # aligned_tiles shape assumption: (num_images, h_tiles, w_tiles, tile_size, tile_size)
    tile_size = aligned_tiles.shape[3]

    # Reconstruct full image dimensions based on the 50% overlapping tiles
    h = (aligned_tiles.shape[1] + 1) * tile_size // 2
    w = (aligned_tiles.shape[2] + 1) * tile_size // 2

    # Allocate the final image buffer directly on the same device as the input
    merged_image = torch.empty((h, w), dtype=dtype, device=device)

    # Get noise curve parameters
    lambda_s, lambda_r = get_noise_params(
        tags, black_level, white_level, params, options
    )

    # Work separately on each channel of the Bayer image.
    # Offsets are ordered [R, Gr, Gb, B] and derived from the sensor's actual CFA
    # pattern when provided, instead of always assuming RGGB.
    if cfa_pattern is not None:
        bayer_offsets = cfa_pattern_to_bayer_offsets(cfa_pattern)
        logger.debug(f"Using CFA-derived Bayer offsets: {bayer_offsets}")
    else:
        logger.warning(
            "No CFA pattern provided to merge_hdrplus; assuming standard RGGB "
            "layout. Pass the burst's actual CFA pattern if the sensor uses a "
            "different layout (BGGR, GRBG, GBRG)."
        )
        bayer_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]

    for c, (di, dj) in enumerate(bayer_offsets):
        # Extract the specific channel from the reference image
        ref_channel = reference_image[di::2, dj::2]

        # Slice the spatial dimensions of the tiles to extract just this Bayer channel.
        # This reduces the patch_size by half for the channel processing.
        tiles_channel = aligned_tiles[..., di::2, dj::2]

        # Perform burst merging as pairwise merging for this channel
        merged_channel = merge_channel_hdrplus(
            ref_channel, tiles_channel, lambda_s, lambda_r, params, options
        )

        # Place the merged channel back into the full resolution Bayer image matrix
        merged_image[di::2, dj::2] = merged_channel

    # Discard the padding that was applied to the original images prior to alignment
    pad_top, pad_bottom, pad_left, pad_right = padding

    # h - pad_bottom is safe even if pad_bottom is 0
    return merged_image[pad_top : h - pad_bottom, pad_left : w - pad_right]
