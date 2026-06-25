import math
from typing import Any
import cv2
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def gamma_compress(
    x: torch.Tensor, threshold: float, gain_min: float, gain_max: float, exponent: float
) -> torch.Tensor:
    """Applies a piecewise gamma compression curve to an image tensor."""
    is_integer = not x.is_floating_point()
    max_val = 65535.0

    # Normalize 16-bit integer inputs to [0.0, 1.0] float for the math
    if is_integer:
        max_val = 65535.0
        x_float = x.to(torch.float32) / max_val
    else:
        x_float = x

    mask = x_float <= threshold

    low_val = gain_min * x_float
    high_val = gain_max * (x_float**exponent) - gain_max + 1.0

    out = torch.where(mask, low_val, high_val)
    out = torch.clamp(out, 0.0, 1.0)

    if is_integer:
        out = torch.round(out * max_val).to(x.dtype)

    return out


def gamma_decompress(
    x: torch.Tensor, threshold: float, gain_min: float, gain_max: float, exponent: float
) -> torch.Tensor:
    """Applies a piecewise gamma decompression curve to an image tensor."""
    is_integer = not x.is_floating_point()
    max_val = 65535.0

    if is_integer:
        x_float = x.to(torch.float32) / max_val
    else:
        x_float = x

    mask = x_float <= threshold

    low_val = x_float / gain_min

    # Clamp the base to 0 to prevent NaNs when taking the power of negative floats
    base = torch.clamp((x_float + gain_max - 1.0) / gain_max, min=0.0)
    high_val = base**exponent

    out = torch.where(mask, low_val, high_val)
    out = torch.clamp(out, 0.0, 1.0)

    if is_integer:
        out = torch.round(out * max_val).to(x.dtype)

    return out


def gamma_srgb(image: torch.Tensor, mode: str = "compress") -> torch.Tensor:
    """sRGB transfer function wrapper."""
    if mode == "compress":
        return gamma_compress(image, 0.0031308, 12.92, 1.055, 1.0 / 2.4)
    elif mode == "decompress":
        return gamma_decompress(image, 0.04045, 12.92, 1.055, 2.4)
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Supported modes are 'compress' or 'decompress'."
        )


def gamma_rec709(image: torch.Tensor, mode: str = "compress") -> torch.Tensor:
    """REC709 transfer function wrapper."""
    if mode == "compress":
        return gamma_compress(image, 0.018, 4.5, 1.099, 1.0 / 2.2)
    elif mode == "decompress":
        return gamma_decompress(image, 0.081, 4.5, 1.099, 2.2)
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Supported modes are 'compress' or 'decompress'."
        )


def rgb_to_yuv(rgb_image: torch.Tensor) -> torch.Tensor:
    """Converts an RGB image tensor to YUV color space."""
    assert rgb_image.ndim == 3, "not a 3D image tensor"
    # Ensuring the channel dimension is at the end (H, W, 3) to match np.dot behavior
    assert rgb_image.shape[-1] == 3, "expected channel dimension to be 3 (H, W, 3)"
    assert rgb_image.is_floating_point(), "expected a float image tensor"
    assert rgb_image.min() >= 0.0 and rgb_image.max() <= 1.0, (
        "expected float image between 0.0 and 1.0"
    )

    # Create the matrix on the same device and dtype as the image tensor
    cv_mat = torch.tensor(
        [
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001],
        ],
        device=rgb_image.device,
        dtype=rgb_image.dtype,
    )

    # In PyTorch, @ performs matrix multiplication (equivalent to np.dot on the last dim)
    return torch.clamp(rgb_image @ cv_mat.T, 0.0, 1.0)


def yuv_to_rgb(yuv_image: torch.Tensor) -> torch.Tensor:
    """Converts a YUV image tensor back to RGB color space."""
    assert yuv_image.ndim == 3, "not a 3D image tensor"
    assert yuv_image.shape[-1] == 3, "expected channel dimension to be 3 (H, W, 3)"
    assert yuv_image.is_floating_point(), "expected a float image tensor"
    assert yuv_image.min() >= 0.0 and yuv_image.max() <= 1.0, (
        "expected float image between 0.0 and 1.0"
    )

    cv_mat = torch.tensor(
        [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]],
        device=yuv_image.device,
        dtype=yuv_image.dtype,
    )

    return torch.clamp(yuv_image @ cv_mat.T, 0.0, 1.0)


def mean_channels(r: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    PyTorch equivalent of Numba vectorized mean_.
    Element-wise operations natively broadcast in PyTorch.
    """
    return (r + g + b) / 3.0


def mean_gain(
    r: torch.Tensor, g: torch.Tensor, b: torch.Tensor, k: float | torch.Tensor
) -> torch.Tensor:
    """
    PyTorch equivalent of Numba vectorized meanGain_.
    Applies gain, clamps to [0, 1], and averages.
    """
    rk = torch.clamp(r * k, 0.0, 1.0)
    gk = torch.clamp(g * k, 0.0, 1.0)
    bk = torch.clamp(b * k, 0.0, 1.0)

    return (rk + gk + bk) / 3.0


def apply_scaling(
    merged_image: torch.Tensor, short_gray: torch.Tensor, fused_gray: torch.Tensor
) -> torch.Tensor:
    """
    Scales each RGB channel of the short exposure based on the fused tone mapping.
    Replaces the Numba @guvectorize nested loops.
    """
    # Create a safe denominator to prevent divide-by-zero NaNs
    safe_short = torch.where(short_gray == 0.0, torch.ones_like(short_gray), short_gray)

    # Calculate scale map: 1.0 if short_gray is 0, else fused_gray / short_gray
    s = torch.where(
        short_gray == 0.0, torch.ones_like(short_gray), fused_gray / safe_short
    )

    # Expand the (H, W) scale map to (H, W, 1) so it broadcasts across RGB channels
    s = s.unsqueeze(-1)

    # Apply scaling and clip
    scaled_image = merged_image * s
    return torch.clamp(scaled_image, 0.0, 1.0)


def local_tone_map(
    merged_image: torch.Tensor, options: dict[str, Any]
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform HDR tone mapping via exposure fusion using synthetic exposures
    (as described in Section 5.2).
    """
    device = merged_image.device

    # Work with grayscale images (assuming merged_image is H, W, C)
    short_gray = mean_channels(
        merged_image[..., 0], merged_image[..., 1], merged_image[..., 2]
    )

    # Compute gain
    ltm_gain = options.get("ltmGain", -1)
    if ltm_gain == -1:
        # Native PyTorch downsampling for heuristic estimation (1/25th scale)
        # F.interpolate requires (N, C, H, W) format
        sg_tensor = short_gray.unsqueeze(0).unsqueeze(0)
        short_s = F.interpolate(sg_tensor, scale_factor=1 / 25.0, mode="area").view(-1)

        best_gain = False
        gain = 0.0
        compression = 1.0
        saturated = 0.0

        short_sg = gamma_srgb(short_s, "compress")
        ss_mean = torch.mean(short_sg).item()

        while (compression < 1.9 and saturated < 0.95) or (
            not best_gain and compression < 6.0 and gain < 30.0 and saturated < 0.33
        ):
            gain += 2.0
            long_sg = torch.clamp(gamma_srgb(gain * short_s, "compress"), 0.0, 1.0)
            ls_mean = torch.mean(long_sg).item()

            compression = ls_mean / ss_mean if ss_mean > 0 else 1.0
            best_gain = ls_mean > (1.0 - ss_mean) / 2.0
            saturated = torch.sum(long_sg > 0.95).item() / long_sg.numel()

    else:
        gain = float(ltm_gain)

    # create a synthetic long exposure
    long_gray = mean_gain(
        merged_image[..., 0], merged_image[..., 1], merged_image[..., 2], gain
    )

    # apply gamma correction to both
    short_g = gamma_srgb(short_gray, "compress")
    long_g = gamma_srgb(long_gray, "compress")

    # --- OpenCV Mertens Bridge ---
    # OpenCV expects float arrays scaled 0 to 255.
    sg_np = (short_g * 255.0).detach().cpu().numpy().astype("float32")
    lg_np = (long_g * 255.0).detach().cpu().numpy().astype("float32")

    # perform tone mapping by exposure fusion in grayscale
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=0.0, saturation_weight=0.0, exposure_weight=1.0
    )

    # MergeMertens returns a float32 array in the range [0.0, 1.0] despite the 255 input
    fused_np = merge_mertens.process([sg_np, lg_np])

    # Bring the fused result safely back to the PyTorch device
    fused_g = torch.from_numpy(fused_np).to(device)
    # ------------------------------

    # undo gamma correction
    fused_gray = gamma_srgb(fused_g, "decompress")

    # scale each RGB channel of the short exposure accordingly
    ltm_image = apply_scaling(merged_image, short_gray, fused_gray)

    return ltm_image, gain, short_g, long_g, fused_g


def enhance_contrast(image: torch.Tensor, options: dict[str, Any]) -> torch.Tensor:
    """
    Perform contrast enhancement with an S-shaped function
    (as described in Section 5.2).
    """
    gain = options.get("gtmContrast", 0.0)

    # Validation checks
    assert 0.0 <= gain <= 1.0, "expected a contrast enhancement ratio between 0 and 1"
    assert image.is_floating_point(), "expected a float image tensor"
    # For performance, you might want to remove the min/max checks in production
    # if you know your pipeline guarantees [0, 1] inputs, but they are kept here for safety.
    assert image.min() >= 0.0 and image.max() <= 1.0, (
        "expected float image between 0.0 and 1.0"
    )

    # Apply an S-shaped contrast enhancement curve natively
    enhanced = image - gain * torch.sin(2.0 * torch.pi * image)

    # Clip the result
    return torch.clamp(enhanced, 0.0, 1.0)


def dist_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Replaces the Numba distL1_ function.
    PyTorch handles absolute differences natively and optimally.
    """
    return torch.abs(x - y)


def sharpen_triple(
    x: torch.Tensor,
    b0: torch.Tensor,
    l0: torch.Tensor,
    th0: float | torch.Tensor,
    k0: float | torch.Tensor,
    b1: torch.Tensor,
    l1: torch.Tensor,
    th1: float | torch.Tensor,
    k1: float | torch.Tensor,
    b2: torch.Tensor,
    l2: torch.Tensor,
    th2: float | torch.Tensor,
    k2: float | torch.Tensor,
) -> torch.Tensor:
    """
    Computes three conditionally sharpened values, averages them, and clips the result.
    Replaces the Numba @vectorize sharpenTriple_ function.
    """
    # Compute the three sharpened values using vectorized conditional logic
    r0 = torch.where(l0 < th0, x, x + k0 * (x - b0))
    r1 = torch.where(l1 < th1, x, x + k1 * (x - b1))
    r2 = torch.where(l2 < th2, x, x + k2 * (x - b2))

    # Average them
    r = (r0 + r1 + r2) / 3.0

    # Clip the result
    return torch.clamp(r, 0.0, 1.0)


def apply_gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Applies a Gaussian blur to a (H, W, C) or (H, W) tensor, keeping operations
    100% on the GPU. Mimics OpenCV's cv2.GaussianBlur ksize=(0,0) behavior.
    """
    if sigma <= 0:
        return image

    device = image.device
    dtype = image.dtype

    # OpenCV's default rule for kernel size based on sigma:
    ksize = int(math.ceil(sigma * 3.0)) * 2 + 1

    # Generate 1D Gaussian kernel
    x = torch.arange(-ksize // 2 + 1.0, ksize // 2 + 1.0, device=device, dtype=dtype)
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Create 2D kernel
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)

    orig_ndim = image.ndim
    if orig_ndim == 2:
        # Pad to (Batch, Channels, H, W) -> (1, 1, H, W)
        image_4d = image.unsqueeze(0).unsqueeze(0)
        groups = 1
    elif orig_ndim == 3:
        # Assuming (H, W, C) based on previous NumPy structures.
        # Permute to (C, H, W) then unsqueeze to (1, C, H, W)
        image_4d = image.permute(2, 0, 1).unsqueeze(0)
        groups = image.shape[2]
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {orig_ndim}D")

    # Expand kernel to match groups (channels)
    # Shape needs to be (out_channels, in_channels/groups, kH, kW) -> (groups, 1, ksize, ksize)
    kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0).expand(groups, 1, ksize, ksize)

    # Pad the image symmetrically (matches OpenCV's BORDER_REFLECT_101)
    pad = ksize // 2
    padded_image = F.pad(image_4d, (pad, pad, pad, pad), mode="reflect")

    # Apply grouped convolution
    blurred_4d = F.conv2d(padded_image, kernel_4d, groups=groups)

    # Revert back to original shape
    if orig_ndim == 2:
        return blurred_4d.squeeze(0).squeeze(0)
    else:
        return blurred_4d.squeeze(0).permute(1, 2, 0)  # Back to (H, W, C)


def apply_triple_sharpening(
    image: torch.Tensor, params: dict[str, Any], options: dict[str, Any]
) -> torch.Tensor:
    """
    Perform sharpening with unsharp masking.
    The mask is a linear combination of convolutions of the input image
    with 3 Gaussian kernels of different sizes (as described in Section 5.2).
    """
    sigmas = params["sharpenSigma"]
    amounts = params["sharpenAmount"]
    thresholds = params["sharpenThreshold"]

    # Compute all Gaussian blurs natively on GPU
    blur0 = apply_gaussian_blur(image, sigmas[0])
    blur1 = apply_gaussian_blur(image, sigmas[1])
    blur2 = apply_gaussian_blur(image, sigmas[2])

    # Compute all low contrast images using the L1 absolute difference helper
    low0 = dist_l1(blur0, image)
    low1 = dist_l1(blur1, image)
    low2 = dist_l1(blur2, image)

    # Compute the triple sharpen (using the previously translated math logic)
    sharp_image = sharpen_triple(
        image,
        blur0,
        low0,
        thresholds[0],
        amounts[0],
        blur1,
        low1,
        thresholds[1],
        amounts[1],
        blur2,
        low2,
        thresholds[2],
        amounts[2],
    )

    return sharp_image


def finish(
    merged_image: torch.Tensor, params: dict[str, Any], options: dict[str, Any]
) -> torch.Tensor:
    """
    Perform the finishing steps (Tone Mapping, Gamma, Sharpening).
    Stripped of all file I/O and raw demosaicking.

    Args:
        merged_image: RGB image tensor.
        params: dict containing algorithm parameters.
        options: dict containing processing options.

    Returns:
        The final processed image as a PyTorch tensor in the range [0.0, 1.0].
    """

    # Ensure we are working with a float tensor in the [0, 1] range.
    # If a 16-bit integer tensor is passed, normalize it.
    if not merged_image.is_floating_point():
        processed_image = merged_image.to(torch.float32) / 65535.0
    else:
        # Clone to avoid modifying the original tensor in-place
        processed_image = merged_image.clone()

    # Local Tone Mapping (LTM)
    if options.get("ltmGain"):
        # local_tone_map returns (ltm_image, gain, short_g, long_g, fused_g)
        # We only care about the final ltm_image output here.
        processed_image, _, _, _, _ = local_tone_map(processed_image, options)

    # Global Tone Mapping (GTM) / Contrast Enhancement
    if options.get("gtmContrast"):
        processed_image = enhance_contrast(processed_image, options)

    # TODO: separate this:
    # Apply the final sRGB gamma compression curve
    processed_image = gamma_srgb(processed_image, mode="compress")

    # Apply Unsharp Masking (Triple Sharpening)
    if "tuning" in params:
        processed_image = apply_triple_sharpening(
            processed_image, params["tuning"], options
        )

    return processed_image
