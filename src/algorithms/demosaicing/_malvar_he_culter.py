import torch
import torch.nn.functional as F


def malvar_he_cutler_demosaicing(
    raw_image: torch.Tensor,
    dx: int = 0,
    dy: int = 0,
) -> torch.Tensor:
    """
    Vectorized implementation of Malvar-He-Cutler demosaicing using PyTorch convolutions.
    """
    if raw_image.ndim != 2:
        raise ValueError("raw_image must be a 2D tensor of shape (H, W)")
    if dx not in (0, 1) or dy not in (0, 1):
        raise ValueError("dx and dy must be 0 or 1")

    # Handle Empty/Zero-Size Input
    height, width = raw_image.shape
    if height == 0 or width == 0:
        return torch.zeros(
            (height, width, 3), dtype=raw_image.dtype, device=raw_image.device
        )

    # Ensure input is float and has proper shape for conv2d (1, 1, H, W)
    img = raw_image.float()
    img_batch = img.unsqueeze(0).unsqueeze(0)

    # Kernel for estimating Green at Red/Blue locations
    k_g = (
        torch.tensor(
            [
                [0, 0, -1, 0, 0],
                [0, 0, 2, 0, 0],
                [-1, 2, 4, 2, -1],
                [0, 0, 2, 0, 0],
                [0, 0, -1, 0, 0],
            ],
            dtype=img.dtype,
            device=img.device,
        )
        / 8.0
    )

    # Kernel for estimating Red at Blue locations (or Blue at Red)
    k_rb = (
        torch.tensor(
            [
                [0, 0, -1.5, 0, 0],
                [0, 2, 0, 2, 0],
                [-1.5, 0, 6, 0, -1.5],
                [0, 2, 0, 2, 0],
                [0, 0, -1.5, 0, 0],
            ],
            dtype=img.dtype,
            device=img.device,
        )
        / 8.0
    )

    # Kernel for estimating Red at Green locations in red rows (or Blue at Green locations in Red rows)
    k_rg_h = (
        torch.tensor(
            [
                [0, 0, 0.5, 0, 0],
                [0, -1, 0, -1, 0],
                [-1, 4, 5, 4, -1],
                [0, -1, 0, -1, 0],
                [0, 0, 0.5, 0, 0],
            ],
            dtype=img.dtype,
            device=img.device,
        )
        / 8.0
    )

    # Kernel for estimating Red at Green locations in blue rows (or Blue at Green locations in Blue rows)
    k_rg_v = k_rg_h.t()

    # Reshape kernels for F.conv2d
    k_g = k_g.view(1, 1, 5, 5)
    k_rb = k_rb.view(1, 1, 5, 5)
    k_rg_h = k_rg_h.view(1, 1, 5, 5)
    k_rg_v = k_rg_v.view(1, 1, 5, 5)

    # Reflect padding crashes if dim < pad (2). Use 'replicate' for tiny images.
    if height < 3 or width < 3:
        pad_mode = "replicate"
    else:
        pad_mode = "reflect"

    padded_img = F.pad(img_batch, (2, 2, 2, 2), mode=pad_mode)

    G_est = F.conv2d(padded_img, k_g)
    RB_est = F.conv2d(padded_img, k_rb)
    RG_h_est = F.conv2d(padded_img, k_rg_h)
    RG_v_est = F.conv2d(padded_img, k_rg_v)

    # Create Bayer Masks
    y_indices = torch.arange(height, device=img.device).view(-1, 1)
    x_indices = torch.arange(width, device=img.device).view(1, -1)

    mask_r = ((y_indices % 2) == dy) & ((x_indices % 2) == dx)
    mask_b = ((y_indices % 2) == (1 - dy)) & ((x_indices % 2) == (1 - dx))
    mask_g_r = ((y_indices % 2) == dy) & ((x_indices % 2) == (1 - dx))
    mask_g_b = ((y_indices % 2) == (1 - dy)) & ((x_indices % 2) == dx)

    mask_r = mask_r.float()
    mask_b = mask_b.float()
    mask_g_r = mask_g_r.float()
    mask_g_b = mask_g_b.float()
    mask_g = mask_g_r + mask_g_b

    # Green Channel
    G_final = img * mask_g + G_est.squeeze() * (mask_r + mask_b)

    # Red Channel
    R_final = (
        img * mask_r
        + RB_est.squeeze() * mask_b
        + RG_h_est.squeeze() * mask_g_r
        + RG_v_est.squeeze() * mask_g_b
    )

    # Blue Channel
    B_final = (
        img * mask_b
        + RB_est.squeeze() * mask_r
        + RG_h_est.squeeze() * mask_g_b
        + RG_v_est.squeeze() * mask_g_r
    )

    output = torch.stack([R_final, G_final, B_final], dim=2)

    return output.to(raw_image.dtype)
