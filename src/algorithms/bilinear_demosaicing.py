import torch
import torch.nn.functional as F


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    # Avoid division by zero
    return torch.where(den == 0, torch.zeros_like(num), num / den)


def bilinear_demosaicing(rgb_image: torch.Tensor, dx: int = 0, dy: int = 0) -> torch.Tensor:
    """
    Vectorized Bilinear Demosaicing using Normalized Convolution.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must be a 3D tensor of shape (H, W, 3)")

    height, width, _ = rgb_image.shape
    if height == 0 or width == 0:
        return rgb_image

    # 1. Prepare Inputs
    img = rgb_image.permute(2, 0, 1).unsqueeze(0).float()

    R_sparse = img[:, 0:1, :, :]
    G_sparse = img[:, 1:2, :, :]
    B_sparse = img[:, 2:3, :, :]

    device = img.device
    dtype = img.dtype

    # 2. Define Kernels
    # k_cross: Sums 4 neighbors (Up, Down, Left, Right)
    k_cross = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=dtype, device=device).view(1, 1, 3, 3)
    # k_diag: Sums 4 diagonal neighbors
    k_diag = torch.tensor([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)

    # 3. Create Masks for Pixel Locations
    y = torch.arange(height, device=device).view(-1, 1)
    x = torch.arange(width, device=device).view(1, -1)

    # Broadcast (H,1) & (1,W) -> (H,W)
    is_red_row = (y % 2) == dy
    is_blue_row = (y % 2) != dy

    mask_r_loc = (is_red_row & ((x % 2) == dx)).float().view(1, 1, height, width)
    mask_b_loc = (is_blue_row & ((x % 2) != dx)).float().view(1, 1, height, width)
    mask_g_loc = (~(mask_r_loc.bool() | mask_b_loc.bool())).float()

    # 4. Normalized Convolution Helper
    def norm_conv(values, presence_mask, kernel):
        v_pad = F.pad(values, (1, 1, 1, 1), mode="constant", value=0)
        m_pad = F.pad(presence_mask, (1, 1, 1, 1), mode="constant", value=0)
        v_sum = F.conv2d(v_pad, kernel)
        m_sum = F.conv2d(m_pad, kernel)
        return _safe_div(v_sum, m_sum)

    # 5. Interpolation Step

    # --- Red Channel ---
    # At Blue locations: Interpolate diagonals
    r_at_b = norm_conv(R_sparse, mask_r_loc, k_diag)
    # At Green locations: Interpolate Cross (Up/Down/Left/Right)
    # This automatically handles both Horizontal (in Red rows) and Vertical (in Blue rows) neighbors.
    r_at_g = norm_conv(R_sparse, mask_r_loc, k_cross)

    R_out = R_sparse + r_at_b * mask_b_loc + r_at_g * mask_g_loc

    # --- Blue Channel ---
    b_at_r = norm_conv(B_sparse, mask_b_loc, k_diag)
    b_at_g = norm_conv(B_sparse, mask_b_loc, k_cross)

    B_out = B_sparse + b_at_r * mask_r_loc + b_at_g * mask_g_loc

    # --- Green Channel ---
    g_at_rb = norm_conv(G_sparse, mask_g_loc, k_cross)
    G_out = G_sparse + g_at_rb * (mask_r_loc + mask_b_loc)

    # 6. Final Assembly
    out = torch.cat([R_out, G_out, B_out], dim=1)
    return out.squeeze(0).permute(1, 2, 0)
