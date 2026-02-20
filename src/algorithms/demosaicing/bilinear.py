import torch
import torch.nn.functional as F


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return torch.where(den == 0, torch.zeros_like(num), num / den)


def mono_to_rgb(
    normalized_image: torch.Tensor, bayer_pattern: torch.Tensor
) -> torch.Tensor:
    """
    Convert a single-channel Bayer image into a sparse RGB image.
    """
    height, width = normalized_image.shape
    rgb_image = torch.zeros((height, width, 3), dtype=normalized_image.dtype)

    channel_indices = torch.clone(bayer_pattern)
    channel_indices[channel_indices == 3] = 1

    rows = torch.arange(height)[:, None]
    cols = torch.arange(width)
    rgb_image[rows, cols, channel_indices] = normalized_image
    return rgb_image


def bilinear_demosaicing(
    rgb_image: torch.Tensor, dx: int = 0, dy: int = 0
) -> torch.Tensor:
    """
    Vectorized Bilinear Demosaicing using normalized convolution.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must be a 3D tensor of shape (H, W, 3)")

    height, width, _ = rgb_image.shape
    if height == 0 or width == 0:
        return rgb_image

    img = rgb_image.permute(2, 0, 1).unsqueeze(0).float()

    r_sparse = img[:, 0:1, :, :]
    g_sparse = img[:, 1:2, :, :]
    b_sparse = img[:, 2:3, :, :]

    device = img.device
    dtype = img.dtype

    k_cross = torch.tensor(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=dtype, device=device
    ).view(1, 1, 3, 3)
    k_diag = torch.tensor(
        [[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=dtype, device=device
    ).view(1, 1, 3, 3)

    y = torch.arange(height, device=device).view(-1, 1)
    x = torch.arange(width, device=device).view(1, -1)

    is_red_row = (y % 2) == dy
    is_blue_row = (y % 2) != dy

    mask_r_loc = (is_red_row & ((x % 2) == dx)).float().view(1, 1, height, width)
    mask_b_loc = (is_blue_row & ((x % 2) != dx)).float().view(1, 1, height, width)
    mask_g_loc = (~(mask_r_loc.bool() | mask_b_loc.bool())).float()

    def norm_conv(values, presence_mask, kernel):
        v_pad = F.pad(values, (1, 1, 1, 1), mode="constant", value=0)
        m_pad = F.pad(presence_mask, (1, 1, 1, 1), mode="constant", value=0)
        v_sum = F.conv2d(v_pad, kernel)
        m_sum = F.conv2d(m_pad, kernel)
        return _safe_div(v_sum, m_sum)

    r_at_b = norm_conv(r_sparse, mask_r_loc, k_diag)
    r_at_g = norm_conv(r_sparse, mask_r_loc, k_cross)
    r_out = r_sparse + r_at_b * mask_b_loc + r_at_g * mask_g_loc

    b_at_r = norm_conv(b_sparse, mask_b_loc, k_diag)
    b_at_g = norm_conv(b_sparse, mask_b_loc, k_cross)
    b_out = b_sparse + b_at_r * mask_r_loc + b_at_g * mask_g_loc

    g_at_rb = norm_conv(g_sparse, mask_g_loc, k_cross)
    g_out = g_sparse + g_at_rb * (mask_r_loc + mask_b_loc)

    out = torch.cat([r_out, g_out, b_out], dim=1)
    return out.squeeze(0).permute(1, 2, 0)
