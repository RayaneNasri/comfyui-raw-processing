import torch
from scipy.ndimage import gaussian_filter

import logging

logger = logging.getLogger(__name__)


def convert8bit_(x):
    scaled = torch.clamp(x, 0.0, 1.0) * 255.0 + 0.5
    return scaled.to(torch.uint8)


def convert16bit_(x):
    max_val = 2**16 - 1
    scaled = torch.clamp(x, 0.0, 1.0) * max_val + 0.5
    return scaled.to(torch.uint32)


def mean4(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    if not a.is_floating_point():
        sum_val = (
            a.to(torch.float32)
            + b.to(torch.float32)
            + c.to(torch.float32)
            + d.to(torch.float32)
        )
        return torch.bitwise_right_shift(sum_val, 2)

    return (a + b + c + d) / 4.0


def downsample(image: torch.Tensor, kernel: str = "gaussian", factor: int = 2):
    if factor == 1:
        return image

    if kernel is None:
        filtered_image = image
    elif kernel == "gaussian":
        image_np = image.detach().cpu().numpy()
        filtered_image = torch.tensor(
            gaussian_filter(
                image_np,
                sigma=factor * 0.5,
                order=0,
                output=None,
                mode="reflect",
            )
        ).to(image.device)
    else:
        return mean4(
            image[0::2, 0::2], image[1::2, 0::2], image[0::2, 1::2], image[1::2, 1::2]
        )

    h, w = filtered_image.shape
    h2, w2 = h // factor, w // factor
    out = filtered_image[: h2 * factor, : w2 * factor]

    if not image.is_floating_point():
        out = torch.round(out.to(torch.float32)).to(image.dtype)
    return out


def get_aligned_tiles(
    image: torch.Tensor, tile_size: int, motion_vectors: torch.Tensor
) -> torch.Tensor:

    device = image.device
    H, W = image.shape
    step = tile_size // 2

    h, w = H // step - 1, W // step - 1

    hm, wm, _ = motion_vectors.shape

    image_tiles = image.unfold(0, tile_size, 1).unfold(1, tile_size, 1)

    base_I, base_J = (
        (torch.arange(h, device=device) * step).view(hm, 1),
        (torch.arange(w, device=device) * step).view(1, wm),
    )

    indIm, indJm = (
        torch.round(base_I + motion_vectors[..., 0]).long(),
        torch.round(base_J + motion_vectors[..., 1]).long(),
    )

    indIm = torch.clamp(indIm, 0, H - tile_size)
    indJm = torch.clamp(indJm, 0, W - tile_size)

    indI = (torch.arange(h, device=device) * step).view(h, 1).expand(h, w).clone()
    indJ = (torch.arange(w, device=device) * step).view(1, w).expand(h, w).clone()

    indI[:hm, :wm] = indIm
    indJ[:hm, :wm] = indJm

    aligned_tiles = image_tiles[indI, indJ]

    return aligned_tiles


def compute_tiles_distance_l1(
    ref_tiles: torch.Tensor, alt_tiles: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:

    m, n, ts, _ = ref_tiles.shape
    h, w, _ = offsets.shape
    step = ts // 2
    device = ref_tiles.device

    base_i = (torch.arange(h, device=device) * step).view(h, 1)
    base_j = (torch.arange(w, device=device) * step).view(1, w)

    ri, rj = base_i.expand(h, w), base_j.expand(h, w)
    off_i, off_j = torch.round(offsets[:, :, 0]), torch.round(offsets[:, :, 1])

    di, dj = ri + off_i, rj + off_j
    di, dj = torch.clamp(di, 0, m - 1).long(), torch.clamp(dj, 0, n - 1).long()
    # ri/rj are pure base indices from the offsets grid; when the upsampled
    # alignment grid is larger than the ref tile grid (h > m or w > n) they
    # exceed the tile array bounds.  Clamp them the same way as di/dj.
    ri, rj = torch.clamp(ri, 0, m - 1).long(), torch.clamp(rj, 0, n - 1).long()

    ref_blocks = ref_tiles[ri, rj]
    alt_blocks = alt_tiles[di, dj]

    l1_distance = torch.sum(torch.abs(ref_blocks - alt_blocks), dim=(-1, -2))

    return l1_distance


def compute_l1_distance(win: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    p = ref.shape[1]
    patches = win.unfold(1, p, 1).unfold(2, p, 1)
    ref_broadcastable = ref.unsqueeze(1).unsqueeze(2)
    l1_distance = torch.sum(torch.abs(patches - ref_broadcastable), dim=(-1, -2))

    return l1_distance


def compute_l2_distance(win: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    p = ref.shape[1]
    patches = win.unfold(1, p, 1).unfold(2, p, 1)
    ref_broadcastable = ref.unsqueeze(1).unsqueeze(2)
    l2_distance = torch.sum((patches - ref_broadcastable) ** 2, dim=(-1, -2))

    return l2_distance


def compute_distance(
    ref_patch: torch.Tensor, search_area: torch.Tensor, distance_type: str = "L2"
) -> torch.Tensor:
    h, w, sP, sP2 = ref_patch.shape
    hs, ws, sW, sW2 = search_area.shape

    assert sP == sP2, "Reference patches must be square."
    assert sW == sW2, "Search areas must be square."
    assert h == hs and w == ws, "Batch dimensions (h, w) must match."

    win = search_area.reshape(h * w, sW, sW)
    ref = ref_patch.reshape(h * w, sP, sP)

    sT = sW - sP + 1

    if distance_type == "L1":
        dst = compute_l1_distance(win, ref)
    elif distance_type == "L2":
        dst = compute_l2_distance(win, ref)
    else:
        raise ValueError(f"Unknown distance metric '{distance_type}'. Aborting.")

    return dst.reshape(h * w, sT**2)


def sub_pixel_minimum_(
    array_distances: torch.Tensor, array_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the sub-pixel offset to the distance minimum by fitting a
    bivariate quadratic function to the 3x3 neighborhood.

    Args:
        array_distances: Tensor of shape (m, n) containing distances.
        array_indices: Tensor of shape (m,) containing corresponding indices of minima.

    Returns:
        resI, resJ: Tuple of two (m,) tensors containing the sub-pixel offsets.
    """
    device = array_distances.device
    m, n = array_distances.shape

    fA11 = torch.tensor(
        [0.25, -0.5, 0.25, 0.5, -1.0, 0.5, 0.25, -0.5, 0.25], device=device
    )
    fA22 = torch.tensor(
        [0.25, 0.5, 0.25, -0.5, -1.0, -0.5, 0.25, 0.5, 0.25], device=device
    )
    fA12 = torch.tensor(
        [0.25, 0.0, -0.25, 0.0, 0.0, 0.0, -0.25, 0.0, 0.25], device=device
    )
    fb1 = torch.tensor(
        [-0.125, 0.0, 0.125, -0.25, 0.0, 0.25, -0.125, 0.0, 0.125], device=device
    )
    fb2 = torch.tensor(
        [-0.125, -0.25, -0.125, 0.0, 0.0, 0.0, 0.125, 0.25, 0.125], device=device
    )

    offsets = torch.arange(-4, 5, device=device)
    col_indices = array_indices.unsqueeze(1) + offsets

    col_indices = torch.clamp(col_indices, 0, n - 1)

    d = torch.gather(array_distances, 1, col_indices)

    A11 = torch.matmul(d, fA11)
    A12 = torch.matmul(d, fA12)
    A22 = torch.matmul(d, fA22)
    b1 = torch.matmul(d, fb1)
    b2 = torch.matmul(d, fb2)

    A11 = torch.clamp(A11, min=0.0)
    A22 = torch.clamp(A22, min=0.0)

    # If A11 * A22 - A12**2 < 0, set A12 to 0
    det_check = A11 * A22 - (A12**2)
    A12 = torch.where(det_check < 0, torch.zeros_like(A12), A12)

    # Recompute determinant
    det = A11 * A22 - (A12**2)

    # Compute sub-pixel offsets safely
    # Create a safe determinant tensor to prevent division by zero (NaNs break gradients/logic)
    det_safe = torch.where(det == 0, torch.ones_like(det), det)

    osvI = -(A11 * b2 - A12 * b1) / det_safe
    osvJ = -(A22 * b1 - A12 * b2) / det_safe

    # Apply masking thresholds
    # Ensure norm < 1 and that the original determinant wasn't 0
    nrm = torch.sqrt(osvI**2 + osvJ**2)
    valid = (nrm < 1.0) & (det != 0.0)

    resI = torch.where(valid, osvI, torch.zeros_like(osvI))
    resJ = torch.where(valid, osvJ, torch.zeros_like(osvJ))

    return resI, resJ


def sub_pixel_minimum(arrDst: torch.Tensor, arrIdx: torch.Tensor) -> torch.Tensor:
    """
    Compute sub-pixel distance minima from 3x3 neighborhoods.
    """
    resI, resJ = sub_pixel_minimum_(arrDst, arrIdx)

    return torch.stack([resI, resJ], dim=1)


def compute_RMSE(image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    """Computes the Root Mean Square Error between two images"""
    assert image1.shape == image2.shape, "images have different sizes"

    diff = image1.to(torch.float32) - image2.to(torch.float32)
    return torch.sqrt(torch.mean(diff**2))


def compute_PSNR(image: torch.Tensor, noisyImage: torch.Tensor) -> float:
    """Computes the Peak Signal-to-Noise Ratio"""
    if image.shape == noisyImage.shape:
        if image.is_floating_point():
            assert image.min() >= 0.0 and image.max() <= 1.0, (
                "not a float image between 0 and 1"
            )
            maxValue = 1.0
        else:
            maxValue = float(torch.iinfo(image.dtype).max)

        mse = torch.mean((image.to(torch.float32) - noisyImage.to(torch.float32)) ** 2)
        if mse == 0:
            return float("inf")
        return 10 * torch.log10(maxValue**2 / mse).item()
    else:
        raise ValueError("images have different sizes")


def get_tiles(image: torch.Tensor, window_size: int, step: int = 1) -> torch.Tensor:
    """
    Create a sliding window view over the input tensor.
    Replaces the complex stride-trick logic with native unfolding.

    Args:
        image: 2D Tensor (H, W)
        window_size: int, size of square window

        step: int, overlap step
    """
    tiles = image.unfold(0, window_size, step).unfold(1, window_size, step)
    return tiles
