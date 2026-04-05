import torch
import tifffile
import math

from tifffile import TiffPage
from torch import Tensor

HUE_SAT_MAP_DIMS_TAG = 50937
HUE_SAT_MAP_DATA_1_TAG = 50938
HUE_SAT_MAP_DATA_2_TAG = 50939
CALIBRATION_ILLUMINANT_1_TAG = 50778
CALIBRATION_ILLUMINANT_2_TAG = 50779
INDOOR_COLOR_MATRIX_TAG = 50721
DAYLIGHT_COLOR_MATRIX_TAG = 50722
FORWARD_MATRIX_1_TAG = 50964
FORWARD_MATRIX_2_TAG = 50965

NORMALIZATION_HSV_SCALE = torch.tensor([2 * math.pi, 1.0, 1.0]).view(3, 1, 1)


def _flatten_num_denum_color_matrix(color_matrix: Tensor) -> Tensor:
    """
    Flattens a 18 sized numerator/denumerator color matrix tensor into a 3x3 tensor
    """
    num = color_matrix[0::2]
    denum = color_matrix[1::2]
    flattened = num / denum

    return flattened.reshape(3, 3)


def read_hue_sat_lut_from_dcp(
    dcp_path: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int] | None:
    try:
        with tifffile.TiffFile(dcp_path) as dcp_file:
            page = dcp_file.pages[0]
            if isinstance(page, TiffPage):
                tags = page.tags
                for tag in tags:
                    print(tag)
                dims = tags[HUE_SAT_MAP_DIMS_TAG].value
                h, s, v = dims
                low_temp_lut = Tensor(tags[HUE_SAT_MAP_DATA_1_TAG].value).reshape(
                    h, s, v, 3
                )
                high_temp_lut = Tensor(tags[HUE_SAT_MAP_DATA_2_TAG].value).reshape(
                    h, s, v, 3
                )
                indoor_color_matrix = Tensor(tags[INDOOR_COLOR_MATRIX_TAG].value)
                daylight_color_matrix = Tensor(tags[DAYLIGHT_COLOR_MATRIX_TAG].value)
                calib_illum_1 = int(tags[CALIBRATION_ILLUMINANT_1_TAG].value)
                calib_illum_2 = int(tags[CALIBRATION_ILLUMINANT_2_TAG].value)
                forward_matrix_1 = _flatten_num_denum_color_matrix(
                    Tensor(tags[FORWARD_MATRIX_1_TAG].value)
                )
                forward_matrix_2 = _flatten_num_denum_color_matrix(
                    Tensor(tags[FORWARD_MATRIX_2_TAG].value)
                )

                return (
                    low_temp_lut,
                    high_temp_lut,
                    _flatten_num_denum_color_matrix(indoor_color_matrix),
                    _flatten_num_denum_color_matrix(daylight_color_matrix),
                    forward_matrix_1,
                    forward_matrix_2,
                    calib_illum_1,
                    calib_illum_2,
                )
    except Exception as e:
        print(e)  # TODO : implement proper exceptions handling
        return None


def rgb_to_hsv(rgb_image: Tensor) -> Tensor:
    """
    Transforms an RGB image into HSV space.
    Expects input shape [H, W, 3] and range [0, 1].
    Returns HSV image with H in [0, 1], S in [0, 1], V in [0, 1].
    """
    # rgb_image: [H, W, 3]
    r, g, b = rgb_image.unbind(-1)

    max_c, _ = torch.max(rgb_image, dim=-1)
    min_c, _ = torch.min(rgb_image, dim=-1)
    delta = max_c - min_c

    # --- Teinte (H) ---
    # On utilise eps pour éviter la division par zéro
    eps = 1e-10
    h = torch.zeros_like(max_c)

    mask_r = (max_c == r) & (delta > eps)
    mask_g = (max_c == g) & (delta > eps)
    mask_b = (max_c == b) & (delta > eps)

    h[mask_r] = ((g[mask_r] - b[mask_r]) / (delta[mask_r] + eps)) % 6
    h[mask_g] = ((b[mask_g] - r[mask_g]) / (delta[mask_g] + eps)) + 2
    h[mask_b] = ((r[mask_b] - g[mask_b]) / (delta[mask_b] + eps)) + 4

    h = h / 6.0  # Normalisation entre 0 et 1

    # --- Saturation (S) ---
    s = torch.zeros_like(max_c)
    s[max_c > eps] = delta[max_c > eps] / (max_c[max_c > eps] + eps)

    # --- Valeur (V) ---
    v = max_c

    return torch.stack([h, s, v], dim=-1)


def hsv_to_rgb(hsv_image: Tensor) -> Tensor:
    """
    Transforms an HSV image into RGB space.
    Expects input shape [H, W, 3] and range [0, 1].
    """
    h, s, v = hsv_image.unbind(-1)

    h_six = h * 6.0
    c = v * s  # Chroma
    x = c * (1 - torch.abs(h_six % 2 - 1))
    m = v - c

    # On initialise les canaux
    r1 = torch.zeros_like(h)
    g1 = torch.zeros_like(h)
    b1 = torch.zeros_like(h)

    # Définition des sextants
    mask0 = (h_six >= 0) & (h_six < 1)
    mask1 = (h_six >= 1) & (h_six < 2)
    mask2 = (h_six >= 2) & (h_six < 3)
    mask3 = (h_six >= 3) & (h_six < 4)
    mask4 = (h_six >= 4) & (h_six < 5)
    mask5 = (h_six >= 5) & (h_six <= 6)

    r1[mask0], g1[mask0], b1[mask0] = c[mask0], x[mask0], 0
    r1[mask1], g1[mask1], b1[mask1] = x[mask1], c[mask1], 0
    r1[mask2], g1[mask2], b1[mask2] = 0, c[mask2], x[mask2]
    r1[mask3], g1[mask3], b1[mask3] = 0, x[mask3], c[mask3]
    r1[mask4], g1[mask4], b1[mask4] = x[mask4], 0, c[mask4]
    r1[mask5], g1[mask5], b1[mask5] = c[mask5], 0, x[mask5]

    rgb = torch.stack([r1 + m, g1 + m, b1 + m], dim=-1)
    return torch.clamp(rgb, 0.0, 1.0)
