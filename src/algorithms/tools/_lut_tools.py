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
    Expects input shape [H, W, 3], range [0, 1].
    Returns HSV with H, S, V in [0, 1].
    """
    eps = 1e-10
    r, g, b = rgb_image.unbind(-1)

    max_c, max_idx = torch.max(rgb_image, dim=-1)  # max_idx : 0=R, 1=G, 2=B
    min_c = torch.min(rgb_image, dim=-1).values
    delta = max_c - min_c

    # --- Hue : calcul entièrement vectorisé avec torch.where ---
    delta_safe = delta.clamp(min=eps)

    h_r = ((g - b) / delta_safe) % 6
    h_g = ((b - r) / delta_safe) + 2
    h_b = ((r - g) / delta_safe) + 4

    # Sélection selon quel canal est max
    h = torch.where(max_idx == 0, h_r, torch.where(max_idx == 1, h_g, h_b)) / 6.0

    # Mise à zéro là où delta ~ 0 (pixel achromatique)
    h = torch.where(delta < eps, torch.zeros_like(h), h)

    # --- Saturation ---
    s = torch.where(max_c > eps, delta / max_c.clamp(min=eps), torch.zeros_like(max_c))

    out = torch.empty_like(rgb_image)
    out[..., 0] = h
    out[..., 1] = s
    out[..., 2] = max_c
    
    return out


def hsv_to_rgb(hsv_image: Tensor) -> Tensor:
    h, s, v = hsv_image.unbind(-1)

    h6 = h * 6.0
    i = h6.long() % 6
    f = h6 - h6.floor()

    c = v * s
    m = v - c
    p = m + c * (1.0 - f)
    q = m + c * f

    r = torch.where(i == 0, v,  torch.where(i == 1, p,
        torch.where(i == 2, m,  torch.where(i == 3, m,
        torch.where(i == 4, q,  v)))))

    g = torch.where(i == 0, q,  torch.where(i == 1, v,
        torch.where(i == 2, v,  torch.where(i == 3, p,
        torch.where(i == 4, m,  m)))))

    b = torch.where(i == 0, m,  torch.where(i == 1, m,
        torch.where(i == 2, q,  torch.where(i == 3, v,
        torch.where(i == 4, v,  p)))))

    return torch.stack([r, g, b], dim=-1).clamp_(0.0, 1.0)
