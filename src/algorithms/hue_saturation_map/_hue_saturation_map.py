import torch
import torch.nn.functional as F

from torch import Tensor 
from algorithms.tools._lut_tools import rgb_to_hsv, hsv_to_rgb

INTERPOLATION_ITERATIONS = 10
XYZ_TO_LP_MAT = torch.tensor([
    [ 1.3459, -0.2556, -0.0511],
    [-0.5446,  1.5082,  0.0205],
    [ 0.0000,  0.0000,  1.2118]
], dtype = torch.float32)

def _cct_from_calibration_illuminant(calib_illum: int) -> float:
    illuminant_to_kelvin = {
        1: 5500,  # Daylight
        3: 3200,  # Tungsten
        17: 2856, # Standard Illuminant A
        20: 5503, # D55
        21: 6504, # D65
        23: 5003  # D50
    }
    temperature = illuminant_to_kelvin.get(calib_illum)
    if temperature is None: 
        raise Exception # TODO : implement more formal exception errors raising (define isp_exceptions.py file and define them)
    
    return temperature

def _color_matrix_linear_interpolation(
    neutral: Tensor, 
    color_matrix_1: Tensor, 
    color_matrix_2: Tensor,
    calib_illuminant_1: int, 
    calib_illuminant_2: int
) -> float: 
    
    t: float = 0.5 
    factor: float = 1000000.0
    epicenter_x = 0.3320
    epicenter_y = 0.1858
    cct_1 = _cct_from_calibration_illuminant(calib_illuminant_1)
    cct_2 = _cct_from_calibration_illuminant(calib_illuminant_2)
    mired_1 = factor / cct_1
    mired_2 = factor / cct_2
    
    for _ in range(INTERPOLATION_ITERATIONS): 
        candidate = (1 - t) * color_matrix_1 + t * color_matrix_2
        xyz = torch.matmul(candidate, neutral)
        xyz_flat = xyz.flatten()
        s = xyz_flat.sum()
        xy = xyz_flat[:2] / s
        n = (xy[0] - epicenter_x) / (xy[1] - epicenter_y)
        cct_scene = - 449 * n ** 3 + 3525 * n ** 2 - 6823.3 * n  + 5520.33
        mired_scene = factor / cct_scene
        raw_t = (mired_scene - mired_1) / (mired_2 - mired_1)
        t = max(0.0, min(1.0, float(raw_t)))
    
    return t
    
def _rgb_to_xyz_matrix(
    color_matrix: Tensor,
    forward_matrix: Tensor,
    neutral: Tensor
) -> Tensor: 
    d = torch.matmul(color_matrix, neutral)
    D = torch.diag((1.0 / d).flatten())
    rgb_to_xyz = torch.matmul(forward_matrix, D)
    
    return rgb_to_xyz

def _xyz_to_linear_rgb(
    image_xyz: Tensor
) -> Tensor: 
    xyz_to_lp_mat = XYZ_TO_LP_MAT.to(image_xyz.device)
    return torch.matmul(image_xyz, xyz_to_lp_mat.t())

def _apply_hue_sat_map(image_hsv, lut_data):
    grid_h = image_hsv[..., 0] * 2 - 1   # (H, W)
    grid_s = image_hsv[..., 1] * 2 - 1   # (H, W)
    grid_v = image_hsv[..., 2] * 2 - 1   # (H, W)
    grid = torch.stack([grid_v, grid_s, grid_h], dim = -1)  # (H, W, 3)
    grid = grid.unsqueeze(0).unsqueeze(0) 
    lut_t = lut_data.permute(3, 0, 1, 2).unsqueeze(0)
    deltas = F.grid_sample(lut_t, grid, mode = 'bilinear', align_corners = True, padding_mode = 'border') 
    deltas = deltas.squeeze(0).squeeze(1).permute(1, 2, 0)
    delta_h, scale_s, scale_v = deltas[..., 0], deltas[..., 1], deltas[..., 2]
    h_prime = (image_hsv[..., 0] + delta_h) % 1.0
    s_prime = torch.clamp(image_hsv[..., 1] * scale_s, 0.0, 1.0)
    v_prime = torch.clamp(image_hsv[..., 2] * scale_v, 0.0, 1.0)
    
    return torch.stack([h_prime, s_prime, v_prime], dim = -1)

def _define_neutral_from_gains(wb_gains: Tensor) -> Tensor:
    if wb_gains.shape[0] == 4:
        r, g1, _, b = wb_gains
    else:
        r, g, b = wb_gains
        g1 = g
    neutral = torch.stack([1.0 / r, 1.0 / g1, 1.0 / b])
    return neutral.view(3, 1)

def apply_hue_sat_map(
    image_rgb: Tensor, 
    wb_gains: Tensor, 
    color_matrix_1: Tensor, 
    color_matrix_2: Tensor, 
    forward_matrix_1: Tensor,
    forward_matrix_2: Tensor,
    low_temp_lut: Tensor,
    high_temp_lut: Tensor,
    calib_illum_1: int, 
    calib_illum_2: int
) -> Tensor: 
    device = image_rgb.device
    neutral = _define_neutral_from_gains(wb_gains).to(device)
    t = _color_matrix_linear_interpolation(neutral, color_matrix_1, color_matrix_2, calib_illum_1, calib_illum_2)
    color_matrix = t * color_matrix_1 + (1 - t) * color_matrix_2
    forward_matrix = t * forward_matrix_1 + (1 - t) * forward_matrix_2
    p_rgb_to_xyz = _rgb_to_xyz_matrix(color_matrix, forward_matrix, neutral)
    image_xyz = torch.matmul(image_rgb, p_rgb_to_xyz.t())
    image_lp = _xyz_to_linear_rgb(image_xyz)
    image_hsv = rgb_to_hsv(image_lp)
    active_lut = t * low_temp_lut + (1 - t) * high_temp_lut
    corrected_hsv = _apply_hue_sat_map(image_hsv, active_lut.to(device))
    corrected_rgb = hsv_to_rgb(corrected_hsv)
    lp_to_xyz_mat = torch.linalg.inv(XYZ_TO_LP_MAT.to(device))
    image_xyz_out = torch.matmul(corrected_rgb, lp_to_xyz_mat.t())
    
    return image_xyz_out