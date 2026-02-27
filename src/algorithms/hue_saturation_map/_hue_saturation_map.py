import colour
import torch

from torch import Tensor 
from algorithms.tools._lut_tools import rgb_to_hsv

def _mired_value_from_calibration_illuminant(calib_illum: int) -> float:
    factor = 1_000_00
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
    
    return factor / temperature

def _get_current_mired_value(wb_gains: Tensor, daylight_color_matrix: Tensor) -> float: 
    neutral_color = 1 / wb_gains
    xyz_proj = torch.matmul(daylight_color_matrix, neutral_color)
    x_chromaticity = float(xyz_proj[0] / (xyz_proj[0] + xyz_proj[1] + xyz_proj[2]))
    y_chromaticity = float(xyz_proj[1] / (xyz_proj[0] + xyz_proj[1] + xyz_proj[2]))
    n = (x_chromaticity - 0.3320) / (0.1858 - y_chromaticity)
    mc_campy_temp_approx = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33
    current_mired = 1 / mc_campy_temp_approx
    
    return current_mired

def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """
    Expects hsv in range [0, 1]
    Returns rgb in range [0, 1]
    """
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    h6 = h * 6.0
    i = torch.floor(h6)
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i = i.long() % 6
    
    # We expand dims to allow broadcasting across the color channels
    v = v.unsqueeze(-1)
    t = t.unsqueeze(-1)
    p = p.unsqueeze(-1)
    q = q.unsqueeze(-1)

    # Use torch.stack to create the 6 possible color cases
    # Each case represents a sector of the color hexagon
    rgb = torch.stack([
        torch.cat([v, t, p], dim=-1), # Case 0
        torch.cat([q, v, p], dim=-1), # Case 1
        torch.cat([p, v, t], dim=-1), # Case 2
        torch.cat([p, q, v], dim=-1), # Case 3
        torch.cat([t, p, v], dim=-1), # Case 4
        torch.cat([v, p, q], dim=-1), # Case 5
    ], dim=-2) # Stack on a temporary dimension

    # Select the correct case based on the 'i' index
    # i is [H, W], so we need to gather the correct RGB triplet
    mask = i.unsqueeze(-1).expand_as(hsv).unsqueeze(-2)
    # Resulting shape [H, W, 1, 3]
    result = torch.gather(rgb, -2, mask).squeeze(-2)
    
    return result.clamp(0.0, 1.0)
    
def apply_hue_saturation_map(
        rgb_image: Tensor, 
        wb_gains: Tensor, 
        low_temp_lut: Tensor, 
        high_temp_lut: Tensor, 
        indoor_color_matrix: Tensor, 
        daylight_color_matrix: Tensor,
        calib_illum_1: int,
        calib_iluum_2: int
        ) -> Tensor: 
    """
    Applies the camera profile `HueSatMap` LUTs depending on white balance gains.
    
    Arguments.
    - `rgb_image` : image to apply hue/saturation map, requires image of shape [H, W, 3].
    - `wb_gains` : white balance gains.
    - `low_temp_lut` : low temperature LUT, referred as `ProfileHueSatMap1` in DCP files.
    - `high_temp_lut` : high temperature LUT, referred as `ProfileHueSatMap2` in DCP files.
    
    Returns. 
    - RGB image of shape [H, W, 3]
    """
    averaged_wb_gains = torch.tensor(
        [wb_gains[0], (wb_gains[1] + wb_gains[2]) / 2, wb_gains[3]], 
        device = wb_gains.device, 
        dtype=wb_gains.dtype
    )
    current_mired = _get_current_mired_value(averaged_wb_gains, daylight_color_matrix)
    mired_1 = _mired_value_from_calibration_illuminant(calib_illum_1)
    mired_2 = _mired_value_from_calibration_illuminant(calib_iluum_2)
    t = (current_mired - mired_2) / (mired_1 - mired_2) # interpolation factor
    interpolated_color_matrix = t * indoor_color_matrix + (1 - t) * daylight_color_matrix
    xyz_image = torch.matmul(rgb_image, interpolated_color_matrix.T)
    xyz_to_rgb_mat = torch.tensor(
        colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB, 
        dtype = rgb_image.dtype,
        device = rgb_image.device
    )
    linear_rgb_image = torch.matmul(xyz_image, xyz_to_rgb_mat.T)
    linear_rgb_image = torch.clamp(linear_rgb_image, 0.0, 1.0).squeeze(0)
    hsv_image = rgb_to_hsv(linear_rgb_image)
    final_lut = t * low_temp_lut + (1 - t) * high_temp_lut
    lut_input = final_lut.permute(3, 0, 1, 2).unsqueeze(0)
    grid_coords = torch.empty_like(hsv_image)
    grid_coords[..., 0] = hsv_image[..., 2] * 2 - 1 # Hue
    grid_coords[..., 1] = hsv_image[..., 1] * 2 - 1 # Saturation
    grid_coords[..., 2] = hsv_image[..., 0] * 2 - 1 
    grid_input = grid_coords.unsqueeze(0).unsqueeze(1) 
    offsets = torch.nn.functional.grid_sample(
        lut_input, 
        grid_input, 
        mode = 'bilinear', 
        padding_mode = 'border', 
        align_corners = True
    )
    offsets = offsets.squeeze(2).permute(0, 2, 3, 1).squeeze(0) # [H, W, 3, *]
    h_deg = hsv_image[..., 0] * 360.0
    h_new = torch.remainder(h_deg + offsets[..., 0], 360.0) / 360.0
    s_new = torch.clamp(hsv_image[..., 1] * offsets[..., 1], 0.0, 1.0)
    v_new = torch.clamp(hsv_image[..., 2] * offsets[..., 2], 0.0, 1.0)
    hsv_corrected = torch.stack([h_new, s_new, v_new], dim = -1)
    print(hsv_corrected.shape)
    corrected_rgb = hsv_to_rgb(hsv_corrected)
    
    return corrected_rgb
    
    
    
    