import torch
from pathlib import Path

# Add source to path if needed
PATH = '/mnt/c/Users/charl/git/'

import sys
sys.path.insert(0, PATH + 'image-processing/src')

# Imports for each module
from algorithms.raw.reader import read_raw_sensor_data
from algorithms.black_light_subtraction.black_light_subtraction import linearize_raw
from algorithms.demosaicing._malvar_he_culter import malvar_he_cutler_demosaicing
from algorithms.white_balance._gray_world import gw
from algorithms.exposure_compensation._exposure_compensation import exposure_compensation
from algorithms.gamma_correction._gamma_correction import gamma_correction
from algorithms.export._jpeg_export import export_jpeg

from algorithms.color_manipulation._lut_color_manipulation import load_cube_lut
from algorithms.color_manipulation._lut_color_manipulation import apply_lut_grid_sample
from algorithms.color_manipulation._saturation_hsv import saturation_hsv
from algorithms.color_manipulation._temperature_simple import temperature_simple
from algorithms.color_manipulation._temperature_tanner_helland import temperature_tanner_helland
from algorithms.color_manipulation._contrast_linear_global import contrast_linear_global

def script_test_one_contrast_factor(contrast_name, contrast_factor):
    """
    Example
    contrast_name : "0-3"
    contrast_factor : 0.3
    """

    # Input file
    raw_file = PATH + "r01cbb7fdt.NEF"
    output_dir = "./outputs"
    output_file = "temp-4100_contrast-linear-global-" + contrast_name + ".jpg"

    # Step 1: Read RAW
    print("Step 1: Reading RAW file...")
    raw_data, bayer_pattern, black_levels, white_level, wb_gains = read_raw_sensor_data(raw_file)
    black_levels = torch.tensor(black_levels, dtype=torch.float32)
    white_level = torch.tensor(white_level, dtype=torch.float32)

    # Create CFA pattern (assuming RGGB)
    H, W = raw_data.shape
    cfa_pattern = torch.zeros((H, W), dtype=torch.int32)
    cfa_pattern[0::2, 0::2] = 0  # Red
    cfa_pattern[0::2, 1::2] = 1  # Green
    cfa_pattern[1::2, 0::2] = 3  # Green
    cfa_pattern[1::2, 1::2] = 2  # Blue

    # Step 2: Linearize (Black Level Subtraction)
    print("Step 2: Linearizing RAW...")
    bayer_img = linearize_raw(raw_data, cfa_pattern, black_levels, white_level)

    # Step 3: Demosaicing
    print("Step 3: Demosaicing (Malvar-He-Cutler)...")
    rgb_img = malvar_he_cutler_demosaicing(bayer_img, dx=0, dy=0)

    # Step 4: White Balance
    print("Step 4: White Balance (Gray World)...")
    wb_img = gw(rgb_img)

    # Step 5: Exposure Compensation (optional)
    print("Step 5: Exposure Compensation...")
    exp_img = exposure_compensation(wb_img, ev_compensation=1.2)  # +0.5 stops

    print("Step 6: Color Manipulation...")
    color_img = exp_img
    #color_img = saturation_hsv(color_img, 1.4)
    #color_img = temperature_simple(color_img, 10)
    color_img = temperature_tanner_helland(color_img, 4100)
    color_img = contrast_linear_global(color_img, contrast_factor)


    # Step 6: Gamma Correction
    print("Step 6: Gamma Correction...")
    gamma_img = gamma_correction(color_img, gamma=2.2, alpha=1.0)

    # Step 7: Export
    print("Step 7: Exporting to JPEG...")
    Path(output_dir).mkdir(exist_ok=True)
    export_jpeg(gamma_img, f"{output_dir}/{output_file}", quality=85)

    print(f"✓ Saved to {output_dir}/{output_file}")

def script_test_several_contrast_factors():

    for i in range(1,7):
        print(i)
        script_test_one_contrast_factor(str(i), i)

script_test_several_contrast_factors()