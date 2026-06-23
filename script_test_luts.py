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
from algorithms.color_manipulation._lut_color_manipulation import linearRGB_to_adobeRGB1998
from algorithms.color_manipulation._lut_color_manipulation import adobeRGB1998_to_linearRGB
from algorithms.color_manipulation._lut_color_manipulation import apply_lut_grid_sample
from algorithms.color_manipulation._saturation_hsv import saturation_hsv

def script_test_one_lut(name_lut, path_lut):
    """
    Example
    name_lut = "NW-1"
    path_lut = "ON1 Nature & Wildlife LUTs/"
    """

    # Input file
    raw_file = PATH + "r01cbb7fdt.NEF"
    output_dir = "./outputs"
    output_file = "AAA-processed_image_exp_1-2_" + name_lut + ".jpg"

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
    #color_img = exp_img
    lut = load_cube_lut(PATH + "ON1_All_LUTs/" + path_lut + name_lut + ".cube")
    lut = lut = lut[...,[2,1,0]]
    adobe_img = linearRGB_to_adobeRGB1998(exp_img)
    color_img = apply_lut_grid_sample(adobe_img, lut)
    color_img = adobeRGB1998_to_linearRGB(color_img)
    #color_img = saturation_hsv(color_img, 1.4)


    # Step 6: Gamma Correction
    print("Step 6: Gamma Correction...")
    gamma_img = gamma_correction(color_img, gamma=2.2, alpha=1.0)

    # Step 7: Export
    print("Step 7: Exporting to JPEG...")
    Path(output_dir).mkdir(exist_ok=True)
    export_jpeg(gamma_img, f"{output_dir}/{output_file}", quality=85)

    print(f"✓ Saved to {output_dir}/{output_file}")

def script_test_ten_luts(path, prefixe):
    """
    Example: 
    path: "ON1 Cinematic LUTs/"
    prefixe: "Cinematic-"
    """
    for i in range(1,11):
        script_test_one_lut(prefixe + str(i), path)


def script_test_all_lut():
    path_luts = ["ON1 Black & White LUTs/",
                 "ON1 Cinematic LUTs/",
                 "ON1 Color Boost LUTs/",
                 "ON1 Landscape LUTs/",
                 "ON1 Lifestyle & Commercial LUTs/",
                 "ON1 Lutify.me LUTs/",
                 "ON1 Moody LUTs/",
                 "ON1 Nature & Wildlife LUTs",
                 "ON1 Portrait LUTs"]
    
    #script_test_ten_luts("ON1 Black & White LUTs/", "BW")
    script_test_ten_luts("ON1 Cinematic LUTs/", "Cinematic-")
    script_test_ten_luts("ON1 Landscape LUTs/", "Landscape")
    script_test_ten_luts("ON1 Lifestyle & Commercial LUTs/", "LC")
    script_test_ten_luts("ON1 Moody LUTs/", "Moody")
    script_test_ten_luts("ON1 Nature & Wildlife LUTs/", "NW-")
    script_test_ten_luts("ON1 Portrait LUTs/", "Portrait")

    # Color Boost
    path = "ON1 Color Boost LUTs/"
    script_test_one_lut("Aqua and Orange Dark", path)
    script_test_one_lut("Aqua", path)
    script_test_one_lut("Blues", path)
    script_test_one_lut("Earth Tone Boost", path)
    script_test_one_lut("Green_Blues", path)
    script_test_one_lut("Green_Yellow", path)
    script_test_one_lut("Oranges", path)
    script_test_one_lut("Purple", path)
    script_test_one_lut("Reds", path)
    script_test_one_lut("Reds_Oranges_Yellows", path)

    # Lutify.me
    path = "ON1 Lutify.me LUTs/"
    script_test_one_lut("2-Strip-Process", path)
    script_test_one_lut("Berlin Sky", path)
    script_test_one_lut("Chrome 01", path)
    script_test_one_lut("Classic Teal and Orange", path)
    script_test_one_lut("Fade to Green", path)
    script_test_one_lut("Film Print 01", path)
    script_test_one_lut("Film Print 02", path)
    script_test_one_lut("French Comedy", path)
    script_test_one_lut("Studio Skin Tone Shaper", path)
    script_test_one_lut("Vintage Chrome", path)

script_test_all_lut()