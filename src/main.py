from algorithms.raw.reader import read_raw_sensor_data
from algorithms.demosaicing._malvar_he_culter import malvar_he_cutler_demosaicing
from algorithms.white_balance._camera_white_balance import camera_white_balance
from algorithms.hue_saturation_map._hue_saturation_map import apply_hue_sat_map
from algorithms.gc.iec_gamma_correction import iec_gamma_correction
from algorithms.export._jpeg_export import export_jpeg
from algorithms.tools._lut_tools import read_hue_sat_lut_from_dcp
from algorithms.black_light_subtraction._black_light_subtraction import linearize_raw


def main(raw_img_path: str, dcp_file_path: str, output_img_path: str):
    raw_img, bayer_pattern, black_levels, white_level, wb_gains = read_raw_sensor_data(
        raw_img_path
    )
    raw_img = linearize_raw(raw_img, bayer_pattern, black_levels, white_level)
    demosaiced_img = malvar_he_cutler_demosaicing(raw_img)
    white_balanced_img = camera_white_balance(demosaiced_img, wb_gains)
    res = read_hue_sat_lut_from_dcp(dcp_file_path)
    if res is None:
        return
    (
        color_matrix_1,
        color_matrix_2,
        forward_matrix_1,
        forward_matrix_2,
        low_temp_lut,
        high_temp_lut,
        calib_illum_1,
        calib_illum_2,
    ) = res
    hue_sat_corrected_img = apply_hue_sat_map(
        white_balanced_img,
        wb_gains,
        color_matrix_1,
        color_matrix_2,
        forward_matrix_1,
        forward_matrix_2,
        low_temp_lut,
        high_temp_lut,
        calib_illum_1,
        calib_illum_2,
    )
    gamma_corrected_img = iec_gamma_correction(hue_sat_corrected_img)
    export_jpeg(gamma_corrected_img, output_img_path)
