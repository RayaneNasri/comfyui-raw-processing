from algorithms.raw.reader import read_raw_sensor_data

import os
import folder_paths # type : ignore

class ReadRawSensorNode:

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"
    SEARCH_ALIASES = [
        "load image",
        "open image",
        "import image",
        "image input",
        "upload image",
        "read image",
        "image loader",
        "read raw",
    ]

    RETURN_TYPES = ("IMAGE", "PATTERN", "BLACK_LEVEL", "WHITE_LEVEL", "WB_GAIN")
    RETURN_NAMES = (
        "raw_img",
        "cfa_pattern",
        "black_levels",
        "white_level",
        "wb_gains",
    )
    FUNCTION = "execute"

    def execute(self, image_path):
        raw_img, cfa_pattern, black_levels, white_level, wb_gains = (
            read_raw_sensor_data(image_path)
        )
        raw_img = raw_img.unsqueeze(0).unsqueeze(-1)
        cfa_pattern = cfa_pattern.unsqueeze(0)

        return (raw_img, cfa_pattern, black_levels, white_level, wb_gains)


NODE_CLASS_MAPPINGS = {
    "ReadRawSensorNode": ReadRawSensorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReadRawSensorNode": "Read RAW Sensor",
}