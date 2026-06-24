from algorithms.raw.reader import read_raw_sensor_data

import os
import hashlib
import folder_paths  # type: ignore


class ReadRawSensorNode:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        all_files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        valid_extensions = (
            ".dng",
            ".cr2",
            ".cr3",
            ".arw",
            ".nef",
            ".raf",
            ".orf",
            ".rw2",
            ".srw",
            ".tiff",
            ".tif",
        )
        files = sorted(f for f in all_files if f.lower().endswith(valid_extensions))

        return {
            "required": {"image": (files, {})},
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
    WEB_DIRECTORY = "./js"

    def execute(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        raw_img, cfa_pattern, black_levels, white_level, wb_gains = (
            read_raw_sensor_data(image_path)
        )
        raw_img = raw_img.unsqueeze(0).unsqueeze(-1)
        cfa_pattern = cfa_pattern.unsqueeze(0)

        return (raw_img, cfa_pattern, black_levels, white_level, wb_gains)

    @classmethod
    def IS_CHANGED(cls, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


NODE_CLASS_MAPPINGS = {
    "ReadRawSensorNode": ReadRawSensorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReadRawSensorNode": "Read RAW Sensor",
}
