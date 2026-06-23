# import os
import json
import logging

import torch

from algorithms.raw.reader import read_raw_sensor_data

import folder_paths  # type: ignore

logger = logging.getLogger(__name__)


class BatchReadRawSensorNode:
    @classmethod
    def INPUT_TYPES(cls):
        # input_dir = folder_paths.get_input_directory()
        # all_files = [
        #     f
        #     for f in os.listdir(input_dir)
        #     if os.path.isfile(os.path.join(input_dir, f))
        # ]
        # valid_extensions = (
        #     ".dng",
        #     ".cr2",
        #     ".cr3",
        #     ".arw",
        #     ".nef",
        #     ".raf",
        #     ".orf",
        #     ".rw2",
        #     ".srw",
        #     ".tiff",
        #     ".tif",
        # )
        # files = sorted(f for f in all_files if f.lower().endswith(valid_extensions))

        return {
            "required": {
                "images": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    CATEGORY = "image"
    SEARCH_ALIASES = [
        "batch load image",
        "burst loader",
        "hdr loader",
        "multi raw loader",
        "batch read raw",
    ]

    RETURN_TYPES = ("IMAGE", "PATTERN", "BLACK_LEVEL", "WHITE_LEVEL", "WB_GAIN")
    RETURN_NAMES = (
        "raw_imgs_batch",
        "cfa_patterns_batch",
        "black_levels_list",
        "white_levels_list",
        "wb_gains_list",
    )
    FUNCTION = "execute"

    @staticmethod
    def _parse_filenames(images: str) -> list[str]:
        filenames = [f.strip() for f in images.strip().split('\n') if f.strip()]
        if not isinstance(filenames, list) or not filenames:
            raise ValueError("No images selected for batch loading.")
        return filenames

    def execute(self, images):
        filenames = self._parse_filenames(images)

        raw_imgs, cfa_patterns, black_levels, white_levels, wb_gains = (
            [],
            [],
            [],
            [],
            [],
        )

        for filename in filenames:
            image_path = folder_paths.get_annotated_filepath(filename)
            try:
                raw_img, cfa_pattern, black_level, white_level, wb_gain = (
                    read_raw_sensor_data(image_path)
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read raw sensor data from '{filename}': {e}"
                ) from e

            raw_imgs.append(raw_img)
            cfa_patterns.append(cfa_pattern)
            black_levels.append(black_level)
            white_levels.append(white_level)
            wb_gains.append(wb_gain)

        try:
            raw_imgs = torch.stack(raw_imgs, dim=0).unsqueeze(-1)
            cfa_patterns = torch.stack(cfa_patterns, dim=0)
            black_levels = torch.stack(black_levels, dim=0)
            white_levels = torch.stack(white_levels, dim=0)
            wb_gains = torch.stack(wb_gains, dim=0)
        except RuntimeError as e:
            raise RuntimeError(
                "Could not stack batch — images likely have mismatched "
                f"resolution or sensor format: {e}"
            ) from e

        return (raw_imgs, cfa_patterns, black_levels, white_levels, wb_gains)

    @classmethod
    def IS_CHANGED(cls, images):
        try:
            filenames = cls._parse_filenames(images)
        except (ValueError, json.JSONDecodeError):
            return images  # let execute() raise the real error

        hasher = __import__("hashlib").sha256()
        for filename in filenames:
            image_path = folder_paths.get_annotated_filepath(filename)
            with open(image_path, "rb") as f:
                hasher.update(f.read())
        return hasher.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, images):
        try:
            filenames = cls._parse_filenames(images)
        except json.JSONDecodeError:
            return "images widget value is not valid JSON"
        except ValueError as e:
            return str(e)

        missing = [
            f for f in filenames if not folder_paths.exists_annotated_filepath(f)
        ]
        if missing:
            return f"Missing file(s): {', '.join(missing)}"
        return True


NODE_CLASS_MAPPINGS = {
    "BatchReadRawSensorNode": BatchReadRawSensorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchReadRawSensorNode": "Batch Read RAW Sensor (Burst)",
}
