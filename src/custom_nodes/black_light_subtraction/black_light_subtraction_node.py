from algorithms.black_light_subtraction._black_light_subtraction import linearize_raw


class LinearizeRawNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_img": ("IMAGE",),
                "cfa_pattern": ("PATTERN",),
                "black_levels": ("BLACK_LEVEL",),
                "white_level": ("WHITE_LEVEL",),
            }
        }

    CATEGORY = "image/processing/black-light-substraction"
    SEARCH_ALIASES = [
        "black level subtraction",
        "raw linearization",
        "normalize raw",
        "black white balancing",
        "sensor linear",
    ]

    RETURN_TYPES = ("IMAGE", "PATTERN")
    RETURN_NAMES = ("bayer_img", "cfa_pattern")
    FUNCTION = "execute"

    def execute(self, raw_img, cfa_pattern, black_levels, white_level):
        raw_2d = raw_img.squeeze()
        pattern_2d = cfa_pattern.squeeze()

        linearized = linearize_raw(
            raw_img=raw_2d,
            bayer_pattern=pattern_2d,
            black_levels=black_levels,
            white_level=white_level,
        )

        return (linearized.unsqueeze(0).unsqueeze(-1), cfa_pattern)


NODE_CLASS_MAPPINGS = {
    "LinearizeRawNode": LinearizeRawNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LinearizeRawNode": "Black Level Subtraction",
}
