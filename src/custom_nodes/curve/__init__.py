from .tone_curve_node import (
    NODE_CLASS_MAPPINGS as _TONE_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _TONE_NAMES,
)

NODE_CLASS_MAPPINGS = {**_TONE_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_TONE_NAMES}

# Tells ComfyUI where to find the frontend JS extension
WEB_DIRECTORY = "./js"
