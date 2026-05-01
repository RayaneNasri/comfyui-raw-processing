from .gamma_curve_node import (
    NODE_CLASS_MAPPINGS as _GAMMA_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _GAMMA_NAMES,
)
from .tone_curve_node import (
    NODE_CLASS_MAPPINGS as _TONE_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _TONE_NAMES,
)

NODE_CLASS_MAPPINGS = {**_TONE_NODES, **_GAMMA_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_TONE_NAMES, **_GAMMA_NAMES}

# Tells ComfyUI where to find the frontend JS extension
WEB_DIRECTORY = "./js"
