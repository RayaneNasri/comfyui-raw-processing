from .interactive_mask_node import InteractiveSegmentationMask

# ---------------------------------------------------------------------------
# ComfyUI node registration mappings
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "InteractiveSegmentationMask": InteractiveSegmentationMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InteractiveSegmentationMask": "Interactive Segmentation Mask",
}

# ---------------------------------------------------------------------------
# Web extension registration
# ComfyUI will automatically serve every JS file listed here under /extensions/
# ---------------------------------------------------------------------------

WEB_DIRECTORY = "./js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
