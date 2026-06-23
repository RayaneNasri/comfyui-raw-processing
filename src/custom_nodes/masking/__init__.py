from .interactive_mask_node import InteractiveSegmentationMask

NODE_CLASS_MAPPINGS = {
    "InteractiveSegmentationMask": InteractiveSegmentationMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InteractiveSegmentationMask": "Interactive Segmentation Mask",
}

WEB_DIRECTORY = "./js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
