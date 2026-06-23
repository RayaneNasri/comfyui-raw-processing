from .camera_wb_node import (
    NODE_CLASS_MAPPINGS as _CAMERA_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _CAMERA_NAMES,
)
from .ground_truth_node import (
    NODE_CLASS_MAPPINGS as _GT_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _GT_NAMES,
)
from .gw_node import (
    NODE_CLASS_MAPPINGS as _GW_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _GW_NAMES,
)
from .wb_comparison_node import (
    NODE_CLASS_MAPPINGS as _COMP_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _COMP_NAMES,
)
from .white_pr_node import (
    NODE_CLASS_MAPPINGS as _WPR_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _WPR_NAMES,
)

NODE_CLASS_MAPPINGS = {
    **_CAMERA_NODES,
    **_GT_NODES,
    **_GW_NODES,
    **_COMP_NODES,
    **_WPR_NODES,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_CAMERA_NAMES,
    **_GT_NAMES,
    **_GW_NAMES,
    **_COMP_NAMES,
    **_WPR_NAMES,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
