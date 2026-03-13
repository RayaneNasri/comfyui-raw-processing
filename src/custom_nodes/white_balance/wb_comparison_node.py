import torch

from algorithms.white_balance import camera_white_balance, gw, white_patch_ref


class CameraWhiteBalanceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "wb_gains": ("WB_GAIN",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "compare_percentil": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
            }
        }

    CATEGORY = "image"
    SEARCH_ALIASES = [
        "white balance adjustment",
        "camera white balance",
        "raw camera whitebalance",
        "raw metadata white balance",
        "gray world comparison",
        "white patch comparison",
    ]

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "camera_wb_image",
        "gray_world_image",
        "white_patch_image",
    )
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        wb_gains: torch.Tensor,
        strength: float,
        compare_percentil: float,
    ) -> tuple:
        input_3d = image.squeeze()

        out_camera = camera_white_balance(input_3d, wb_gains, strength)
        out_gw = gw(input_3d)
        out_wpr = white_patch_ref(input_3d, compare_percentil)

        return (
            out_camera.unsqueeze(0),
            out_gw.unsqueeze(0),
            out_wpr.unsqueeze(0),
        )


NODE_CLASS_MAPPINGS = {"CameraWhiteBalanceNode": CameraWhiteBalanceNode}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraWhiteBalanceNode": "Camera White Balance + Compare",
}
