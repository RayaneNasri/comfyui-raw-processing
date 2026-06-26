import torch

from algorithms.white_balance import camera_white_balance


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
            }
        }

    CATEGORY = "image/processing/white-balance"
    SEARCH_ALIASES = [
        "camera white balance",
        "raw camera whitebalance",
        "raw metadata white balance",
        "awb",
    ]

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("camera_wb_image",)
    FUNCTION = "execute"

    def execute(
        self,
        image: torch.Tensor,
        wb_gains: torch.Tensor,
        strength: float,
    ) -> tuple:
        input_3d = image.squeeze()
        out_camera = camera_white_balance(input_3d, wb_gains, strength)
        return (out_camera.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"CameraWhiteBalanceNode": CameraWhiteBalanceNode}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraWhiteBalanceNode": "Camera White Balance",
}
