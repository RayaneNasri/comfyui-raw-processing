from algorithms.bilinear_demosaicing import bilinear_demosaicing

import torch


class BilinearDemosaicNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bayer_img": ("IMAGE",),
                "cfa_pattern": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGB_image",)
    FUNCTION = "execute"
    CATEGORY = "image/processing"

    def execute(self, bayer_img, cfa_pattern):
        # Remove batch dim and channel dim to get [H, W]
        bayer_2d = bayer_img.squeeze()
        pattern_2d = cfa_pattern.squeeze()
        H, W = bayer_2d.shape

        # We look for the first Red pixel (value 0 in rawpy) in the top-left 2x2
        top_left_2x2 = pattern_2d[:2, :2]
        coords = (top_left_2x2 == 0).nonzero(as_tuple=True)

        if len(coords[0]) > 0:
            detected_dy = int(coords[0][0].item())
            detected_dx = int(coords[1][0].item())
        else:
            detected_dy, detected_dx = 0, 0

        # Convert 1-channel Bayer to 3-channel sparse RGB
        # the function expects a (H, W, 3) tensor where only the sampled
        # color for each pixel is non-zero.
        sparse_rgb = torch.zeros((H, W, 3), device=bayer_2d.device, dtype=bayer_2d.dtype)

        # Map Bayer pixels to their respective RGB channels based on detected phase
        sparse_rgb[detected_dy::2, detected_dx::2, 0] = bayer_2d[detected_dy::2, detected_dx::2]  # Red
        sparse_rgb[1 - detected_dy :: 2, 1 - detected_dx :: 2, 2] = bayer_2d[
            1 - detected_dy :: 2, 1 - detected_dx :: 2
        ]  # Blue
        sparse_rgb[detected_dy::2, 1 - detected_dx :: 2, 1] = bayer_2d[detected_dy::2, 1 - detected_dx :: 2]  # Green 1
        sparse_rgb[1 - detected_dy :: 2, detected_dx::2, 1] = bayer_2d[1 - detected_dy :: 2, detected_dx::2]  # Green 2

        result = bilinear_demosaicing(sparse_rgb, dx=detected_dx, dy=detected_dy)

        return (result.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"BilinearDemosaicNode": BilinearDemosaicNode}

NODE_DISPLAY_NAME_MAPPINGS = {"BilinearDemosaicNode": "Bilinear Demosaicing"}
