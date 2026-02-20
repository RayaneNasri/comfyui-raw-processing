from algorithms.demosaicing.bilinear import bilinear_demosaicing, mono_to_rgb


class BilinearDemosaicNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bayer_img": ("IMAGE",),
                "cfa_pattern": ("PATTERN",),
            }
        }

    CATEGORY = "image"
    SEARCH_ALIASES = [
        "bilinear demosaicing",
        "debayer image",
        "bayer filter interpolation",
        "raw to rgb",
        "demosaic bayer",
        "reconstruct image colors",
        "bilinear debayer",
    ]

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGB_image",)
    FUNCTION = "execute"

    def execute(self, bayer_img, cfa_pattern):
        img_2d = bayer_img.squeeze()
        pattern_2d = cfa_pattern.squeeze()

        top_left_2x2 = pattern_2d[:2, :2]
        coords = (top_left_2x2 == 0).nonzero(as_tuple=True)

        if len(coords[0]) > 0:
            detected_dy = int(coords[0][0].item())
            detected_dx = int(coords[1][0].item())
        else:
            raise ValueError("No red pixel found on the top left square 2x2")

        sparse_rgb = mono_to_rgb(img_2d, pattern_2d)
        result = bilinear_demosaicing(sparse_rgb, dx=detected_dx, dy=detected_dy)
        return (result.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"BilinearDemosaicNode": BilinearDemosaicNode}

NODE_DISPLAY_NAME_MAPPINGS = {"BilinearDemosaicNode": "Bilinear Demosaicing"}
