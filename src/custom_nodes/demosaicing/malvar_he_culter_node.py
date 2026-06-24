from algorithms.demosaicing._malvar_he_culter import malvar_he_cutler_demosaicing


class MalvarHeCutlerDemosaicNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bayer_img": ("IMAGE",),
                "cfa_pattern": ("PATTERN",),
            }
        }

    SEARCH_ALIASES = [
        "malvar demosaicing",
        "debayer image",
        "bayer filter interpolation",
        "raw to rgb",
        "demosaic bayer",
        "reconstruct image colors",
        "malvar debayer",
    ]

    FUNCTION = "execute"
    CATEGORY = "image/processing/demosaicing"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGB_image",)

    def execute(self, bayer_img, cfa_pattern):
        bayer_2d = bayer_img.squeeze()
        pattern_2d = cfa_pattern.squeeze()

        top_left_2x2 = pattern_2d[:2, :2]
        coords = (top_left_2x2 == 0).nonzero(as_tuple=True)

        if len(coords[0]) > 0:
            detected_dy = int(coords[0][0].item())
            detected_dx = int(coords[1][0].item())
        else:
            detected_dy, detected_dx = 0, 0

        result = malvar_he_cutler_demosaicing(
            bayer_2d,
            dx=detected_dx,
            dy=detected_dy,
        )

        return (result.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "MalvarHeCutlerDemosaicNode": MalvarHeCutlerDemosaicNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MalvarHeCutlerDemosaicNode": "Malvar-He-Cutler Demosaicing",
}
