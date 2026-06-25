from algorithms.deblurring._deblurring_goldstein_fattal import (
    deblurring_goldstein_fattal,
)


class DeblurringGoldsteinFattalNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "RGB_image": ("IMAGE",),
            }
        }

    FUNCTION = "process"
    CATEGORY = "image/processing/deblurring"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGB_image",)

    def process(self, RGB_image):
        RGB_image = RGB_image.squeeze(0)
        res = deblurring_goldstein_fattal(RGB_image)
        return (res.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "DeblurringGoldsteinFattalNode": DeblurringGoldsteinFattalNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeblurringGoldsteinFattalNode": "Deblurring Goldstein-Fattal",
}
