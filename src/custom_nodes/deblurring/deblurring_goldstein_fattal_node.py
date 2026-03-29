from algorithms.deblurring._deblurring_goldstein_fattal import deblurring_goldstein_fattal


class DeblurringGoldsteinFattalNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "RGB_image": ("IMAGE",),
            }
        }

    FUNCTION = "process"
    CATEGORY = "image/processing"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGB_image",)

    def process(self, RGB_image):
        return deblurring_goldstein_fattal(RGB_image)


NODE_CLASS_MAPPINGS = {
    "DeblurringGoldsteinFattalNode": DeblurringGoldsteinFattalNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeblurringGoldsteinFattalNode": "Deblurring Goldstein-Fattal",
}