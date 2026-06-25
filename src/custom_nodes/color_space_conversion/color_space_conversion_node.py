from algorithms.color_space_conversion.rgb_to_yuv import rgb_to_yuv
from custom_nodes.deblurring.deblurring_goldstein_fattal_node import DeblurringGoldsteinFattalNode


class RGBToYUVNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "RGB_image": ("IMAGE",),
            }
        }

    FUNCTION = "process"
    CATEGORY = "image/processing/color_space_conversion"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("YUV_image",)

    def process(self, RGB_image):
        RGB_image = RGB_image.squeeze(0)
        res = rgb_to_yuv(RGB_image)
        return (res.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "RGBToYUVNode": RGBToYUVNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGBToYUVNode": "RGB to YUV", 
}
