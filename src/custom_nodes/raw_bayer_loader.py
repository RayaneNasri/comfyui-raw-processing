from algorithms.raw_processing import read_raw

class RawBayerNode: 
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "input/image.ARW"}),
            }
        }

    CATEGORY = "image"
    SEARCH_ALIASES = ["load image", "open image", "import image", "image input", "upload image", "read image", "image loader"]

    RETURN_TYPES = ("IMAGE", "PATTERN", "WB_GAIN") 
    RETURN_NAMES = ("bayer_img", "cfa_pattern", "wb_gains")
    FUNCTION = "execute"
    
    def execute(self, image_path):
        bayer_img, cfa_pattern, wb_gains = read_raw(image_path)
        bayer_img = bayer_img.unsqueeze(0).unsqueeze(-1)
        cfa_pattern = cfa_pattern.unsqueeze(0)
        
        return (bayer_img, cfa_pattern, wb_gains)

NODE_CLASS_MAPPINGS = {
    "RawBayerNode": RawBayerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RawBayerNode": "Read RAW"
}