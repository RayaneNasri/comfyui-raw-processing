import torch
import numpy as np
import rawpy
import os

from algorithms.raw_processing import read_raw

class RawBayerNode: 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path": ("STRING", {"default": "input/image.ARW"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "VEC4") 
    RETURN_NAMES = ("bayer_img", "cfa_pattern", "wb_gains")
    FUNCTION = "execute"
    CATEGORY = "image/raw"
    
    def execute(self, image_path):
        bayer_img, cfa_pattern, wb_gains = read_raw(image_path)
        bayer_img = bayer_img.unsqueeze(0).unsqueeze(-1)
        cfa_pattern = cfa_pattern.unsqueeze(0)
        
        return (bayer_img, cfa_pattern, wb_gains)

NODE_CLASS_MAPPINGS = {
    "RawBayerNode": RawBayerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RawBayerNode": "Read RAW (Bayer Mosaic)"
}