import rawpy as rp
import numpy as np
import torch
import os

class RawImageProcessor:
    """
    Custom ComfyUI node for processing RAW image files.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # TEXT input allows you to paste a full path
                "raw_file_path": ("STRING", {"default": "C:/path/to/your/image.ARW"}), 
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_raw"
    CATEGORY = "image/raw"

    def process_raw(self, raw_file_path):
        # 1. Check if file exists to avoid hard crashes
        if not os.path.exists(raw_file_path):
            print(f"Error: File not found at {raw_file_path}")
            # Return a blank black image to prevent pipeline breakage
            return (torch.zeros((1, 512, 512, 3)),)

        try:
            # 2. Use 'with' context manager for safe file handling
            with rp.imread(raw_file_path) as raw:
                data = raw.raw_image_visible.copy() # Copy to ensure data persists after close
                
                # Your logic: Align Bayer pattern
                dy, dx = np.argwhere(raw.raw_colors == 0)[0]
                raw_data = data[dy:, dx:]

                # Normalize to 0-1
                normalized = raw_data.astype(np.float32) / raw_data.max()

                # Expand to (H, W, 1)
                tensor = np.expand_dims(normalized, axis=2)
                
                # Concatenate to (H, W, 3) -> RGB Grayscale
                tensor = np.concatenate([tensor] * 3, axis=2)

                # 3. CRITICAL FIX: Add Batch Dimension (1, H, W, C)
                tensor = np.expand_dims(tensor, axis=0)

                # Convert to torch tensor (ComfyUI prefers torch over numpy)
                return (torch.from_numpy(tensor),)
                
        except Exception as e:
            print(f"Error processing RAW file: {e}")
            return (torch.zeros((1, 512, 512, 3)),)

# Node registration
NODE_CLASS_MAPPINGS = {"RawImageProcessor": RawImageProcessor}
NODE_DISPLAY_NAME_MAPPINGS = {"RawImageProcessor": "Raw Image Processor"}