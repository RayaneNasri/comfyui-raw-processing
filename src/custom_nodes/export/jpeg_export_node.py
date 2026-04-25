import os


from algorithms.export._jpeg_export import export_jpeg


class JpegExportNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "folder_path": ("STRING", {"default": "./output"}),
                "filename": ("STRING", {"default": "processed_image"}),
                "quality": ("INT", {"default": 75, "min": 1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "image/export"

    def process(self, image, folder_path, filename, quality):
        if not os.path.exists(folder_path):
            raise ValueError(f"The folder path '{folder_path}' does not exist.")

        if os.path.isfile(folder_path):
            raise ValueError(
                f"The folder path '{folder_path}' is a file, not a directory."
            )

        if os.path.exists(f"{folder_path}/{filename}.jpg"):
            raise ValueError(
                f"The file '{filename}.jpg' already exists in the folder '{folder_path}'. Please choose a different filename or folder path."
            )

        squeezed_image = image.squeeze(0)  # Remove batch dimension if it exists
        export_jpeg(squeezed_image, f"{folder_path}/{filename}.jpg", quality)

        return ()


NODE_CLASS_MAPPINGS = {"JpegExportNode": JpegExportNode}

NODE_DISPLAY_NAME_MAPPINGS = {"JpegExportNode": "Save JPEG (Custom Path)"}
