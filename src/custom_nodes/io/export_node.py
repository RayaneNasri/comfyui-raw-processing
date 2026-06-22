from algorithms.io import export


class ExportNode:
    @classmethod
    def INPUT_TYPES(cls):
        formats: list[str] = ["JPEG", "PNG", "TIFF", "WEBP"]

        return {
            "required": {
                "image": ("IMAGE",),
                "folder": ("STRING", {"default": "./output"}),
                "filename": ("STRING", {"default": "processed_image"}),
                "quality": (
                    "INT",
                    {
                        "default": 75,
                        "min": 1,
                        "max": 100,
                        "display": "slider",
                        "step": 1,
                    },
                ),
                "format": (formats,),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "image/export"

    def process(self, image, folder, filename, quality, format):

        input_2d = image.squeeze(0)
        export(input_2d, format, folder, filename, quality)

        return ()


NODE_CLASS_MAPPINGS = {"ExportNode": ExportNode}

NODE_DISPLAY_NAME_MAPPINGS = {"ExportNode": "Export"}
