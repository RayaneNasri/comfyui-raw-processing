import torch

class ExposureCompensationNode: 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "ev_compensation": ("FLOAT", {
                    "default": 0.0, 
                    "min": -10.0, 
                    "max": 10.0, 
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "image/processing"
    
    def process(self, image, ev_compensation):
        
        gain = 2.0 ** ev_compensation
        result = torch.clamp(image * gain, 0.0, 1.0)
        
        return (result,)

NODE_CLASS_MAPPINGS = {
    "ExposureCompensationNode": ExposureCompensationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExposureCompensationNode": "Exposure Compensation",
}