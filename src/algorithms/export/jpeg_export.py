import torch

from torch import Tensor 
from torchvision.io import write_jpeg

def export_jpeg(image: Tensor, path: str, quality: int = 75): 
    """ 
    Exports an sRGB tensor image to a JPEG file.
    Args:
        image (Tensor): A tensor of shape [H, W, 3] with float values in the range [0, 1].
        path (str): The file path to save the JPEG image.
        quality (int, optional): The quality of the JPEG image (0-100). Default is 75.
    """

    img_reshaped = image.permute(2, 0, 1)
    write_jpeg(img_reshaped, path, quality = quality)

