import torch
from torch import Tensor
from torchvision.io import write_jpeg

from PIL import Image

from algorithms._utils import _to_uint8_numpy, validate_image_input

import os


def to_jpeg(image: Tensor, path: str, quality: int = 75):
    """
    Exports an sRGB tensor image to a JPEG file.
    Args:
        image (Tensor): A tensor of shape [H, W, 3] with float values in the range [0, 1]
        path (str): The file path to save the JPEG image
        quality (int, optional): The quality of the JPEG image (0-100). Default is 75
    """

    img_chw = image.permute(2, 0, 1)
    img_chw = img_chw.nan_to_num_(nan=0.0, posinf=1.0, neginf=0.0)
    image_uint8 = img_chw.mul(255).clamp_(0, 255).to(torch.uint8)

    write_jpeg(image_uint8, path + ".jpeg", quality=quality)


def to_png(image: Tensor, path: str, quality: int = 75):
    """
    Exports an sRGB tensor image to a PNG file.
    Args:
        image (Tensor): A tensor of shape [H, W, 3] with float values in the range [0, 1]
        path (str): The file path to save the PNG image
        quality (int, optional): The quality of the PNG image (0-100). Default is 75
    """
    arr = _to_uint8_numpy(image)
    img = Image.fromarray(arr)
    compress_level = int(round(quality * 9 / 100))
    img.save(path + ".png", format="PNG", compress_level=compress_level)


def to_tiff(image: Tensor, path: str, quality: int = 75):
    """
    Exports an sRGB tensor image to a TIFF file.
    Args:
        image (Tensor): A tensor of shape [H, W, 3] with float values in the range [0, 1]
        path (str): The file path to save the TIFF image
        quality (int, optional): The quality of the TIFF image (0-100). Default is 75
    """
    arr = _to_uint8_numpy(image)
    img = Image.fromarray(arr)
    # Use LZW compression for smaller files, leave uncompressed when very high quality requested
    compression = "none" if quality >= 95 else "tiff_lzw"
    img.save(path + ".tiff", format="TIFF", compression=compression)


def to_webp(image: Tensor, path: str, quality: int = 75):
    """
    Exports an sRGB tensor image to a WebP file.
    Args:
        image (Tensor): A tensor of shape [H, W, 3] with float values in the range [0, 1]
        path (str): The file path to save the WEBP image
        quality (int, optional): The quality of the WEBP image (0-100). Default is 75
    """
    arr = _to_uint8_numpy(image)
    img = Image.fromarray(arr)
    img.save(path + ".webp", format="WEBP", quality=int(max(0, min(100, quality))))


@validate_image_input
def export(
    img: Tensor,
    format: str,
    folder: str = "/output",
    file: str = "image_processed",
    quality: int = 75,
) -> None:
    """Exports an sRGB tensor image to the format specified

    Args:
        img (Tensor): a tensor of shape [H, W, 3] with float values in the range [0, 1]
        format (str): the format desired (JPEG, PNG, TIFF, WEBP)
        folder (str): the folder path to save the image
        file (str): the file path to save the image
        quality (int, optional): the quality of the WEBP image (0-100). Default is 75
    """

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    if os.path.isfile(folder):
        raise ValueError(f"The folder path '{folder}' is a file, not a directory.")

    formats = {"JPEG": to_jpeg, "PNG": to_png, "TIFF": to_tiff, "WEBP": to_webp}

    path = folder + "/" + file
    formats[format](img, path, quality)
