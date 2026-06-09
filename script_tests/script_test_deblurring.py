import torch
from pathlib import Path

# Add source to path if needed
PATH = '/mnt/c/Users/charl/git/'

import sys
sys.path.insert(0, PATH + 'image-processing/src')

# Imports for each module
from algorithms.deblurring._deblurring_goldstein_fattal import read_image
from algorithms.deblurring._deblurring_goldstein_fattal import deblurring_goldstein_fattal
from algorithms.export._jpeg_export import export_jpeg

def script_test_deblurring():

    # Input file
    numpy_blurred_img = read_image(PATH + "rochers_mer.png")
    blurred_img = torch.from_numpy(numpy_blurred_img).float()
    blurred_img /= 255.0

    output_dir = "./outputs"
    output_file = "deblurred_image_rochers_mer.jpg"

    print("Deblurring Goldstein-Fattal...")
    deblurred_img = deblurring_goldstein_fattal(blurred_img)

    print("Exporting to JPEG...")
    Path(output_dir).mkdir(exist_ok=True)
    export_jpeg(deblurred_img, f"{output_dir}/{output_file}", quality=85)

    print(f"✓ Saved to {output_dir}/{output_file}")

script_test_deblurring()