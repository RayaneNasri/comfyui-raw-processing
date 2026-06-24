import numpy as np
import torch
import cv2
import math

# Import from the original HDR+ repository files
from .algorithm.alignment import alignHdrplus
from .algorithm.merging import mergeHdrplus

class MockTag:
    """A simple mock class to emulate exifread tag objects for the original HDR+ code."""
    def __init__(self, values):
        self.values = values

def process_hdrplus_burst(raw_imgs, black_levels, white_levels, exif_tags, ref_idx=0, temporal_factor=75.0, spatial_factor=0.1):
    """
    Bridges ComfyUI PyTorch tensors with the original HDR+ numpy/numba implementation.
    """
    B, H, W, C = raw_imgs.shape
    
    # 1. Convert Tensors to list of 2D NumPy arrays
    images_np = [raw_imgs[i, ..., 0].cpu().numpy() for i in range(B)]
    
    # 2. Extract specific metadata for the reference image
    ref_exif = exif_tags[ref_idx]
    tags = {}
    if "noise_profile" in ref_exif and ref_exif["noise_profile"]:
        tags['Image Tag 0xC761'] = MockTag(ref_exif["noise_profile"])
    if "iso" in ref_exif and ref_exif["iso"]:
        tags['Image ISOSpeedRatings'] = MockTag([ref_exif["iso"]])
        
    b_level = black_levels[ref_idx].cpu().numpy()
    w_level = white_levels[ref_idx].item()

    # 3. Setup parameters according to the original params.py ('merge' mode equivalent)
    params = {
        'tuning': {
            'factors': [1, 2, 4, 4],
            'tileSizes': [16, 16, 16, 8],
            'searchRadia': [1, 4, 4, 4],
            'distances': ['L1', 'L2', 'L2', 'L2'],
            'subpixels': [False, True, True, True],
            'patchSize': 16,
            'method': 'DFTWiener',
            'noiseCurve': 'exifNoiseProfile'
        },
        'mode': 'bayer'
    }
    
    options = {
        'verbose': 1,
        'temporalFactor': temporal_factor,
        'spatialFactor': spatial_factor
    }

    # 4. Padding calculation (identical to alignment.py)
    tileSize = 2 * params['tuning']['tileSizes'][0]
    paddingPatchesHeight = (tileSize - H % (tileSize)) * (H % (tileSize) != 0)
    paddingPatchesWidth = (tileSize - W % (tileSize)) * (W % (tileSize) != 0)
    paddingOverlapHeight = paddingOverlapWidth = tileSize // 2
    
    paddingTop = paddingOverlapHeight
    paddingBottom = paddingOverlapHeight + paddingPatchesHeight
    paddingLeft = paddingOverlapWidth
    paddingRight = paddingOverlapWidth + paddingPatchesWidth
    padding = (paddingTop, paddingBottom, paddingLeft, paddingRight)

    # Pad all images (symmetric mirroring)
    images_padded = [np.pad(im, ((paddingTop, paddingBottom), (paddingLeft, paddingRight)), 'symmetric') for im in images_np]
    
    imRef = images_padded[ref_idx]
    alternateImages = [img for i, img in enumerate(images_padded) if i != ref_idx]

    # 5. Run Alignment
    print(f"Aligning {B-1} alternate images to reference index {ref_idx}...")
    motionVectors, alignedTiles = alignHdrplus(imRef, alternateImages, params, options)
    
    # 6. Run Merging
    print("Fusing aligned burst...")
    merged_image_np = mergeHdrplus(imRef, alignedTiles, padding, tags, b_level, w_level, params['tuning'], options)
    
    # 7. Convert back to PyTorch Tensor [1, H, W, 1]
    merged_tensor = torch.from_numpy(merged_image_np).unsqueeze(0).unsqueeze(-1).to(raw_imgs.dtype)
    
    return merged_tensor

# --- Tone Mapping Implementations adapted from finishing.py ---

def apply_global_tone_mapping(image_tensor, contrast=0.075):
    """PyTorch native implementation of the HDR+ S-curve contrast enhancement."""
    # S-Curve: x -= gain * sin(2 * pi * x)
    enhanced = image_tensor - contrast * torch.sin(2 * math.pi * image_tensor)
    return torch.clamp(enhanced, 0.0, 1.0)

def apply_local_tone_mapping(image_tensor, gain=0.0):
    """Applies Mertens Exposure Fusion using OpenCV (requires CPU switch)."""
    img_np = image_tensor.squeeze(0).cpu().numpy()
    
    short_gray = img_np.mean(axis=2)
    
    # Auto-gain logic from finishing.py
    if gain <= 0.0:
        dsFactor = 25
        shortS = cv2.resize(short_gray, (0, 0), fx=1/dsFactor, fy=1/dsFactor).flatten()
        bestGain = False
        gain_val, compression, saturated = 0, 1.0, 0.0
        
        # Approximate sRGB compression for metering
        threshold = 0.0031308
        shortSg = np.where(shortS <= threshold, 12.92 * shortS, 1.055 * (shortS ** (1/2.4)) - 0.055)
        sSMean = np.mean(shortSg)
        
        while (compression < 1.9 and saturated < .95) or (not bestGain and compression < 6 and gain_val < 30 and saturated < 0.33):
            gain_val += 2
            longS_scaled = np.clip(gain_val * shortS, 0., 1.)
            longSg = np.where(longS_scaled <= threshold, 12.92 * longS_scaled, 1.055 * (longS_scaled ** (1/2.4)) - 0.055)
            lSMean = np.mean(longSg)
            compression = lSMean / sSMean if sSMean > 0 else 1.0
            bestGain = lSMean > (1 - sSMean) / 2
            saturated = np.sum(longSg > 0.95) / np.size(longSg)
        gain = gain_val

    # Synthetic Long Exposure
    long_img = np.clip(img_np * gain, 0., 1.)
    
    # Gamma compress both for Mertens fusion
    shortg = np.where(img_np <= 0.0031308, 12.92 * img_np, 1.055 * (img_np ** (1/2.4)) - 0.055).astype(np.float32)
    longg = np.where(long_img <= 0.0031308, 12.92 * long_img, 1.055 * (long_img ** (1/2.4)) - 0.055).astype(np.float32)
    
    # Exposure fusion
    mergeMertens = cv2.createMergeMertens(contrast_weight=0., saturation_weight=0., exposure_weight=1.)
    fusedg = mergeMertens.process([shortg, longg])
    
    # Gamma decompress
    fused_gray = np.where(fusedg <= 0.04045, fusedg / 12.92, ((fusedg + 0.055) / 1.055) ** 2.4)
    
    # Scale original RGB
    epsilon = 1e-6
    scale = np.where(short_gray == 0, 1.0, fused_gray.mean(axis=-1) / (short_gray + epsilon))
    scale = scale[..., np.newaxis]
    
    ltm_img = np.clip(img_np * scale, 0., 1.)
    return torch.from_numpy(ltm_img).unsqueeze(0).to(image_tensor.dtype)