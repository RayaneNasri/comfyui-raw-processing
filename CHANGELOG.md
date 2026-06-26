# Changelog
All the important changes and releases will be documented in this file. 

## [1.0.0] - 2026-06-26
A fally implemented pipeline with all stages of processing with extensions.

### Added 
- Lens Correction algorithm & node.
- Denoising algorithms & nodes.
- A new IEC Gamma Correction node & algorithm.
- A modular general node for curve editing.
- Tone Curve algorithm & node.
- Masking node & algorithm for differential processing using SLIC algorithm and SAM (deep learning model).
- Saturation manipulation algorithm & node.
- LUT application algorithm & node.
- Temperature (simple and Tanner-Helland) manipulation algorithm & node.
- Simple deblurring algorithm & node.
- Presets for camera DCP profiles and LUTs.
- Batch RAW Reading node for multiple RAW images reading.

### Fixed
- UI for RAW reading node by integrating OS based finder.
- UI for Hue Saturation Mapping node for reading DCP files based on camera profile.
- Problems related to memory consumption and usage for pipelines.
- Supporting multiple RAW images formats

### Known Limitations
- Python 3.13+ required.
- ComfyUI integration is non-official.

## [0.1.0] - 2026-03-26
This first version implements a complete pipeline for processing RAW image files `.arw` to 8-bit sRGB JPEG output.

### Added
- ComfyUI node for reading RAW data (ARW image files).
- Bilinear demosaicing algorithm & ComfyUI node (Bayer pattern).
- Malvar-He-Cutler demosaicing algorithm & ComfyUI node.
- Black light subtraction algorithm & ComfyUI node.
- Exposure compensation algorithm & ComfyUI node.
- Gray World white balance algorithm & ComfyUI node.
- Ground Truth white balance algorithm & ComfyUI node.
- White Patch Reference white balance algorithm & ComfyUI node.
- Camera white balance algorithm & ComfyUI node.
- Automatic CI/CD pipeline: format checks, linting, type checking, unit tests.

### Fixed
- Makefile device-depedent installation process, now it automatically detects architecture and installs the right PyTorch's dependencies.

### Known Limitations
- Tested on ARW format only (Sony).
- Python 3.13+ required.
- ComfyUI integration is non-official.


