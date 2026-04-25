# Changelog
All the important changes and releases will be documented in this file. 

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


