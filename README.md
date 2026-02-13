# Designing a Modular Image Processing Pipeline Software

This document explains installation and configuration steps driven by the repository Makefile.

## Prerequisites

- **Git** with submodule support
- **UV** package manager ([installation guide](https://github.com/astral-sh/uv))
- **Python 3.13+**
- **Hardware-specific requirements:**
  - NVIDIA GPU: CUDA 13.0 compatible drivers
  - Intel GPU: Compatible Intel Graphics ([check compatibility](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html))
  - macOS: Apple Silicon with macOS 12.3+ for MPS acceleration

## Clone Repository (with Submodules)

**Recommended:**
```bash
git clone --recursive <repo-url>
```

**If already cloned without submodules:**
```bash
git submodule update --init --recursive
```

## Quick Status Checks

### Show Complete Environment Status
Displays venv status, ComfyUI submodule status, and PyTorch installation details:
```bash
make status
```

### Individual Checks
- **Check virtual environment exists:**
```bash
  make check-venv
```

- **Check ComfyUI submodule and ensure it is initialized:**
```bash
  make check-comfyui
```

- **Check PyTorch installation and device info:**
```bash
  make check-torch
```

## Setup (Create venv + Install Dependencies)

All setup targets create a virtual environment (`.venv/`) and install ComfyUI dependencies, ComfyUI Manager requirements, and project-specific dependencies.

### NVIDIA GPU (CUDA 13.0)
Uses PyTorch nightly builds with CUDA 13.0 support:
```bash
make setup-cuda130
```

### CPU Only
For systems without GPU or testing purposes:
```bash
make setup-cpu
```

### Intel XPU (Intel Graphics)
For Intel integrated graphics and Arc GPUs. 

**Important:** Check [compatibility requirements](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html) first:
```bash
make setup-xpu
```

### macOS (Apple Silicon/MPS)
For Macs with M1/M2/M3 chips, uses Metal Performance Shaders:
```bash
make setup-mac
```

**Notes:**
- If `.venv` already exists, setup will abort with a warning
- Remove existing environment first with `make clean` if you need to recreate it
- Setup automatically installs:
  - PyTorch (hardware-specific version)
  - ComfyUI dependencies (`external/ComfyUI/requirements.txt`)
  - ComfyUI Manager dependencies (`external/ComfyUI/manager_requirements.txt`)
  - Project dependencies (`requirements_project.txt`, if present)
  - Current project in editable mode

## Running ComfyUI

### Auto-Detect Hardware (Recommended)
Automatically detects CUDA or MPS (Mac) GPU and uses it, falls back to CPU if unavailable:
```bash
make run
```

### Force CPU Mode
Explicitly run with CPU (useful for testing or troubleshooting):
```bash
make run-cpu
```

### Force GPU Mode
Run with GPU acceleration (CUDA only, will fail if CUDA is not available):
```bash
make run-gpu
```

**What Gets Launched:**
- Script: `external/ComfyUI/main.py`
- Default flags: `--enable-manager --preview-method latent2rgb`
- CPU mode adds: `--cpu`

## Updating

### Update Everything
Updates both the ComfyUI submodule and all Python dependencies:
```bash
make update
```

### Update ComfyUI Submodule Only
Updates only the ComfyUI repository to the latest commit:
```bash
make update-comfyui
```

## Cleanup

### Remove Virtual Environment
Deletes the `.venv/` directory (useful before recreating environment):
```bash
make clean
```

## Submodule Troubleshooting

### ComfyUI Not Present or Outdated
Manually initialize or update the submodule:
```bash
git submodule update --init --recursive external/ComfyUI
```

### Submodule Not Configured
If you get submodule-related errors, the Makefile will attempt to initialize it automatically. If issues persist:
```bash
# Add the submodule manually
git submodule add https://github.com/comfyanonymous/ComfyUI.git external/ComfyUI
git submodule update --init --recursive
```

## Hardware Detection Details

The Makefile automatically detects available hardware:

- **CUDA (NVIDIA):** Checks `torch.cuda.is_available()`
- **MPS (Apple Silicon):** Checks `torch.backends.mps.is_available()`
- **Fallback:** Uses CPU if neither CUDA nor MPS is available

View detection results with `make status`.

## Project Structure
```
your-project/
├── .venv/                          # Virtual environment (created by setup)
├── external/
│   └── ComfyUI/                    # ComfyUI submodule
├── custom_nodes/                   # Your custom nodes
├── requirements_project.txt        # Optional: Your project dependencies
├── Makefile                        # Build automation
└── pyproject.toml                  # Project configuration
```

## Notes

- **UV vs Manual venv:** The Makefile uses `uv run python -c ...` for environment inspection. If using manual venv activation instead of UV, replace with `python -c ...`
- **Lock Files:** If you want reproducible installs, consider tracking `uv.lock` in version control; otherwise keep it in `.gitignore`
- **CUDA 13.0:** This uses PyTorch nightly builds which may be unstable. For production, consider using stable CUDA 12.x builds
- **First Setup:** Always run `make status` after initial setup to verify installation
- **Color Output:** The Makefile uses colored output for better readability. If colors don't display properly, check your terminal's ANSI support

## Common Workflows

### First Time Setup (NVIDIA GPU)
```bash
git clone --recursive <repo-url>
cd <repo-name>
make setup-cuda130
make status  # Verify installation
make run     # Launch ComfyUI
```

### First Time Setup (macOS)
```bash
git clone --recursive <repo-url>
cd <repo-name>
make setup-mac
make status  # Verify installation
make run     # Launch ComfyUI
```

### Regular Updates
```bash
make update  # Update ComfyUI and dependencies
make run     # Launch with updated version
```

### Switching Hardware Configurations
```bash
make clean           # Remove old environment
make setup-cpu       # Install CPU-only version
make run-cpu         # Run in CPU mode
```

## Getting Help

Display all available commands:
```bash
make help
```

Or simply:
```bash
make
```