# Designing a Modular Image Processing Pipeline Software

This document explains installation and configuration steps driven by the repository Makefile.

## Prerequisites

- **Git** with submodule support
- **uv** package manager ([installation guide](https://github.com/astral-sh/uv))
- **Python 3.13+**
- **Hardware-specific requirements:**
  - NVIDIA GPU: CUDA 13.0 compatible drivers
  - Intel GPU: Compatible Intel Graphics ([check compatibility](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html))
  - macOS: Apple Silicon with macOS 12.3+ for MPS acceleration

> **Note:** It can be run on CPU-only systems, but performance will be significantly slower.

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

## Setup (Create venv + Install Dependencies)

All setup targets create a virtual environment (`.venv/`) and install ComfyUI dependencies, ComfyUI Manager requirements, and project-specific dependencies.

### Auto-Detect Setup (Recommended)
Automatically detects your hardware (NVIDIA GPU, macOS Apple Silicon, or CPU) and installs the appropriate PyTorch version:
```bash
 make setup
```
*   **NVIDIA GPU:** Installs PyTorch with CUDA 13.0 support.
*   **macOS:** Installs PyTorch with MPS support.
*   **CPU:** Installs standard CPU-only PyTorch if no accelerator is found.

### Intel XPU (Intel Graphics)
For Intel integrated graphics and Arc GPUs.

**Important:** Check [compatibility requirements](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html) first:
```bash
make setup-xpu
```

**Notes:**
- If `.venv` already exists, setup skip creation.
- Remove existing environment first with `make clean` if you need to recreate it
- Setup automatically installs:
  - PyTorch (hardware-specific version)
  - ComfyUI dependencies (`external/ComfyUI/requirements.txt`)
  - ComfyUI Manager dependencies (`external/ComfyUI/manager_requirements.txt`)
  - Project dependencies (`project_requirements.txt`, if present)
  - Current project in editable mode

## Running ComfyUI

### Auto-Detect Hardware (Recommended)
Automatically detects CUDA or MPS (Mac) GPU and uses it, falls back to CPU if unavailable:
```bash
make run [FLAGS="..."]
```

### Force CPU Mode
Explicitly run with CPU (useful for testing or troubleshooting):
```bash
make run-cpu [FLAGS="..."]
```

### Force GPU Mode
Run with GPU acceleration (CUDA only, will fail if CUDA is not available):
```bash
make run-gpu [FLAGS="..."]
```

**What Gets Launched:**
- Script: `external/ComfyUI/main.py`
- Default flags: `--enable-manager --preview-method latent2rgb`
- Additional flags can be passed onto ComfyUI via the `FLAGS` variable, e.g.:
  - Listen on an IP address: `FLAGS="--listen [IP]"`
- CPU mode adds: `--cpu`

## Updating

### Update Everything
Updates both the ComfyUI submodule and all Python dependencies:
```bash
make update
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
├── custom_nodes/                   # ComfyUI custom nodes
├── project_requirements.txt        # project dependencies
├── Makefile                        # Build automation
└── pyproject.toml                  # Project configuration
```

## Notes
- **Lock Files:** If you want reproducible installs, consider tracking `uv.lock` in version control; otherwise keep it in `.gitignore`
- **First Setup:** Always run `make status` after initial setup to verify installation

## Common Workflows

### First Time Setup
```bash
git clone --recursive <repo-url>
cd <repo-name>
make setup   # Auto-detects GPU/CPU and installs
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
make setup           # Re-run setup (detects hardware)
# Or for forced CPU:
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