# Modular Image Processing for ComfyUI

This repository provides a modular image-processing pipeline integrated into ComfyUI through custom nodes located in `src/custom_nodes`.

The project is managed primarily with `Makefile` targets and uses `uv` for environment/package management.

## Licence
This project is licensed under GNU GPL v3 - see the [LICENSE](LICENSE) file for details.

## Prerequisites

- Git with submodule and [Git LFS](https://git-lfs.com/) support
- `uv` package manager: https://github.com/astral-sh/uv
- Python `>= 3.13`

Optional hardware acceleration:
- NVIDIA GPU (CUDA 13.0 wheel channel)
- Apple Silicon (MPS)
- Intel GPU (XPU, via dedicated setup target)

CPU-only execution is supported.

## Clone the Repository

Recommended clone (includes submodules):

```bash
git clone --recursive <repo-url>
cd <repo-name>
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Setup

### Standard setup (auto-detect hardware)

Creates `.venv`, installs hardware-appropriate PyTorch, ComfyUI dependencies, optional project requirements, and installs this package in editable mode.

```bash
make setup
```

### Intel XPU setup

Use this only for Intel graphics/XPU environments - check [compatibility requirements](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html) first.

```bash
make setup-xpu
```

## Run ComfyUI

Main run target (auto-select GPU/CPU):

```bash
make run
```

Force CPU:

```bash
make run-cpu
```

Force GPU (fails if unavailable):

```bash
make run-gpu
```

Pass additional ComfyUI flags via `FLAGS`:

```bash
make run FLAGS="--listen 0.0.0.0 --port 8188"
```

What `make run*` executes:
- Launches `external/ComfyUI/main.py`
- Uses default flags `--enable-manager --preview-method latent2rgb`
- Adds `--cpu` in CPU mode
- Links files from `src/custom_nodes/**/*.py` into `external/ComfyUI/custom_nodes/` before launch

## Status, Update, Cleanup

Check environment/submodule/torch status:

```bash
make status
```

Update submodule and reinstall dependencies:

```bash
make update
```

Clean virtualenv and Python cache files:

```bash
make clean
```

## Testing

Run tests with:

```bash
uv run pytest
```

Current test suite (under `tests/`) includes coverage for demosaicing and raw processing components.

## Project Layout

```text
.
├── src/
│   ├── algorithms/                 # Core image-processing algorithms
│   └── custom_nodes/               # ComfyUI node implementations
├── tests/                          # Pytest test suite
├── external/
│   └── ComfyUI/                    # ComfyUI submodule
├── project_requirements.txt        # Project-specific dependencies
├── ci-requirements.txt             # CI-only dependencies
├── pyproject.toml                  # Package/test configuration
└── Makefile                        # Main developer entrypoint
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

## Getting Help
Display all available commands:
```bash
make help
```
Or simply:
```bash
make
```
