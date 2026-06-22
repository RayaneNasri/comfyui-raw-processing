# Installation

## Prerequisites

- Git with submodule support
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

### First Time Setup Workflow

```bash
git clone --recursive <repo-url>
cd <repo-name>
make setup   # Auto-detects GPU/CPU and installs
make status  # Verify installation
make run     # Launch ComfyUI
```