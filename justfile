# ──────────────────────────────────────────────────────────────
# ComfyUI Project Manager — Justfile
# ──────────────────────────────────────────────────────────────
# Equivalent of the Makefile, rewritten to leverage Justfile features:
#   • No .PHONY needed — recipes are commands, never file targets
#   • `just --list` gives a beautiful help menu for free (via [doc] + [group])
#   • Built-in os()/arch() functions replace $(shell uname …)
#   • Recipe parameters with defaults replace Make's FLAGS ?=
#   • [confirm] attribute on destructive recipes asks before proceeding
#   • [private] hides helper recipes from `just --list`
#   • Backtick evaluation for computed variables
#   • Recipes call each other directly (no $(MAKE) re-invocation)
#   • Shebang recipes: multi-line bash without backslash-continuations
# ──────────────────────────────────────────────────────────────
# Use bash for consistent behavior everywhere

set shell := ["bash", "-euo", "pipefail", "-c"]

# ── Colours ──────────────────────────────────────────────────
# In just, '\\' is interpreted as a literal '\'.
# So '\\033[0;34m' becomes the string \033[0;34m which echo -e renders.
# All usages MUST be inside double quotes to avoid ';' splitting the command.

BLUE := '\\033[0;34m'
CYAN := '\\033[0;36m'
GREEN := '\\033[0;32m'
YELLOW := '\\033[0;33m'
PURPLE := '\\033[0;35m'
RED := '\\033[0;31m'
BOLD := '\\033[1m'
NC := '\\033[0m'

# ── Detected system info (evaluated once, lazily) ───────────
# `os()` and `arch()` are built-in just functions — no shell needed.

OS := os()
ARCH := arch()
HAS_NVIDIA := `command -v nvidia-smi > /dev/null 2>&1 && echo "True" || echo "False"`

# ── Paths / flags ───────────────────────────────────────────

VENV_SENTINEL := ".venv/pyvenv.cfg"
COMFY_FLAGS := "--enable-manager --preview-method latent2rgb"
FILTERED_COMFY_REQ := ".venv/comfyui_requirements.no_torch.txt"

# ── Default recipe (runs when you just type `just`) ─────────

[doc("Show all available recipes")]
help:
    @just --list --unsorted

# ══════════════════════════════════════════════════════════════
#  Setup
# ══════════════════════════════════════════════════════════════

[doc("Auto-detect hardware and install everything")]
[group("setup")]
setup: _check-comfyui _ensure-venv
    @echo -e "{{ BLUE }}{{ BOLD }}Setting up environment for detected hardware...{{ NC }}"
    just install-torch
    just install-deps
    @echo -e "{{ GREEN }}Setup complete! Run 'just status' to verify.{{ NC }}"

[doc("Force setup for Intel XPU (special case for Intel Graphics, check compatibility first: https://docs.pytorch.org/docs/main/notes/get_start_xpu.html)")]
[group("setup")]
setup-xpu: _check-comfyui _ensure-venv
    @echo -e "{{ BLUE }}{{ BOLD }}Setting up environment for Intel XPU...{{ NC }}"
    uv pip install torch torchvision torchaudio intel-extension-for-pytorch
    just install-deps
    @echo -e "{{ GREEN }}Setup complete for XPU!{{ NC }}"

# ══════════════════════════════════════════════════════════════
#  Run
# ══════════════════════════════════════════════════════════════

[doc("Run ComfyUI (auto-detects GPU/CPU). Extra flags: just run '--listen'")]
[group("run")]
run *FLAGS: _ensure-venv
    set -euo pipefail
    echo -e "{{ BLUE }}{{ BOLD }}Launching ComfyUI...{{ NC }}"
    if uv run python -c "import torch; exit(0 if torch.cuda.is_available() or torch.backends.mps.is_available() else 1)" 2>/dev/null; then
        echo -e "{{ GREEN }}GPU acceleration detected{{ NC }}"
        uv run external/ComfyUI/main.py {{ COMFY_FLAGS }} {{ FLAGS }}
    else
        echo -e "{{ YELLOW }}Running on CPU{{ NC }}"
        uv run external/ComfyUI/main.py --cpu {{ COMFY_FLAGS }} {{ FLAGS }}
    fi

[doc("Force run on CPU. Extra flags: just run-cpu '--listen'")]
[group("run")]
run-cpu *FLAGS: _ensure-venv
    @echo -e "{{ BLUE }}Launching ComfyUI (CPU Forced)...{{ NC }}"
    uv run external/ComfyUI/main.py --cpu {{ COMFY_FLAGS }} {{ FLAGS }}

[doc("Force run on GPU. Extra flags: just run-gpu '--listen'")]
[group("run")]
run-gpu *FLAGS: _ensure-venv
    #!/usr/bin/env bash
    set -euo pipefail
    echo -e "{{ BLUE }}Launching ComfyUI (GPU Forced)...{{ NC }}"
    if ! uv run python -c "import torch; exit(0 if torch.cuda.is_available() or torch.backends.mps.is_available() else 1)" 2>/dev/null; then
        echo -e "{{ RED }}GPU not available!{{ NC }}"
        exit 1
    fi
    uv run external/ComfyUI/main.py {{ COMFY_FLAGS }} {{ FLAGS }}

# ══════════════════════════════════════════════════════════════
#  Maintenance
# ══════════════════════════════════════════════════════════════

[doc("Show detailed environment status")]
[group("maintenance")]
status:
    #!/usr/bin/env bash
    set -euo pipefail
    echo -e "{{ BLUE }}{{ BOLD }}=== Environment Status ==={{ NC }}"
    # Venv check
    echo -n "Venv: "
    if [ -f "{{ VENV_SENTINEL }}" ]; then
        echo -e "{{ GREEN }}{{ BOLD }}Active{{ NC }}"
    else
        echo -e "{{ RED }}{{ BOLD }}Missing{{ NC }}"
    fi
    # ComfyUI check
    echo -n "ComfyUI: "
    if [ -f "external/ComfyUI/.git" ] || [ -d "external/ComfyUI/.git" ]; then
        echo -e "{{ GREEN }}{{ BOLD }}Present{{ NC }}"
    else
        echo -e "{{ RED }}{{ BOLD }}Missing{{ NC }}"
    fi
    echo ""
    # PyTorch device info (only when venv exists)
    if [ -f "{{ VENV_SENTINEL }}" ]; then
        echo "PyTorch Device Check:"
        uv run python -c "\
    import torch; \
    print(f'  Version: {torch.__version__}'); \
    print(f'  MPS:     {torch.backends.mps.is_available()}'); \
    print(f'  CUDA:    {torch.cuda.is_available()}'); \
    cuda = torch.cuda.is_available(); \
    print(f'  GPU Count: {torch.cuda.device_count()}') if cuda else None; \
    print(f'  CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else \"N/A\"}') if cuda else None; \
    "
    fi

[doc("Update ComfyUI and all dependencies")]
[group("maintenance")]
update: _ensure-venv
    @echo -e "{{ BLUE }}Updating ComfyUI...{{ NC }}"
    git submodule update --init --recursive
    @echo -e "{{ BLUE }}Updating PyTorch for current hardware...{{ NC }}"
    just install-torch
    @echo -e "{{ BLUE }}Updating Python packages...{{ NC }}"
    just install-deps
    @echo -e "{{ GREEN }}{{ BOLD }}Update complete!{{ NC }}"

[confirm("This will delete .venv and all __pycache__ dirs. Continue?")]
[doc("Remove virtual environment and cache")]
[group("maintenance")]
clean:
    @echo -e "{{ YELLOW }}Cleaning environment...{{ NC }}"
    rm -rf .venv
    find . -type d -name "__pycache__" -exec rm -rf {} +
    @echo -e "{{ GREEN }}{{ BOLD }}Clean complete{{ NC }}"

# ══════════════════════════════════════════════════════════════
#  Internal / helper recipes  (hidden from `just --list`)
# ══════════════════════════════════════════════════════════════

# Create the venv only if it doesn't already exist.
[private]
_ensure-venv:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -f "{{ VENV_SENTINEL }}" ]; then
        echo -e "{{ BLUE }}Creating virtual environment...{{ NC }}"
        uv venv
    fi

# Install Python deps (ComfyUI + manager + project), filtering out torch lines.
[private]
install-deps: _ensure-venv
    #!/usr/bin/env bash
    set -euo pipefail
    echo -e "{{ BLUE }}Installing ComfyUI dependencies...{{ NC }}"
    grep -Ev '^(torch|torchvision|torchaudio)([<>=~!].*)?$' external/ComfyUI/requirements.txt > {{ FILTERED_COMFY_REQ }}
    uv pip install -r {{ FILTERED_COMFY_REQ }}
    if [ -f "external/ComfyUI/manager_requirements.txt" ]; then
        uv pip install -r external/ComfyUI/manager_requirements.txt
    fi
    if [ -f "project_requirements.txt" ]; then
        echo "Installing project dependencies..."
        uv pip install -r project_requirements.txt
    fi
    uv pip install -e .

# Install the right PyTorch variant based on OS / GPU.

# Uses just's built-in `os()` instead of shelling out to `uname`.
[private]
install-torch: _ensure-venv
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{ OS }}" = "macos" ]; then
        echo "Detected macOS. Installing PyTorch (MPS supported)..."
        uv pip install torch torchvision torchaudio
    elif [ "{{ HAS_NVIDIA }}" = "True" ]; then
        echo "Detected NVIDIA GPU. Installing PyTorch (CUDA 13.0)..."
        uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
    else
        echo -e "{{ YELLOW }}No NVIDIA GPU detected. Installing PyTorch (CPU version)...{{ NC }}"
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

# Ensure ComfyUI git submodule is present and up to date.
[private]
_check-comfyui:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Checking ComfyUI submodule..."
    if [ ! -d "external/ComfyUI" ]; then
        echo "ComfyUI directory not found. Creating external directory..."
        mkdir -p external
    fi
    if [ ! -f "external/ComfyUI/.git" ] && [ ! -d "external/ComfyUI/.git" ]; then
        echo "ComfyUI submodule not initialized. Checking if it's configured..."
        if git config --file .gitmodules --get submodule.external/ComfyUI.url > /dev/null 2>&1; then
            echo "Submodule configured, initializing..."
            git submodule update --init --recursive
        else
            echo -e "{{ YELLOW }}{{ BOLD }}Submodule not configured in .gitmodules{{ NC }}"
            echo "Initializing anyway..."
            git submodule update --init --recursive
        fi
    else
        echo "ComfyUI submodule exists, updating..."
        git submodule update --init --recursive
    fi
    echo -e "{{ GREEN }}{{ BOLD }}✓ ComfyUI submodule ready!{{ NC }}"
