.PHONY: help status update clean run run-cpu run-gpu install-deps install-torch setup setup-xpu

BLUE := \033[0;34m
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
PURPLE := \033[0;35m
RED := \033[0;31m
BOLD := \033[1m
NC := \033[0m

OS := $(shell uname -s)
ARCH := $(shell uname -m)
HAS_NVIDIA := $(shell command -v nvidia-smi > /dev/null 2>&1 && echo "True" || echo "False")

VENV_SENTINEL := .venv/pyvenv.cfg

COMFY_FLAGS := --enable-manager --preview-method latent2rgb
FLAGS ?=

FILTERED_COMFY_REQ := .venv/comfyui_requirements.no_torch.txt

SOURCE_DIR = src/custom_nodes
COMFY_TARGET = external/ComfyUI/custom_nodes
PY_FILES = $(shell find $(SOURCE_DIR) -type f -name "*.py" ! -name "__init__.py")

help:
	@echo "$(CYAN)$(BOLD)============ ComfyUI Project Manager ============$(NC)"
	@echo ""
	@echo "Detected System: $(BLUE)$(BOLD)$(OS) ($(ARCH))$(NC) | NVIDIA: $(BLUE)$(BOLD)$(HAS_NVIDIA)$(NC)"
	@echo ""
	@echo "$(PURPLE)$(BOLD)Setup:$(NC)"
	@echo "  $(GREEN)setup$(NC)          - Auto-detect hardware and install everything"
	@echo "  $(GREEN)setup-xpu$(NC)      - Force setup for Intel XPU (special case for Intel Graphics, check compatibility first: https://docs.pytorch.org/docs/main/notes/get_start_xpu.html)"
	@echo ""
	@echo "$(PURPLE)$(BOLD)Run:$(NC)"
	@echo "  $(GREEN)run$(NC)            - Run ComfyUI (auto-detects GPU/CPU)"
	@echo "  $(GREEN)run-cpu$(NC)        - Force run on CPU"
	@echo "  $(GREEN)run-gpu$(NC)        - Force run on GPU"
	@echo "                 * Use FLAGS=\"...\" to pass flags (e.g. make run FLAGS=\"--listen\")"
	@echo ""
	@echo "$(PURPLE)$(BOLD)Maintenance:$(NC)"
	@echo "  $(GREEN)status$(NC)         - Show detailed environment status"
	@echo "  $(GREEN)update$(NC)         - Update ComfyUI and all dependencies"
	@echo "  $(GREEN)clean$(NC)          - Remove virtual environment and cache"


# create the venv only if it doesn't exist
$(VENV_SENTINEL):
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	@uv venv

install-deps: $(VENV_SENTINEL)
	@echo "$(BLUE)Installing ComfyUI dependencies...$(NC)"
	@grep -Ev '^(torch|torchvision|torchaudio)([<>=~!].*)?$$' external/ComfyUI/requirements.txt > $(FILTERED_COMFY_REQ)
	@uv pip install -r $(FILTERED_COMFY_REQ)
	@if [ -f "external/ComfyUI/manager_requirements.txt" ]; then \
		uv pip install -r external/ComfyUI/manager_requirements.txt; \
	fi
	@if [ -f "project_requirements.txt" ]; then \
		echo "Installing project dependencies..."; \
		uv pip install -r project_requirements.txt; \
	fi
	@uv pip install -e .

install-cuda:
	@echo "$(BLUE)Checking for CUDA support...$(NC)"
	@if [ "$(OS)" = "Darwin" ]; then \
		echo "Detected macOS. Installing PyTorch (MPS supported)..."; \
		uv pip install torch torchvision torchaudio; \
	elif [ "$(HAS_NVIDIA)" = "True" ]; then \
		echo "Detected NVIDIA GPU. Installing PyTorch (CUDA 13.0)..."; \
		uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130; \
	else \
		echo "$(YELLOW)No NVIDIA GPU detected. Installing PyTorch (CPU version)...$(NC)"; \
		uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
	fi

setup: check-comfyui $(VENV_SENTINEL)
	@echo "$(BLUE)$(BOLD)Setting up environment for detected hardware...$(NC)"
	@$(MAKE) install-cuda
	@$(MAKE) install-deps
	@echo "$(GREEN)Setup complete! Run 'make status' to verify.$(NC)"

# Specialized setup for Intel XPU (hard to auto-detect reliably)
setup-xpu: check-comfyui $(VENV_SENTINEL)
	@echo "$(BLUE)$(BOLD)Setting up environment for Intel XPU...$(NC)"
	@uv pip install torch torchvision torchaudio intel-extension-for-pytorch
	@$(MAKE) install-deps
	@echo "$(GREEN)Setup complete for XPU!$(NC)"

setup-CI: $(VENV_SENTINEL)
	@echo "$(BLUE)Setting up environment for CI/CD...$(NC)"
	@$(MAKE) install-cuda
	@uv pip install -r ci-requirements.txt
	@echo "$(GREEN)CI/CD setup complete!$(NC)"

run: $(VENV_SENTINEL)
	@echo "$(BLUE)$(BOLD)Launching ComfyUI...$(NC)"
	@if uv run python -c "import torch; exit(0 if torch.cuda.is_available() or torch.backends.mps.is_available() else 1)" 2>/dev/null; then \
		echo "$(GREEN)GPU acceleration detected$(NC)"; \
		uv run external/ComfyUI/main.py $(COMFY_FLAGS) $(FLAGS); \
	else \
		echo "$(YELLOW)Running on CPU$(NC)"; \
		uv run external/ComfyUI/main.py --cpu $(COMFY_FLAGS) $(FLAGS); \
	fi

run-cpu: $(VENV_SENTINEL) link-nodes
	@echo "$(BLUE)Launching ComfyUI (CPU Forced)...$(NC)"
	uv run external/ComfyUI/main.py --cpu $(COMFY_FLAGS) $(FLAGS)

run-gpu: $(VENV_SENTINEL) link-nodes
	@echo "$(BLUE)Launching ComfyUI (GPU Forced)...$(NC)"
	@if ! uv run python -c "import torch; exit(0 if torch.cuda.is_available() or torch.backends.mps.is_available() else 1)" 2>/dev/null; then \
		echo "$(RED)GPU not available!$(NC)"; exit 1; \
	fi
	uv run external/ComfyUI/main.py $(COMFY_FLAGS) $(FLAGS)

status:
	@echo "$(BLUE)$(BOLD)=== Environment Status ===$(NC)"
	@echo -n "Venv: "
	@if [ -f "$(VENV_SENTINEL)" ]; then echo "$(GREEN)$(BOLD)Active$(NC)"; else echo "$(RED)$(BOLD)Missing$(NC)"; fi
	@echo -n "ComfyUI: "
	@if [ -f "external/ComfyUI/.git" ] || [ -d "external/ComfyUI/.git" ]; then echo "$(GREEN)$(BOLD)Present$(NC)"; else echo "$(RED)$(BOLD)Missing$(NC)"; fi
	@echo ""
	@if [ -f "$(VENV_SENTINEL)" ]; then \
		echo "PyTorch Device Check:"; \
		uv run python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  MPS:     {torch.backends.mps.is_available()}'); print(f'  CUDA:    {torch.cuda.is_available()}'); print(f'  GPU Count: {torch.cuda.device_count()}' if torch.cuda.is_available() else ''); print(f'  CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else \"N/A\"}' if torch.cuda.is_available() else '')"; \
	fi

check-comfyui:
	@echo "Checking ComfyUI submodule..."
	@if [ ! -d "external/ComfyUI" ]; then \
		echo "ComfyUI directory not found. Creating external directory..."; \
		mkdir -p external; \
	fi
	@if [ ! -f "external/ComfyUI/.git" ] && [ ! -d "external/ComfyUI/.git" ]; then \
		echo "ComfyUI submodule not initialized. Checking if it's configured..."; \
		if git config --file .gitmodules --get submodule.external/ComfyUI.url > /dev/null 2>&1; then \
			echo "Submodule configured, initializing..."; \
			git submodule update --init --recursive; \
		else \
			echo "$(YELLOW)$(BOLD)Submodule not configured in .gitmodules$(NC)"; \
			echo "Initializing anyway..."; \
			git submodule update --init --recursive; \
		fi \
	else \
		echo "ComfyUI submodule exists, updating..."; \
		git submodule update --init --recursive; \
	fi
	@echo "$(GREEN)$(BOLD)✓ ComfyUI submodule ready!$(NC)"

update: $(VENV_SENTINEL)
	@echo "$(BLUE)Updating ComfyUI...$(NC)"
	git submodule update --init --recursive
	@echo "$(BLUE)Updating PyTorch for current hardware...$(NC)"
	@$(MAKE) install-torch
	@echo "$(BLUE)Updating Python packages...$(NC)"
	@$(MAKE) install-deps
	@echo "$(GREEN)$(BOLD)Update complete!$(NC)"

clean: remove-link-nodes
	@echo "$(YELLOW)Cleaning environment...$(NC)"
	@rm -rf .venv
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "$(GREEN)$(BOLD)Clean complete$(NC)"