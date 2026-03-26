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
	@printf "%b\n" "$(CYAN)$(BOLD)============ ComfyUI Project Manager ============$(NC)"
	@printf "%b\n" ""
	@printf "%b\n" "Detected System: $(BLUE)$(BOLD)$(OS) ($(ARCH))$(NC) | NVIDIA: $(BLUE)$(BOLD)$(HAS_NVIDIA)$(NC)"
	@printf "%b\n" ""
	@printf "%b\n" "$(PURPLE)$(BOLD)Setup:$(NC)"
	@printf "%b\n" "  $(GREEN)setup$(NC)          - Auto-detect hardware and install everything"
	@printf "%b\n" "  $(GREEN)setup-xpu$(NC)      - Force setup for Intel XPU (special case for Intel Graphics, check compatibility first: https://docs.pytorch.org/docs/main/notes/get_start_xpu.html)"
	@printf "%b\n" ""
	@printf "%b\n" "$(PURPLE)$(BOLD)Run:$(NC)"
	@printf "%b\n" "  $(GREEN)run$(NC)            - Run ComfyUI (auto-detects GPU/CPU)"
	@printf "%b\n" "  $(GREEN)run-cpu$(NC)        - Force run on CPU"
	@printf "%b\n" "  $(GREEN)run-gpu$(NC)        - Force run on GPU"
	@printf "%b\n" "                 * Use FLAGS=\"...\" to pass flags (e.g. make run FLAGS=\"--listen\")"
	@printf "%b\n" ""
	@printf "%b\n" "$(PURPLE)$(BOLD)Maintenance:$(NC)"
	@printf "%b\n" "  $(GREEN)status$(NC)         - Show detailed environment status"
	@printf "%b\n" "  $(GREEN)update$(NC)         - Update ComfyUI and all dependencies"
	@printf "%b\n" "  $(GREEN)clean$(NC)          - Remove virtual environment and cache"


# create the venv only if it doesn't exist
$(VENV_SENTINEL):
	@printf "%b\n" "$(BLUE)Creating virtual environment...$(NC)"
	@uv venv

install-deps: $(VENV_SENTINEL)
	@printf "%b\n" "$(BLUE)Installing ComfyUI dependencies...$(NC)"
	@grep -Ev '^(torch|torchvision|torchaudio)([<>=~!].*)?$$' external/ComfyUI/requirements.txt > $(FILTERED_COMFY_REQ)
	@uv pip install -r $(FILTERED_COMFY_REQ)
	@if [ -f "external/ComfyUI/manager_requirements.txt" ]; then \
		uv pip install -r external/ComfyUI/manager_requirements.txt; \
	fi
	@printf "%b\n" "Installing project dependencies..."
	@uv pip install -e .

install-torch: $(VENV_SENTINEL)
	@if [ "$(OS)" = "Darwin" ]; then \
		printf "%b\n" "Detected macOS. Installing PyTorch (MPS supported)..."; \
		uv pip install torch torchvision torchaudio; \
	elif [ "$(HAS_NVIDIA)" = "True" ]; then \
		printf "%b\n" "Detected NVIDIA GPU. Installing PyTorch (CUDA 13.0)..."; \
		uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130; \
	else \
		printf "%b\n" "$(YELLOW)No NVIDIA GPU detected. Installing PyTorch (CPU version)...$(NC)"; \
		uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
	fi


setup: check-comfyui $(VENV_SENTINEL)
	@printf "%b\n" "$(BLUE)$(BOLD)Setting up environment for detected hardware...$(NC)"
	@$(MAKE) install-torch
	@$(MAKE) install-deps
	@printf "%b\n" "$(GREEN)Setup complete! Run 'make status' to verify.$(NC)"

# Specialized setup for Intel XPU (hard to auto-detect reliably)
setup-xpu: check-comfyui $(VENV_SENTINEL)
	@printf "%b\n" "$(BLUE)$(BOLD)Setting up environment for Intel XPU...$(NC)"
	@uv pip install torch torchvision torchaudio intel-extension-for-pytorch
	@$(MAKE) install-deps
	@printf "%b\n" "$(GREEN)Setup complete for XPU!$(NC)"

setup-CI: $(VENV_SENTINEL)
	@printf "%b\n" "$(BLUE)Setting up environment for CI/CD...$(NC)"
	@$(MAKE) install-torch
	@uv export --only-group ci --no-emit-project -o .ci-reqs.txt
	@uv pip install -r .ci-reqs.txt
	@rm .ci-reqs.txt
	@printf "%b\n" "$(GREEN)CI/CD setup complete!$(NC)"

link-nodes: remove-link-nodes
	@printf "%b\n" "$(BLUE)$(BOLD)Linking all nodes files to $(COMFY_TARGET)...$(NC)"
	@for file in $(PY_FILES); do \
		FILENAME=$$(basename $$file); \
		ln -sf $(shell pwd)/$$file $(COMFY_TARGET)/$$FILENAME; \
	done
	@printf "%b\n" "$(GREEN)Linking completed$(NC)"

remove-link-nodes: 
	@printf "%b\n" "$(BLUE)$(BOLD)Cleaning nodes from $(COMFY_TARGET)...$(NC)"
	@find $(COMFY_TARGET) -maxdepth 1 \( -type l -o -type f \) \
		-name "*.py" \
		! -name "__init__.py" \
		! -name "websocket_image_save.py" \
		-delete
	@printf "%b\n" "$(GREEN)Nodes cleanup complete.$(NC)"

run: $(VENV_SENTINEL) link-nodes
	@printf "%b\n" "$(BLUE)$(BOLD)Launching ComfyUI...$(NC)"
	@if uv run python -c "import torch; exit(0 if torch.cuda.is_available() or torch.backends.mps.is_available() else 1)" 2>/dev/null; then \
		printf "%b\n" "$(GREEN)GPU acceleration detected$(NC)"; \
		uv run external/ComfyUI/main.py $(COMFY_FLAGS) $(FLAGS); \
	else \
		printf "%b\n" "$(YELLOW)Running on CPU$(NC)"; \
		uv run external/ComfyUI/main.py --cpu $(COMFY_FLAGS) $(FLAGS); \
	fi

run-cpu: $(VENV_SENTINEL) link-nodes
	@printf "%b\n" "$(BLUE)Launching ComfyUI (CPU Forced)...$(NC)"
	uv run external/ComfyUI/main.py --cpu $(COMFY_FLAGS) $(FLAGS)

run-gpu: $(VENV_SENTINEL) link-nodes
	@printf "%b\n" "$(BLUE)Launching ComfyUI (GPU Forced)...$(NC)"
	@if ! uv run python -c "import torch; exit(0 if torch.cuda.is_available() or torch.backends.mps.is_available() else 1)" 2>/dev/null; then \
		printf "%b\n" "$(RED)GPU not available!$(NC)"; exit 1; \
	fi
	uv run external/ComfyUI/main.py $(COMFY_FLAGS) $(FLAGS)

status:
	@printf "%b\n" "$(BLUE)$(BOLD)=== Environment Status ===$(NC)"
	@printf "%s" "Venv: "
	@if [ -f "$(VENV_SENTINEL)" ]; then printf "%b\n" "$(GREEN)$(BOLD)Active$(NC)"; else printf "%b\n" "$(RED)$(BOLD)Missing$(NC)"; fi
	@printf "%s" "ComfyUI: "
	@if [ -f "external/ComfyUI/.git" ] || [ -d "external/ComfyUI/.git" ]; then printf "%b\n" "$(GREEN)$(BOLD)Present$(NC)"; else printf "%b\n" "$(RED)$(BOLD)Missing$(NC)"; fi
	@printf "%b\n" ""
	@if [ -f "$(VENV_SENTINEL)" ]; then \
		printf "%b\n" "PyTorch Device Check:"; \
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
			printf "%b\n" "$(YELLOW)$(BOLD)Submodule not configured in .gitmodules$(NC)"; \
			echo "Initializing anyway..."; \
			git submodule update --init --recursive; \
		fi \
	else \
		echo "ComfyUI submodule exists, updating..."; \
		git submodule update --init --recursive; \
	fi
	@printf "%b\n" "$(GREEN)$(BOLD)✓ ComfyUI submodule ready!$(NC)"

update: $(VENV_SENTINEL)
	@printf "%b\n" "$(BLUE)Updating ComfyUI...$(NC)"
	git submodule update --init --recursive
	@printf "%b\n" "$(BLUE)Updating PyTorch for current hardware...$(NC)"
	@$(MAKE) install-torch
	@printf "%b\n" "$(BLUE)Updating Python packages...$(NC)"
	@$(MAKE) install-deps
	@printf "%b\n" "$(GREEN)$(BOLD)Update complete!$(NC)"

clean: remove-link-nodes
	@printf "%b\n" "$(YELLOW)Cleaning environment...$(NC)"
	@rm -rf .venv
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@printf "%b\n" "$(GREEN)$(BOLD)Clean complete$(NC)"