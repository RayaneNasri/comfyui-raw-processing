.PHONY: help check-venv check-torch check-comfyui setup-cuda130 setup-cpu setup-xpu update clean run-cpu run-gpu status

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
BOLD := \033[1m
NC := \033[0m # No Color

help:
	@echo "$(BLUE)$(BOLD)ComfyUI Project Manager$(NC)"
	@echo ""
	@echo "$(GREEN)$(BOLD)Status & Checks:$(NC)"
	@echo "  status         - Show current environment status"
	@echo "  check-comfyui  - Check if ComfyUI is correctly cloned"
	@echo "  check-venv     - Check if virtual environment exists"
	@echo "  check-torch    - Check if PyTorch is installed and detect device"
	@echo ""
	@echo "$(GREEN)$(BOLD)Setup:$(NC)"
	@echo "  setup-cuda130  - Setup with CUDA 13.0 (for NVIDIA GPU users)"
	@echo "  setup-cpu      - Setup with CPU only"
	@echo "  setup-xpu      - Setup with Intel XPU (for Intel Graphics, check compatibility first: https://docs.pytorch.org/docs/main/notes/get_start_xpu.html)"
	@echo ""
	@echo "$(GREEN)$(BOLD)Maintenance:$(NC)"
	@echo "  update         - Update all dependencies"
	@echo "  update-comfyui - Update ComfyUI submodule only"
	@echo "  clean          - Remove virtual environment"
	@echo ""
	@echo "$(GREEN)$(BOLD)Run:$(NC)"
	@echo "  run            - Auto-detect hardware and run ComfyUI"
	@echo "  run-cpu        - Force run with CPU"
	@echo "  run-gpu        - Force run with GPU"

status:
	@echo "$(BLUE)$(BOLD)=== Environment Status ===$(NC)"
	@echo -n "Virtual Environment: "
	@if [ -d ".venv" ]; then \
		echo "$(GREEN)✓ Exists$(NC)"; \
	else \
		echo "$(RED)✗ Not found$(NC) (run 'make setup-cuda130' or 'make setup-cpu')"; \
	fi
	@echo ""
	@echo -n "ComfyUI Submodule: "
	@if [ -f "external/ComfyUI/.git" ] || [ -d "external/ComfyUI/.git" ]; then \
		echo "$(GREEN)✓ Cloned$(NC)"; \
	else \
		echo "$(RED)✗ Not cloned$(NC) (run 'make check-comfyui')"; \
	fi
	@echo ""
	@echo -n "PyTorch: "
	@if [ -d ".venv" ]; then \
		if uv run python -c "import torch" 2>/dev/null; then \
			echo "$(GREEN)✓ Installed$(NC)"; \
			uv run python -c "import torch; print('  Version:', torch.__version__)"; \
			uv run python -c "import torch; print('  CUDA Available:', torch.cuda.is_available())"; \
			if uv run python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then \
				uv run python -c "import torch; print('  CUDA Version:', torch.version.cuda)"; \
				uv run python -c "import torch; print('  GPU Count:', torch.cuda.device_count())"; \
				uv run python -c "import torch; print('  GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A')"; \
			fi; \
		else \
			echo "$(RED)✗ Not installed$(NC)"; \
		fi \
	else \
		echo "$(YELLOW)⚠ Cannot check (no venv)$(NC)"; \
	fi
	@echo ""

check-venv:
	@if [ ! -d ".venv" ]; then \
		echo "$(YELLOW)Virtual environment not found.$(NC)"; \
		echo "Please run one of:"; \
		echo "  make setup-cuda130  (for NVIDIA GPU)"; \
		echo "  make setup-cpu      (for CPU only)"; \
		echo "  make setup-xpu      (for Intel GPU)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Virtual environment exists$(NC)"

check-torch: check-venv
	@echo "Checking PyTorch installation..."
	@if ! uv run python -c "import torch" 2>/dev/null; then \
		echo "$(RED)✗ PyTorch not installed!$(NC)"; \
		echo "Please run one of:"; \
		echo "  make setup-cuda130  (for NVIDIA GPU)"; \
		echo "  make setup-cpu      (for CPU only)"; \
		echo "  make setup-xpu      (for Intel GPU)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ PyTorch is installed$(NC)"
	@uv run python -c "import torch; print('Version:', torch.__version__)"
	@uv run python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

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
			echo "$(YELLOW)Submodule not configured in .gitmodules$(NC)"; \
			echo "Initializing anyway..."; \
			git submodule update --init --recursive; \
		fi \
	else \
		echo "ComfyUI submodule exists, updating..."; \
		git submodule update --init --recursive; \
	fi
	@echo "$(GREEN)✓ ComfyUI submodule ready!$(NC)"

setup-cuda130: check-comfyui
	@echo "$(BLUE)$(BOLD)Setting up environment for CUDA 13.0...$(NC)"
	@if [ -d ".venv" ]; then \
		echo "$(YELLOW)Virtual environment already exists. Remove it first with 'make clean' if you want to recreate it.$(NC)"; \
		exit 1; \
	fi
	uv venv
	@echo "Installing PyTorch with CUDA 13.0 support..."
	uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
	@echo "Installing ComfyUI dependencies..."
	uv pip install -r external/ComfyUI/requirements.txt
	@if [ -f "external/ComfyUI/manager_requirements.txt" ]; then \
		uv pip install -r external/ComfyUI/manager_requirements.txt; \
	fi
	@if [ -f "requirements_project.txt" ]; then \
		echo "Installing project dependencies..."; \
		uv pip install -r requirements_project.txt; \
	fi
	uv pip install -e .
	@echo "$(GREEN)✓ Setup complete for CUDA 13.0!$(NC)"
	@echo "Run 'make status' to verify installation"

setup-cpu: check-comfyui
	@echo "$(BLUE)$(BOLD)Setting up environment for CPU...$(NC)"
	@if [ -d ".venv" ]; then \
		echo "$(YELLOW)Virtual environment already exists. Remove it first with 'make clean' if you want to recreate it.$(NC)"; \
		exit 1; \
	fi
	uv venv
	@echo "Installing PyTorch (CPU version)..."
	uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	@echo "Installing ComfyUI dependencies..."
	uv pip install -r external/ComfyUI/requirements.txt
	@if [ -f "external/ComfyUI/manager_requirements.txt" ]; then \
		uv pip install -r external/ComfyUI/manager_requirements.txt; \
	fi
	@if [ -f "requirements_project.txt" ]; then \
		echo "Installing project dependencies..."; \
		uv pip install -r requirements_project.txt; \
	fi
	uv pip install -e .
	@echo "$(GREEN)✓ Setup complete for CPU!$(NC)"
	@echo "Run 'make status' to verify installation"

setup-xpu: check-comfyui
	@echo "$(BLUE)$(BOLD)Setting up environment for Intel XPU...$(NC)"
	@if [ -d ".venv" ]; then \
		echo "$(YELLOW)Virtual environment already exists. Remove it first with 'make clean' if you want to recreate it.$(NC)"; \
		exit 1; \
	fi
	uv venv
	@echo "Installing PyTorch with Intel XPU support..."
	uv pip install torch torchvision torchaudio intel-extension-for-pytorch
	@echo "Installing ComfyUI dependencies..."
	uv pip install -r external/ComfyUI/requirements.txt
	@if [ -f "external/ComfyUI/manager_requirements.txt" ]; then \
		uv pip install -r external/ComfyUI/manager_requirements.txt; \
	fi
	@if [ -f "requirements_project.txt" ]; then \
		echo "Installing project dependencies..."; \
		uv pip install -r requirements_project.txt; \
	fi
	uv pip install -e .
	@echo "$(GREEN)✓ Setup complete for Intel XPU!$(NC)"
	@echo "Run 'make status' to verify installation"

update: check-venv
	@echo "$(BLUE)$(BOLD)Updating all dependencies...$(NC)"
	@echo "Updating ComfyUI submodule..."
	git submodule update --remote --merge external/ComfyUI
	@echo "Updating Python packages..."
	uv pip install --upgrade -r external/ComfyUI/requirements.txt
	@if [ -f "external/ComfyUI/manager_requirements.txt" ]; then \
		uv pip install --upgrade -r external/ComfyUI/manager_requirements.txt; \
	fi
	@if [ -f "requirements_project.txt" ]; then \
		uv pip install --upgrade -r requirements_project.txt; \
	fi
	uv pip install --upgrade -e .
	@echo "$(GREEN)✓ Update complete!$(NC)"

update-comfyui:
	@echo "$(BLUE)$(BOLD)Updating ComfyUI submodule...$(NC)"
	git submodule update --remote --merge external/ComfyUI
	@echo "$(GREEN)✓ ComfyUI updated!$(NC)"
	@echo "Run 'make update' to also update Python dependencies"

clean:
	@echo "$(YELLOW)Removing virtual environment...$(NC)"
	@if [ -d ".venv" ]; then \
		rm -rf .venv; \
		echo "$(GREEN)✓ Virtual environment removed$(NC)"; \
	else \
		echo "No virtual environment to remove"; \
	fi

run: check-venv check-torch
	@echo "$(BLUE)$(BOLD)Auto-detecting hardware and launching ComfyUI...$(NC)"
	@if uv run python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then \
		echo "$(GREEN)✓ CUDA detected, running with GPU$(NC)"; \
		uv run external/ComfyUI/main.py --enable-manager --preview-method latent2rgb; \
	else \
		echo "$(YELLOW)⚠ No CUDA detected, running with CPU$(NC)"; \
		uv run external/ComfyUI/main.py --cpu --enable-manager --preview-method latent2rgb; \
	fi

run-cpu: check-venv check-torch
	@echo "$(BLUE)$(BOLD)Launching ComfyUI (CPU mode)...$(NC)"
	uv run external/ComfyUI/main.py --cpu --enable-manager --preview-method latent2rgb

run-gpu: check-venv check-torch
	@echo "$(BLUE)$(BOLD)Launching ComfyUI (GPU mode)...$(NC)"
	@if ! uv run python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then \
		echo "$(RED)✗ CUDA not available!$(NC)"; \
		echo "Either install CUDA support or use 'make run-cpu'"; \
		exit 1; \
	fi
	uv run external/ComfyUI/main.py --enable-manager --preview-method latent2rgb