.PHONY: help setup-cuda130 setup-cpu setup-xpu check-comfyui

help:
	@echo "Check if ComfyUI repository is correctly cloned:"
	@echo "  check-comfyui"
	@echo "Setup ComfyUI environment, available targets:"
	@echo "  setup-cuda130  - Setup with CUDA 13.0 (for NVIDIA GPU users)"
	@echo "  setup-cpu      - Setup with CPU only"
	@echo "  setup-xpu      - Setup with Intel XPU (for Intel UHD Graphics, Xe Graphics and Arc)"

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
			echo "Submodule not configured. Cloning ComfyUI..."; \
			git submodule update --init --recursive; \
		fi \
	else \
		echo "ComfyUI submodule exists, updating..."; \
		git submodule update --init --recursive; \
	fi
	@echo "ComfyUI submodule ready!"

setup-cuda130: check-comfyui
	uv venv
	uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
	uv pip install -r external/ComfyUI/requirements.txt
	uv pip install -r external/ComfyUI/manager_requirements.txt
	uv pip install -e .
	@echo "Setup complete for CUDA 13.0!"

setup-cpu: check-comfyui
	uv venv
	uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	uv pip install -r external/ComfyUI/requirements.txt
	uv pip install -r external/ComfyUI/manager_requirements.txt
	uv pip install -e .
	@echo "Setup complete for CPU!"

setup-xpu: check-comfyui
	uv venv
	uv pip install torch torchvision torchaudio intel-extension-for-pytorch
	uv pip install -r external/ComfyUI/requirements.txt
	uv pip install -r external/ComfyUI/manager_requirements.txt
	uv pip install -e .
	@echo "Setup complete for Intel XPU!"