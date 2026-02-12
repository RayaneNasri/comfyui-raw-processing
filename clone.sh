#!/bin/bash


COMFY_PATH="external/ComfyUI"

clear
echo "Checking for ComfyUI submodule..."

if [ ! -f "$COMFY_PATH/.git" ]; then

    echo "ComfyUI not found or not initialized. Initializing submodule..."
    git submodule update --init --recursive
    
    if [ $? -eq 0 ]; then
        echo "ComfyUI cloned successfully."
    else
        echo "Error: Failed to clone ComfyUI. Check your internet connection or git permissions."
        exit 1
    fi

else
    echo "ComfyUI is already present."
fi