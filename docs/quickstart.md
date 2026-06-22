# Quickstart Guide

Get up and running with ISP (Image Signal Processing) nodes in ComfyUI in under 5 minutes.

## What This Does

Modular image processing pipeline for raw camera files → finished JPEG. Choose and combine processing nodes:

- **RAW Reading** (Sony `.arw` format)
- **Demosaicing** (Bayer → RGB)
- **White Balance** (multiple methods)
- **Exposure & Gamma** (color science)

Usable in **ComfyUI UI** or **Python API**.

## Setup (5 min)

```bash
# Clone with submodules (if you did not clone the current project)
git clone --recursive <repo-url>
cd image-processing

# Auto-detect GPU/CPU and install
make setup

# Launch ComfyUI
make run
```

Done. ComfyUI runs at `http://localhost:8188`

## First Workflow in ComfyUI (2 min)

1. **Add nodes:**
   - Right-click → Add Node → `image` → "Read RAW Sensor"
   - Add → `image` → "Malvar-He-Cutler Demosaicing"
   - Add → `image` → "Gray World White Balance"
   - Add → `image` → "Gamma Correction"
   - Add → `image/export` → "Save JPEG (Custom Path)"

2. **Connect:**
   ```
   Read RAW Sensor
       ↓
   Malvar-He-Cutler Demosaicing
       ↓
   Gray World White Balance
       ↓
   Gamma Correction
       ↓
   Save JPEG
   ```

3. **Set paths:**
   - Read RAW: Set `image_path` to your `.arw` file
   - Save JPEG: Set `folder_path` to `./output` and `filename`

4. **Execute:** Click "Queue Prompt" → Output saved as JPEG

## First Workflow in Python (5 min)

```python
import torch
from pathlib import Path
import sys
sys.path.insert(0, 'src')

from algorithms.raw.reader import read_raw_sensor_data
from algorithms.black_light_subtraction.black_light_subtraction import linearize_raw
from algorithms.demosaicing._malvar_he_culter import malvar_he_cutler_demosaicing
from algorithms.white_balance._gray_world import gw
from algorithms.gamma_correction.gamma_correction import gamma_correction
from algorithms.export.jpeg_export import export_jpeg

# Read raw file
raw_data, metadata = read_raw_sensor_data("input/image.arw")

# Create CFA pattern (RGGB)
H, W = raw_data.shape
cfa = torch.zeros((H, W), dtype=torch.int32)
cfa[0::2, 0::2] = 0  # R
cfa[0::2, 1::2] = 1  # G
cfa[1::2, 0::2] = 3  # G
cfa[1::2, 1::2] = 2  # B

# Process
black_levels = torch.tensor(metadata['black_levels'], dtype=torch.float32)
white_level = torch.tensor([metadata['white_level']], dtype=torch.float32)

bayer, _ = linearize_raw(raw_data, cfa, black_levels, white_level)
rgb = malvar_he_cutler_demosaicing(bayer, dx=0, dy=0)
wb = gw(rgb)
gamma = gamma_correction(wb, gamma=2.2, alpha=1.0)

# Export
Path('output').mkdir(exist_ok=True)
export_jpeg(gamma, 'output/result.jpg', quality=85)
print("✓ Saved to output/result.jpg")
```

Run: `python script.py`

## Next Steps

- **Explore node options:** See [usage_nodes.md](usage_nodes.md) for all 12 nodes, parameters, and decisions
- **Batch processing:** Check the Python API section in usage_nodes.md
- **White balance methods:** Compare Camera / Gray World / White Patch / Ground Truth approaches
- **Custom workflows:** Mix and match nodes; each is optional and independent

## Common Parameters Quick Ref

| What | How |
|------|-----|
| **Brighter image** | Exposure Compensation: +1 to +2 EV |
| **Darker image** | Exposure Compensation: -1 to -2 EV |
| **Different white balance** | Swap "Gray World" for "White Patch Reference" |
| **Faster (preview mode)** | Use "Bilinear Demosaicing" instead of Malvar-He-Cutler |
| **Better quality** | Use "Malvar-He-Cutler Demosaicing" |

## Resources

- **Full reference:** [Nodes Usage Documentation](usage_nodes.md)
- **Architecture:** [Project Architecture Documentation](architecture.md)
- **Setup details:** [README File](../README.md)
