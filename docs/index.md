# Documentation Index

Modular ISP (Image Signal Processing) pipeline for ComfyUI: process raw camera images through white balance, demosaicing, exposure, gamma correction, and export to JPEG.

## Quick Navigation

### Getting Started
- **[Installation Guide](installation.md)** — Guide for project setup and dependency installing
- **[Quickstart](quickstart.md)** — First workflow in 5 minutes (ComfyUI + Python)

### Reference & Deep Dives
- **[Usage & Nodes](usage_nodes.md)** — Complete reference for all 12 custom nodes, algorithms, parameters, examples, Python API, best practices, troubleshooting

### Architecture & Design
- **[Project Architecture Documentation](architecture.md)** — Module structure, design patterns, extension points (coming soon)

---

## Document Overview

| Document | Purpose |
|----------|---------|
| [installation.md](installation.md) | Setup project |
| [quickstart.md](quickstart.md) | Get first result in 5 min |
| [usage_nodes.md](usage_nodes.md) | Complete node & API reference |
| [architecture.md](architecture.md) | Design & extension guide |

---

## 12 Nodes at a Glance

**Pipeline order:** RAW → Linearize → Demosaicing → White Balance → Exposure → Gamma → JPEG

| Category | Nodes | Key Choice |
|----------|-------|-----------|
| **Input** | Read RAW Sensor | Sony `.arw` files only |
| **Processing** | Black Level Subtraction | Essential (always use) |
| **Demosaicing** | Bilinear, Malvar-He-Cutler | Choose: speed vs. quality |
| **Color** | Camera WB, Gray World, White Patch, Ground Truth, Comparison | Choose: automatic or reference-based |
| **Brightness** | Exposure Compensation | Optional; ±1–2 EV typical |
| **Tone** | Gamma Correction | Essential; sRGB γ=2.2 standard |
| **Output** | Save JPEG | Quality 1–100; default 75 |

👉 Full specifications in [usage_nodes.md](usage_nodes.md)

---

## Need Help?

| Question | Answer |
|----------|--------|
| **How do I start?** | [Quickstart](quickstart.md) (5 min) |
| **What does node X do?** | Search [Usage Nodes](usage_nodes.md) for node name |
| **How do I choose white balance?** | [Usage Nodes > Decision Tree](usage_nodes.md#white-balance-decision-tree) |
| **Python API example?** | [Usage Nodes > Python API](usage_nodes.md#python-api-usage-without-comfyui) |

---

**Document Version:** 1.0  
**Last Updated:** March 27, 2026  
**Compatibility:** Python 3.13+, PyTorch 2.0+, ComfyUI latest
