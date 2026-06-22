# Project Architecture

This document outlines the repository structure and file organization of the project.

## Key Concepts

The architecture relies on a clear separation into two main layers:

* **Core Algorithms (`src/algorithms/`)**: This layer contains pure image processing functions. For example, the RAW reading algorithm uses the `rawpy` library to extract sensor values, the Bayer matrix, and metadata (like black levels and white balance) as tensors.
* **ComfyUI Nodes (`src/custom_nodes/`)**: This layer acts as an integration interface for ComfyUI. It wraps the core algorithms to expose them as visual nodes. For instance, the `ReadRawSensorNode` calls the underlying algorithm and formats the data so they can be processed by the ComfyUI pipeline.

## Directory Structure

* 📁 **[`src/`](../src/)** - Main source code directory
  * 📁 **[`algorithms/`](../src/algorithms/)** - Core image processing algorithms
    * 📁 [`black_light_subtraction/`](../src/algorithms/black_light_subtraction/) - Sensor black level correction
    * 📁 [`demosaicing/`](../src/algorithms/demosaicing/) - Bayer to RGB interpolation algorithms
    * 📁 [`export/`](../src/algorithms/export/) - Output formatting (e.g., JPEG export)
    * 📁 [`exposure_compensation/`](../src/algorithms/exposure_compensation/) - Global exposure adjustments
    * 📁 [`gamma_correction/`](../src/algorithms/gamma_correction/) - Linear to non-linear color space conversion
    * 📁 [`raw/`](../src/algorithms/raw/) - Raw file reading and metadata extraction
    * 📁 [`white_balance/`](../src/algorithms/white_balance/) - White balance algorithms
  * 📁 **[`custom_nodes/`](../src/custom_nodes/)** - ComfyUI integration layer (wrappers for algorithms)
    * 📁 [`black_light_subtraction/`](../src/custom_nodes/black_light_subtraction/)
    * 📁 [`demosaicing/`](../src/custom_nodes/demosaicing/)
    * 📁 [`export/`](../src/custom_nodes/export/)
    * 📁 [`exposure_compensation/`](../src/custom_nodes/exposure_compensation/)
    * 📁 [`gamma_correction/`](../src/custom_nodes/gamma_correction/)
    * 📁 [`raw/`](../src/custom_nodes/raw/)
    * 📁 [`white_balance/`](../src/custom_nodes/white_balance/)
  * 📁 **[`image_processing.egg-info/`](../src/image_processing.egg-info/)** - Generated package metadata
* 📁 **[`tests/`](../tests/)** - Pytest test suite for core algorithms
  * 📄 [`test_bilinear_demosaicing.py`](../tests/test_bilinear_demosaicing.py)
  * 📄 [`test_camera_white_balance.py`](../tests/test_camera_white_balance.py)
  * 📄 [`test_exposure_compensation.py`](../tests/test_exposure_compensation.py)
  * 📄 [`test_gamma_correction.py`](../tests/test_gamma_correction.py)
  * 📄 [`test_gray_world.py`](../tests/test_gray_world.py)
  * 📄 [`test_ground_truth.py`](../tests/test_ground_truth.py)
  * 📄 [`test_malvar_he_culter.py`](../tests/test_malvar_he_culter.py)
  * 📄 [`test_raw_processing.py`](../tests/test_raw_processing.py)
  * 📄 [`test_white_patch_ref.py`](../tests/test_white_patch_ref.py)
* 📁 **[`docs/`](./)** - Project documentation
  * 📁 [`examples/`](./examples/) - Example workflows and usage guides
  * 📄 [`architecture.md`](./architecture.md) - This architecture overview
  * 📄 [`faq.md`](./faq.md) - Frequently Asked Questions
  * 📄 [`index.md`](./index.md) - Main documentation index
  * 📄 [`installation.md`](./installation.md) - Installation instructions
  * 📄 [`quickstart.md`](./quickstart.md) - Quickstart guide
  * 📄 [`usage_nodes.md`](./usage_nodes.md) - Guide to using the ComfyUI nodes
* 📄 **[`.gitignore`](../.gitignore)** - Git ignore rules
* 📄 **[`.gitmodules`](../.gitmodules)** - Git submodules configuration (ComfyUI)
* 📄 **[`.gitlab-ci.yml`](../.gitlab-ci.yml)** - CI/CD pipeline configuration
* 📄 **[`Dockerfile.ci`](../Dockerfile.ci)** - Dockerfile for CI environments
* 📄 **[`Makefile`](../Makefile)** - Main developer entrypoint (setup, run, clean)
* 📄 **[`pyproject.toml`](../pyproject.toml)** - Package and test configuration (uv/pytest)
* 📄 **[`README.md`](../README.md)** - Project introduction and common workflows
* 📄 **[`CHANGELOG.md`](../CHANGELOG.md)** - Project changelog
* 📄 **[`LICENSE.md`](../LICENSE.md)** - GNU GPL v3 License