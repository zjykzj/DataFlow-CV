# DataFlow-CV

> **Where Vibe Coding meets CV data.** 🌊
> Convert & visualize datasets. Built with the flow of Claude Code.

![Python Version](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.1.1-orange)
![Development Status](https://img.shields.io/badge/status-alpha-yellow)

A data processing library for computer vision datasets, focusing on format conversion and visualization between LabelMe, COCO, and YOLO formats. Provides both a CLI and Python API.

## Table of Contents

- [DataFlow-CV](#dataflow-cv)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Requirements](#requirements)
  - [License](#license)

## Project Structure

```
dataflow/
├── __init__.py              # Package exports and convenience functions
├── cli.py                   # Command-line interface
├── config.py                # Configuration management
├── convert/                 # Format conversion module
│   ├── __init__.py
│   ├── base.py             # Converter base class
│   ├── coco_to_yolo.py     # COCO to YOLO converter
│   └── yolo_to_coco.py     # YOLO to COCO converter
tests/
├── __init__.py
├── convert/                # Conversion tests
│   ├── __init__.py
│   ├── test_coco_to_yolo.py
│   └── test_yolo_to_coco.py
├── run_tests.py           # Test runner
samples/
├── __init__.py
├── cli/                   # CLI usage examples
│   ├── __init__.py
│   └── convert/
│       ├── cli_coco_to_yolo.py
│       └── cli_yolo_to_coco.py
└── api/                   # Python API examples
    ├── __init__.py
    └── convert/
        ├── api_coco_to_yolo.py
        └── api_yolo_to_coco.py
```

## Requirements

### Core Dependencies
- Python 3.8 or higher
- Linux environment (POSIX compatible, assumes POSIX paths)
- `click` >= 8.1.0 – CLI framework
- `numpy` >= 2.0.0 – numerical operations
- `opencv-python` >= 4.8.0 – image processing (optional, used for some image operations)
- `Pillow` >= 10.0.0 – image reading (optional, used for reading image dimensions)

### Optional Dependencies (install with `pip install -e .[full]`)
- `pycocotools` >= 2.0.0 – COCO‑format utilities
- `torch` >= 1.9.0 – PyTorch integration
- `torchvision` >= 0.10.0 – vision datasets

**Note:** The library is designed to work with only the core dependencies; optional packages enable additional functionality.

## Quick Start

### Installation

```bash
# Install in development mode (core dependencies only)
pip install -e .

# Install with all optional dependencies (pycocotools, torch, torchvision)
pip install -e .[full]

# Install directly from source (development mode)
python setup.py develop
```

### Command Line Usage

Global options: `--verbose` (`-v`) for progress output, `--overwrite` to replace existing files.

```bash
# COCO to YOLO conversion
dataflow convert coco2yolo annotations.json output_dir/

# YOLO to COCO conversion
dataflow convert yolo2coco images/ labels/ classes.names output.json

# Show configuration
dataflow config

# Get help
dataflow --help
dataflow convert coco2yolo --help
```

See the [CLI Reference](#cli-reference) below for detailed usage.

### Python API Usage

```python
import dataflow

# COCO to YOLO
result = dataflow.coco_to_yolo("annotations.json", "output_dir")
print(f"Processed {result['images_processed']} images")

# YOLO to COCO
result = dataflow.yolo_to_coco("images/", "labels/", "classes.names", "output.json")
print(f"Generated {result['annotations_processed']} annotations")
```

### CLI Reference

The CLI follows a hierarchical structure: `dataflow <main‑task> <sub‑task> [arguments]`. Global options can be placed before the main task.

#### Global Options
- `--verbose`, `-v`: Enable verbose output (progress information)
- `--overwrite`: Overwrite existing files

#### Conversion Commands

**COCO to YOLO**
```bash
dataflow convert coco2yolo COCO_JSON_PATH OUTPUT_DIR [--segmentation]
```
- `COCO_JSON_PATH`: Path to COCO JSON annotation file
- `OUTPUT_DIR`: Directory where `labels/` and `class.names` will be created
- `--segmentation`, `-s`: Handle segmentation annotations (polygon format)

**YOLO to COCO**
```bash
dataflow convert yolo2coco IMAGE_DIR YOLO_LABELS_DIR YOLO_CLASS_PATH COCO_JSON_PATH
```
- `IMAGE_DIR`: Directory containing image files
- `YOLO_LABELS_DIR`: Directory containing YOLO label files (`.txt`)
- `YOLO_CLASS_PATH`: Path to YOLO class names file (e.g., `class.names`)
- `COCO_JSON_PATH`: Path to save COCO JSON file

#### Configuration Command
```bash
dataflow config
```
Shows the current configuration (file extensions, default values, CLI context).

#### Getting Help
```bash
dataflow --help
dataflow convert --help
dataflow convert coco2yolo --help
dataflow convert yolo2coco --help
```

### Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Run specific test
python tests/run_tests.py --test TestCocoToYoloConverter

# With verbose output
python tests/run_tests.py -v
```

### Examples

Check the `samples/` directory for detailed usage examples:

- `samples/cli/convert/` - CLI usage examples
- `samples/api/convert/` - Python API examples

## License

[MIT License](LICENSE) © 2026 zjykzj