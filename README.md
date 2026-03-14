# DataFlow-CV

> **Where Vibe Coding meets CV data.** 🌊
> Convert & visualize datasets. Built with the flow of Claude Code.

![Python Version](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue) ![License](https://img.shields.io/badge/license-MIT-green) [![PyPI](https://img.shields.io/pypi/v/dataflow-cv.svg)](https://pypi.org/project/dataflow-cv/) ![Development Status](https://img.shields.io/badge/status-alpha-yellow) [![GitHub Actions](https://github.com/zjykzj/DataFlow-CV/actions/workflows/python-publish.yml/badge.svg)](https://github.com/zjykzj/DataFlow-CV/actions/workflows/python-publish.yml) 
![Linux](https://img.shields.io/badge/Linux-Supported-fcc624?logo=linux) ![Windows](https://img.shields.io/badge/Windows-Supported-00a2e8?logo=windows) ![macOS](https://img.shields.io/badge/macOS-Supported-999999?logo=apple)


A data processing library for computer vision datasets, focusing on format conversion and visualization between LabelMe, COCO, and YOLO formats. Provides both a CLI and Python API.

## Table of Contents

- [DataFlow-CV](#dataflow-cv)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Requirements](#requirements)
    - [Core Dependencies](#core-dependencies)
  - [Quick Start](#quick-start)
    - [Installation](#installation)
      - [Editable Installation (Development Mode)](#editable-installation-development-mode)
      - [Build System](#build-system)
    - [Command Line Usage](#command-line-usage)
    - [Python API Usage](#python-api-usage)
    - [CLI Reference](#cli-reference)
      - [Global Options](#global-options)
      - [Conversion Commands](#conversion-commands)
      - [Visualization Commands](#visualization-commands)
      - [Configuration Command](#configuration-command)
      - [Getting Help](#getting-help)
    - [Segmentation Support](#segmentation-support)
    - [Running Tests](#running-tests)
    - [Examples](#examples)
    - [Documentation](#documentation)
  - [Development](#development)
    - [Cross-Platform Development](#cross-platform-development)
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
│   ├── coco_and_yolo.py    # COCO ↔ YOLO converters
│   ├── coco_and_labelme.py # COCO ↔ LabelMe converters
│   └── yolo_and_labelme.py # YOLO ↔ LabelMe converters
├── visualize/               # Annotation visualization module
│   ├── __init__.py
│   ├── base.py            # Visualizer base class
│   ├── generic.py         # Generic visualizer base class using label handlers
│   ├── yolo.py            # YOLO annotation visualizer
│   ├── coco.py            # COCO annotation visualizer
│   └── labelme.py         # LabelMe annotation visualizer
└── label/                   # Label format handlers module
    ├── __init__.py
    ├── yolo.py            # YOLO format handler
    ├── coco.py            # COCO format handler
    └── labelme.py         # LabelMe format handler
tests/
├── __init__.py
├── convert/                # Conversion tests
│   ├── __init__.py
│   ├── test_coco_to_yolo.py
│   ├── test_yolo_to_coco.py
│   ├── test_coco_to_labelme.py
│   ├── test_labelme_to_coco.py
│   ├── test_labelme_to_yolo.py
│   └── test_yolo_to_labelme.py
├── visualize/              # Visualization tests
│   ├── __init__.py
│   ├── test_yolo.py
│   ├── test_coco.py
│   ├── test_labelme.py
│   └── test_generic.py    # Generic visualizer tests
├── run_tests.py           # Test runner
samples/
├── __init__.py
├── example_usage.py       # Quick usage demonstration
├── template.py            # Example template for creating new examples
├── cli/                   # CLI usage examples
│   ├── __init__.py
│   ├── convert/
│   │   ├── cli_coco_to_yolo.py
│   │   ├── cli_yolo_to_coco.py
│   │   ├── cli_coco_to_labelme.py
│   │   ├── cli_labelme_to_coco.py
│   │   ├── cli_labelme_to_yolo.py
│   │   └── cli_yolo_to_labelme.py
│   └── visualize/
│       ├── cli_yolo.py
│       ├── cli_coco.py
│       └── cli_labelme.py
└── api/                   # Python API examples
    ├── __init__.py
    ├── convert/
    │   ├── api_coco_to_yolo.py
    │   ├── api_yolo_to_coco.py
    │   ├── api_coco_to_labelme.py
    │   ├── api_labelme_to_coco.py
    │   ├── api_labelme_to_yolo.py
    │   └── api_yolo_to_labelme.py
    └── visualize/
        ├── api_yolo.py
        ├── api_coco.py
        └── api_labelme.py
docs/                       # Data format documentation
├── README.md              # Documentation index
├── yolo.md                # YOLO format specification
├── labelme.md             # LabelMe format specification
└── coco.md                # COCO format specification
```

## Requirements

### Core Dependencies
- Python 3.8 or higher
- **Cross-platform compatible**: Windows, Linux, macOS (no platform-specific code)
- `click` >= 7.0.0 – CLI framework
- `numpy` >= 1.24.0 – numerical operations
- `opencv-python` >= 4.6.0.66 – image processing (optional, used for some image operations)
- `Pillow` >= 8.0.0 – image reading (optional, used for reading image dimensions)

**Note**: DataFlow-CV is fully cross-platform compatible and uses only standard Python libraries. All platform-specific code and hardcoded Unix paths have been eliminated.

## Quick Start

DataFlow-CV provides simple and consistent APIs for computer vision dataset processing:

- **Simplified API**: COCO to YOLO conversion now requires only 2 parameters (`coco_json_path`, `output_dir`)
- **Full segmentation support**: Polygon annotations across all formats (COCO, YOLO, LabelMe)
- **Cross-platform compatibility**: Works on Windows, Linux, macOS with no platform-specific code
- **Enhanced visualization**: Distinct colors for many classes with golden ratio distribution

### Installation

```bash
# Regular installation from source
pip install .

# Install from PyPI
pip install dataflow-cv
```

After installation, two command aliases are available: `dataflow` and `dataflow-cv`. Both point to the same functionality and can be used interchangeably.

#### Editable Installation (Development Mode)

Due to setuptools compatibility, use `python setup.py develop` instead of `pip install -e .`:

```bash
# Editable installation (development mode)
python setup.py develop

# After editable installation, use python -m dataflow.cli instead of the dataflow command
# (both dataflow and dataflow-cv commands may not work with editable installation)
python -m dataflow.cli --help
```

#### Build System

The project uses setuptools with a `pyproject.toml` configuration. Distribution packages are built with `python -m build`.

```bash
# Build wheel and source distribution
python -m build

# Install from built wheel
pip install dist/dataflow_cv-*.whl
```

### Command Line Usage

Command line options: `-v/--verbose` is available as a local option for each subcommand. The `--overwrite` option has been removed and is not supported in the current version.

All examples use the `dataflow` command, but you can also use `dataflow-cv` as an alias (e.g., `dataflow-cv convert coco2yolo ...`).

```bash
# COCO to YOLO conversion (use --segmentation for polygon annotations)
dataflow convert coco2yolo annotations.json output_dir/
dataflow convert coco2yolo annotations.json output_dir/ --segmentation

# YOLO to COCO conversion
dataflow convert yolo2coco images/ labels/ classes.names output.json

# COCO to LabelMe conversion (use --segmentation for polygon annotations)
dataflow convert coco2labelme annotations.json output_dir/
dataflow convert coco2labelme annotations.json output_dir/ --segmentation

# LabelMe to COCO conversion
dataflow convert labelme2coco labels/ classes.names output.json

# LabelMe to YOLO conversion (use --segmentation for polygon annotations)
dataflow convert labelme2yolo labels/ output_dir/
dataflow convert labelme2yolo labels/ output_dir/ --segmentation

# YOLO to LabelMe conversion
dataflow convert yolo2labelme images/ labels/ classes.names output_dir/

# Visualize YOLO annotations (use --save to export images, --segmentation for strict segmentation mode)
dataflow visualize yolo images/ labels/ classes.names
dataflow visualize yolo images/ labels/ classes.names --save output_dir/
dataflow visualize yolo images/ labels/ classes.names --segmentation

# Visualize COCO annotations (use --save to export images, --segmentation for strict segmentation mode)
dataflow visualize coco images/ annotations.json
dataflow visualize coco images/ annotations.json --save output_dir/
dataflow visualize coco images/ annotations.json --segmentation

# Visualize LabelMe annotations (use --save to export images, --segmentation for strict segmentation mode)
dataflow visualize labelme images/ labels/
dataflow visualize labelme images/ labels/ --save output_dir/
dataflow visualize labelme images/ labels/ --segmentation

# Show configuration
dataflow config

# Get help
dataflow --help
dataflow convert coco2yolo --help
dataflow visualize yolo --help
dataflow visualize labelme --help
```

See the [CLI Reference](#cli-reference) below for detailed usage.

### Python API Usage

```python
import dataflow

# COCO to YOLO conversion (classes_path is optional, will be auto-generated in output_dir)
result = dataflow.coco_to_yolo("annotations.json", "output_dir")
result = dataflow.coco_to_yolo("annotations.json", "output_dir", segmentation=True)
print(f"Processed {result['images_processed']} images")

# YOLO to COCO conversion
result = dataflow.yolo_to_coco("images/", "labels/", "classes.names", "output.json")
print(f"Generated {result['annotations_processed']} annotations")

# Additional conversions (import converters directly)
from dataflow.convert import (
    CocoToLabelMeConverter,
    LabelMeToCocoConverter,
    LabelMeToYoloConverter,
    YoloToLabelMeConverter
)

# COCO to LabelMe conversion
converter = CocoToLabelMeConverter()
result = converter.convert("annotations.json", "output_dir/", segmentation=True)
print(f"Converted {result['images_processed']} images to LabelMe format")

# LabelMe to COCO conversion
converter = LabelMeToCocoConverter()
result = converter.convert("labels/", "classes.names", "output.json")
print(f"Converted {result['annotations_processed']} annotations to COCO format")

# LabelMe to YOLO conversion
converter = LabelMeToYoloConverter()
result = converter.convert("labels/", "output_dir/")
print(f"Converted {result['images_processed']} images to YOLO format")

# YOLO to LabelMe conversion
converter = YoloToLabelMeConverter()
result = converter.convert("images/", "labels/", "classes.names", "output_dir/")
print(f"Converted {result['images_processed']} images to LabelMe format")

# Visualize YOLO annotations (save_dir is optional, segmentation=True for strict segmentation mode)
result = dataflow.visualize_yolo("images/", "labels/", "classes.names")
result = dataflow.visualize_yolo("images/", "labels/", "classes.names", save_dir="output_dir/")
result = dataflow.visualize_yolo("images/", "labels/", "classes.names", segmentation=True)
print(f"Visualized {result['images_processed']} images")

# Visualize COCO annotations (save_dir is optional, segmentation=True for strict segmentation mode)
result = dataflow.visualize_coco("images/", "annotations.json")
result = dataflow.visualize_coco("images/", "annotations.json", save_dir="output_dir/")
result = dataflow.visualize_coco("images/", "annotations.json", segmentation=True)
print(f"Visualized {result['images_processed']} images")

# Visualize LabelMe annotations (save_dir is optional, segmentation=True for strict segmentation mode)
result = dataflow.visualize_labelme("images/", "labels/")
result = dataflow.visualize_labelme("images/", "labels/", save_dir="output_dir/")
result = dataflow.visualize_labelme("images/", "labels/", segmentation=True)
print(f"Visualized {result['images_processed']} images")
print(f"Classes found: {result['classes_found']}")
```

### CLI Reference

The CLI follows a hierarchical structure: `dataflow <main‑task> <sub‑task> [arguments]`. Each subcommand supports local options such as `-v/--verbose` for progress output.

Both `dataflow` and `dataflow-cv` commands are available and can be used interchangeably.

#### Global Options
- `--version`: Show version information
- `--help`, `-h`: Show help message

#### Local Options (available for each subcommand)
- `--verbose`, `-v`: Enable verbose output (progress information)

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

**COCO to LabelMe**
```bash
dataflow convert coco2labelme COCO_JSON_PATH OUTPUT_DIR [--segmentation]
```
- `COCO_JSON_PATH`: Path to COCO JSON annotation file
- `OUTPUT_DIR`: Directory where LabelMe JSON files will be created
- `--segmentation`, `-s`: Handle segmentation annotations (polygon format)

**LabelMe to COCO**
```bash
dataflow convert labelme2coco LABEL_DIR CLASSES_PATH OUTPUT_JSON_PATH [--segmentation]
```
- `LABEL_DIR`: Directory containing LabelMe JSON files
- `CLASSES_PATH`: Path to class names file (e.g., `class.names`)
- `OUTPUT_JSON_PATH`: Path to save COCO JSON file
- `--segmentation`, `-s`: Handle segmentation annotations (polygon format)

**LabelMe to YOLO**
```bash
dataflow convert labelme2yolo LABEL_DIR OUTPUT_DIR [--segmentation]
```
- `LABEL_DIR`: Directory containing LabelMe JSON files
- `OUTPUT_DIR`: Directory where `labels/` and `class.names` will be created
- `--segmentation`, `-s`: Handle segmentation annotations (polygon format)

**YOLO to LabelMe**
```bash
dataflow convert yolo2labelme IMAGE_DIR LABEL_DIR CLASSES_PATH OUTPUT_DIR [--segmentation]
```
- `IMAGE_DIR`: Directory containing image files
- `LABEL_DIR`: Directory containing YOLO label files (`.txt`)
- `CLASSES_PATH`: Path to YOLO class names file (e.g., `class.names`)
- `OUTPUT_DIR`: Directory where LabelMe JSON files will be created
- `--segmentation`, `-s`: Handle segmentation annotations (polygon format)

#### Visualization Commands

**Visualize YOLO annotations**
```bash
dataflow visualize yolo IMAGE_DIR LABEL_DIR CLASS_PATH [--save SAVE_DIR]
```
- `IMAGE_DIR`: Directory containing image files
- `LABEL_DIR`: Directory containing YOLO label files (`.txt`)
- `CLASS_PATH`: Path to class names file (e.g., `class.names`)
- `--save SAVE_DIR`: Optional directory to save visualized images

**Visualize COCO annotations**
```bash
dataflow visualize coco IMAGE_DIR ANNOTATION_JSON [--save SAVE_DIR]
```
- `IMAGE_DIR`: Directory containing image files
- `ANNOTATION_JSON`: Path to COCO JSON annotation file
- `--save SAVE_DIR`: Optional directory to save visualized images

**Visualize LabelMe annotations**
```bash
dataflow visualize labelme IMAGE_DIR LABEL_DIR [--save SAVE_DIR]
```
- `IMAGE_DIR`: Directory containing image files
- `LABEL_DIR`: Directory containing LabelMe JSON files
- `--save SAVE_DIR`: Optional directory to save visualized images

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
dataflow visualize --help
dataflow visualize yolo --help
dataflow visualize coco --help
dataflow visualize labelme --help
```

### Segmentation Support

DataFlow-CV supports both bounding box and polygon segmentation annotations across all formats:

**YOLO Segmentation Format**
- Detection format: `class_id x_center y_center width height` (normalized coordinates)
- Segmentation format: `class_id x1 y1 x2 y2 ...` (polygon vertices, normalized)
- YOLO segmentation files have the same `.txt` extension as detection files

**COCO Segmentation Format**
- Polygon coordinates in `segmentation` field (list of `[x1, y1, x2, y2, ...]`)
- Both single-polygon and multi-polygon annotations are supported

**LabelMe Segmentation Format**
- Rectangle shapes (`shape_type: "rectangle"`) for bounding box annotations
- Polygon shapes (`shape_type: "polygon"`) for segmentation annotations
- Each JSON file contains `shapes` array with annotation data

**Usage Examples**

```bash
# Convert COCO to YOLO with segmentation annotations
dataflow convert coco2yolo annotations.json output_dir/ --segmentation

# Visualize YOLO annotations in strict segmentation mode (only polygons)
dataflow visualize yolo images/ labels/ classes.names --segmentation

# Visualize COCO annotations in strict segmentation mode
dataflow visualize coco images/ annotations.json --segmentation

# Visualize LabelMe annotations in strict segmentation mode (only polygons)
dataflow visualize labelme images/ labels/ --segmentation
```

**Python API**
```python
# Convert COCO to YOLO with segmentation
result = dataflow.coco_to_yolo("annotations.json", "output_dir", segmentation=True)

# Visualize in strict segmentation mode
result = dataflow.visualize_yolo("images/", "labels/", "classes.names", segmentation=True)
result = dataflow.visualize_coco("images/", "annotations.json", segmentation=True)
result = dataflow.visualize_labelme("images/", "labels/", segmentation=True)
```

**Notes**
- Without the `--segmentation` flag, both bounding boxes and polygons are processed automatically
- With `--segmentation` flag, only valid polygon annotations are processed (strict mode)
- YOLO segmentation format requires at least 3 points (6 coordinates)
- COCO segmentation polygons are automatically converted to YOLO normalized coordinates
- LabelMe format supports both rectangle (`shape_type: "rectangle"`) and polygon (`shape_type: "polygon"`) shapes
- In segmentation mode, LabelMe visualizer rejects rectangle shapes and only accepts polygon shapes

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

- `samples/cli/convert/` - CLI conversion examples
- `samples/cli/visualize/` - CLI visualization examples
- `samples/api/convert/` - Python API conversion examples
- `samples/api/visualize/` - Python API visualization examples

### Documentation

Detailed data format specifications are available in the `docs/` directory:

- [`docs/README.md`](docs/README.md) - Documentation index
- [`docs/yolo.md`](docs/yolo.md) - YOLO format specification
- [`docs/labelme.md`](docs/labelme.md) - LabelMe format specification
- [`docs/coco.md`](docs/coco.md) - COCO format specification

These documents describe the annotation formats supported by DataFlow-CV, without covering tool usage.
## Development

For development guidelines, architecture details, and contribution instructions, see [CLAUDE.md](CLAUDE.md). This file provides guidance for working with the codebase, including common development commands, architectural patterns, and writing principles.

### Cross-Platform Development

DataFlow-CV is designed for full cross-platform compatibility (Windows, Linux, macOS). Key principles:
- Uses only standard Python libraries with no platform-specific APIs
- File operations use `os.path.join()`, `pathlib.Path`, and `shutil` modules
- Temporary files use `tempfile.mkdtemp()` and `tempfile.mkstemp()`
- All 193 tests pass on both Linux and Windows platforms
- No hardcoded Unix paths remain in the codebase

## License

[MIT License](LICENSE) © 2026 zjykzj