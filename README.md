# DataFlow-CV

> **Where Vibe Coding meets CV data.** 🌊
> Convert & visualize datasets. Built with the flow of Claude Code.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) ![License](https://img.shields.io/badge/license-MIT-green) [![PyPI](https://img.shields.io/pypi/v/dataflow-cv.svg)](https://pypi.org/project/dataflow-cv/) ![Development Status](https://img.shields.io/badge/status-alpha-yellow) [![GitHub Actions](https://github.com/zjykzj/DataFlow-CV/actions/workflows/python-publish.yml/badge.svg)](https://github.com/zjykzj/DataFlow-CV/actions/workflows/python-publish.yml) 
![Linux](https://img.shields.io/badge/Linux-Supported-fcc624?logo=linux) ![Windows](https://img.shields.io/badge/Windows-Supported-00a2e8?logo=windows) ![macOS](https://img.shields.io/badge/macOS-Supported-999999?logo=apple)

A computer vision dataset processing library for seamless format conversion and visualization between LabelMe, COCO, and YOLO annotation formats. Designed for researchers and developers working with multi-format annotation pipelines.

## Features

- **Bidirectional Conversion**: Convert between LabelMe, COCO, and YOLO formats in any direction
- **Multi-format Support**: Handle object detection bounding boxes and instance segmentation polygons
- **Lossless Round-trip**: Preserve original coordinates through conversion chains
- **Visualization**: Visualize annotations with OpenCV, supporting both display and save modes
- **Command-line Interface**: User-friendly CLI with `convert` and `visualize` subcommands
- **Python API**: Programmatic access for integration into larger pipelines
- **Verbose Logging**: Detailed logging with file output for debugging
- **Headless Mode**: Run visualization in server/Docker environments with `--no-display`
- **Flexible Error Handling**: Choose between strict (abort on error) or lenient (skip and continue) modes
- **Cross-platform**: Full support for Windows, Linux, and macOS

## Table of Contents

- [DataFlow-CV](#dataflow-cv)
  - [Features](#features)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [From PyPI](#from-pypi)
    - [From Source](#from-source)
    - [Optional Dependencies](#optional-dependencies)
  - [Quick Start](#quick-start)
    - [Command-line Interface](#command-line-interface)
      - [Format Conversion](#format-conversion)
      - [Visualization](#visualization)
    - [Python API](#python-api)
  - [Documentation](#documentation)
    - [Key Concepts](#key-concepts)
  - [Development](#development)
    - [Testing](#testing)
    - [Linting and Formatting](#linting-and-formatting)
    - [Project Structure](#project-structure)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Installation

### From PyPI
```bash
pip install dataflow-cv
```

### From Source
```bash
# Clone the repository
git clone https://github.com/zjykzj/DataFlow-CV.git
cd DataFlow-CV

# Regular installation
pip install .

# Editable installation (for development)
pip install -e .
```

**Note**: When installed in editable mode, use `python -m dataflow.cli` instead of the `dataflow-cv` command.

### Optional Dependencies
- `pycocotools`: Required for COCO RLE segmentation support
  ```bash
  pip install pycocotools
  ```

## Quick Start

### Command-line Interface

All required parameters (image directories, label directories, class files, output paths) are positional arguments for better usability. Use `--help` on any subcommand for detailed usage.

#### Format Conversion
```bash
# YOLO to COCO
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt coco_annotations.json

# With RLE encoding
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt coco_annotations.json --do-rle

# YOLO to LabelMe
dataflow-cv convert yolo2labelme images/ yolo_labels/ classes.txt labelme_json/

# COCO to YOLO
dataflow-cv convert coco2yolo coco_annotations.json yolo_labels/

# COCO to LabelMe
dataflow-cv convert coco2labelme coco_annotations.json labelme_json/

# LabelMe to YOLO
dataflow-cv convert labelme2yolo labelme_json/ classes.txt yolo_labels/

# LabelMe to COCO
dataflow-cv convert labelme2coco labelme_json/ classes.txt coco_annotations.json

# With RLE encoding
dataflow-cv convert labelme2coco labelme_json/ classes.txt coco_annotations.json --do-rle

# Enable verbose logging
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt coco_annotations.json --verbose

# Disable strict mode (skip invalid annotations instead of aborting)
dataflow-cv convert yolo2coco --no-strict images/ yolo_labels/ classes.txt coco_annotations.json
```

#### Visualization
```bash
# Visualize YOLO annotations
dataflow-cv visualize yolo images/ yolo_labels/ classes.txt --save visualized/

# Visualize COCO annotations
dataflow-cv visualize coco images/ coco_annotations.json --save visualized/

# Visualize LabelMe annotations
dataflow-cv visualize labelme images/ labelme_json/ --save visualized/

# Enable verbose logging for detailed debug output
dataflow-cv visualize yolo --verbose images/ yolo_labels/ classes.txt --save visualized/

# Run on headless server (no display window)
dataflow-cv visualize yolo --no-display images/ yolo_labels/ classes.txt --save visualized/
```

### Python API

```python
from dataflow.convert import YoloAndCocoConverter
from dataflow.visualize import YOLOVisualizer

# Convert YOLO to COCO
converter = YoloAndCocoConverter(source_to_target=True, verbose=True, strict_mode=True)
result = converter.convert(
    source_path="yolo_labels/",
    target_path="coco_annotations.json",
    class_file="classes.txt",
    image_dir="images/",
    do_rle=False  # Set to True for RLE encoding
)

# Visualize YOLO annotations
visualizer = YOLOVisualizer(
    label_dir="yolo_labels/",
    image_dir="images/",
    class_file="classes.txt",
    is_show=True,
    is_save=True,
    output_dir="visualized/",
    verbose=True,
    strict_mode=True
)
result = visualizer.visualize()
```

See the `samples/` directory for complete examples:
- `samples/visualize/yolo_demo.py` - YOLO visualization example
- `samples/visualize/labelme_demo.py` - LabelMe visualization example
- `samples/visualize/coco_demo.py` - COCO visualization example
- `samples/convert/` - Conversion examples

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Detailed architecture, development guide, and known gotchas
- **`CHANGELOG.md`**: Version history and breaking changes
- **[`specs/`](specs/)**: Canonical specifications organized into two layers:
  - **`formats/`** — External format contracts (YOLO, COCO, LabelMe) and conversion rules
  - **`modules/`** — Internal module architecture, interface contracts, and dependency constraints

### Key Concepts

- **Normalized Coordinates**: All internal coordinates are in 0-1 range with center-based bounding boxes (YOLO convention)
- **Original Data Preservation**: Lossless round-trip conversion through `OriginalData` system, preserving exact source values
- **Strict Mode**: Validation errors raise exceptions (default). Disable in CLI with `--no-strict`, or in Python API with `strict_mode=False`
- **Verbose Logging**: Detailed debug logs saved to files when `--verbose` is used. The CLI prints "Verbose log saved to: <path>" after operations.
- **Headless Support**: Use `--no-display` for servers/Docker; use `--save` to output visualization images without a window
- **Keyboard Shortcuts**: During visualization, press `q` or `ESC` to exit early; `Enter`/`Space` to advance; any other key continues
- **Missing Image Handling**: Missing images are skipped with warnings, allowing processing to continue
- **RLE Mask Visualization**: COCO RLE masks are displayed with semi-transparent fills for better visibility
- **Color Management**: Each class ID gets a unique color from an HSV-based palette for consistent visualization
- **Specifications**: The `specs/` directory contains the canonical format and module specifications — the authoritative reference for expected behavior

## Development
For detailed developer guidance including advanced test commands, debugging, and architecture overview, see [CLAUDE.md](CLAUDE.md).

### Testing

294 tests, **73%** code coverage.

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=dataflow --cov-report=term

# Run specific test module
pytest tests/convert/test_yolo_and_coco.py
```

**Coverage by module:**

| Module | Coverage | Notes |
|--------|----------|-------|
| `dataflow/label/` | 78% | Core data models (66%), handlers (67-82%) |
| `dataflow/convert/` | 83% | Converters (81-92%), base pipeline (62%) |
| `dataflow/visualize/` | 69% | Visualizers (94-100%), base drawing (79%) |
| `dataflow/cli/` | 76% | CLI commands (47-96%), utils (86%) |
| `dataflow/util/` | 87% | File ops (84%), logging (99%) |

### Linting and Formatting
```bash
# Install development dependencies
pip install -e .[dev]

# Format code
black dataflow tests samples

# Sort imports
isort dataflow tests samples

# Type checking
mypy dataflow

# Linting
flake8 dataflow tests samples
```

### Pre-commit Hooks (Optional)

Automatically check code quality before each commit:

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks (run once)
pre-commit install

# After this, every `git commit` will auto-run:
#   black (code formatting)
#   isort (import sorting)
#   flake8 (linting)
#   trailing-whitespace / end-of-file-fixer / check-yaml / check-toml

# Manually run against all files
pre-commit run --all-files
```

### Project Structure
```
dataflow/
├── label/           # Annotation handlers + data models (YOLO, LabelMe, COCO)
├── convert/         # Format converters + RLE conversion utility
├── visualize/       # Visualization modules (OpenCV-based)
├── util/            # Logging and file operation utilities
└── cli/             # CLI entry point, commands, and validation
tests/               # Unit and integration tests (label, convert, visualize, cli, util)
samples/             # Python API usage examples (visualize, convert, label, cli)
assets/              # Test data organized by format (det/seg) and annotation type
specs/               # Canonical specifications (formats/ + modules/ layers)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Before contributing, review [CLAUDE.md](CLAUDE.md) for architecture and development patterns.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add or update tests as needed
5. Ensure code passes formatting and linting checks
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the creators of LabelMe, COCO, and YOLO formats for establishing these annotation standards
- Built with OpenCV, NumPy, and Click
- Inspired by the need for seamless format conversion in multi-tool CV pipelines