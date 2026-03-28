# DataFlow-CV

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Actions](https://github.com/zjykzj/DataFlow-CV/actions/workflows/python-publish.yml/badge.svg)](https://github.com/zjykzj/DataFlow-CV/actions)

A computer vision dataset processing library for seamless format conversion and visualization between LabelMe, COCO, and YOLO annotation formats. Designed for researchers and developers working with multi-format annotation pipelines.

## Features

- **Bidirectional Conversion**: Convert between LabelMe, COCO, and YOLO formats in any direction
- **Multi-format Support**: Handle object detection bounding boxes and instance segmentation polygons
- **Lossless Round-trip**: Preserve original coordinates through conversion chains
- **Visualization**: Visualize annotations with OpenCV, supporting both display and save modes
- **Command-line Interface**: User-friendly CLI with `convert` and `visualize` subcommands
- **Python API**: Programmatic access for integration into larger pipelines
- **Verbose Logging**: Detailed logging with file output for debugging
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

#### Format Conversion
```bash
# YOLO to COCO
dataflow-cv convert yolo2coco yolov8-labels/ coco-annotations/ \
  --image-dir images/ \
  --class-file classes.txt

# COCO to LabelMe
dataflow-cv convert coco2labelme annotations_trainval2017/ labelme-json/ \
  --image-dir images/

# LabelMe to YOLO
dataflow-cv convert labelme2yolo labelme-json/ yolo-labels/ \
  --image-dir images/ \
  --class-file classes.txt

# Enable verbose logging
dataflow-cv convert yolo2coco --verbose --log-dir ./logs ...
```

#### Visualization
```bash
# Visualize YOLO annotations
dataflow-cv visualize yolo \
  --label-dir yolov8-labels/ \
  --image-dir images/ \
  --class-file classes.txt \
  --output-dir visualized/

# Visualize COCO annotations
dataflow-cv visualize coco \
  --coco-json annotations_trainval2017/annotations.json \
  --image-dir images/ \
  --output-dir visualized/

# Visualize LabelMe annotations
dataflow-cv visualize labelme \
  --label-dir labelme-json/ \
  --image-dir images/ \
  --output-dir visualized/
```

### Python API

```python
from dataflow.convert import YoloAndCocoConverter
from dataflow.visualize import YOLOVisualizer

# Convert YOLO to COCO
converter = YoloAndCocoConverter(source_to_target=True, verbose=True)
result = converter.convert(
    source_path="yolov8-labels/",
    target_path="coco-annotations/",
    class_file="classes.txt",
    image_dir="images/"
)

# Visualize YOLO annotations
visualizer = YOLOVisualizer(
    label_dir="yolov8-labels/",
    image_dir="images/",
    class_file="classes.txt",
    is_show=True,
    is_save=True,
    output_dir="visualized/"
)
result = visualizer.visualize()
```

See the `samples/` directory for complete examples:
- `samples/visualize/yolo_demo.py` - YOLO visualization example
- `samples/visualize/labelme_demo.py` - LabelMe visualization example
- `samples/visualize/coco_demo.py` - COCO visualization example
- `samples/convert/` - Conversion examples

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Detailed architecture and development guide
- **`docs/specs/`**: Module specifications and design documents
- **`CHANGELOG.md`**: Version history and breaking changes

### Key Concepts

- **Normalized Coordinates**: All internal coordinates are in 0-1 range
- **Original Data Preservation**: Lossless round-trip conversion through `OriginalData` system
- **Strict Mode**: Validation errors raise exceptions (default: enabled, disable with `--skip-errors`)
- **Verbose Logging**: Detailed debug logs saved to files when `--verbose` is used

## Development
For detailed developer guidance including advanced test commands, debugging, and architecture overview, see [CLAUDE.md](CLAUDE.md).

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=dataflow

# Run specific test module
pytest tests/convert/test_yolo_and_coco.py
```

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

### Project Structure
```
dataflow/
├── label/           # Annotation handlers (YOLO, LabelMe, COCO)
├── convert/         # Format converters
├── visualize/       # Visualization modules
├── util/           # Utilities (logging, file operations)
└── cli/            # Command-line interface
tests/              # Comprehensive test suite
samples/            # Usage examples
assets/             # Sample data for testing
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