# DataFlow-CV

> **Where Vibe Coding meets CV data.** 🌊
> Convert, visualize & evaluate datasets. Built with the flow of Claude Code.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) ![License](https://img.shields.io/badge/license-MIT-green) [![PyPI](https://img.shields.io/pypi/v/dataflow-cv.svg)](https://pypi.org/project/dataflow-cv/) ![Development Status](https://img.shields.io/badge/status-alpha-yellow) [![GitHub Actions](https://github.com/zjykzj/DataFlow-CV/actions/workflows/python-publish.yml/badge.svg)](https://github.com/zjykzj/DataFlow-CV/actions/workflows/python-publish.yml) 
![Linux](https://img.shields.io/badge/Linux-Supported-fcc624?logo=linux) ![Windows](https://img.shields.io/badge/Windows-Supported-00a2e8?logo=windows) ![macOS](https://img.shields.io/badge/macOS-Supported-999999?logo=apple)

A computer vision dataset processing library for seamless format conversion, visualization, and evaluation between YOLO, LabelMe, and COCO annotation formats. Designed for researchers and developers working with multi-format annotation pipelines.

## Features

- **Format Conversion**: Convert between YOLO, LabelMe, and COCO formats in any direction (6 conversion paths). Supports prediction file conversion with confidence scores.
- **Multi-format Support**: Handle object detection bounding boxes and instance segmentation polygons
- **Native Coordinate Storage**: Coordinates stored in format-native representation (YOLO normalized, LabelMe/COCO absolute pixels)
- **Visualization**: Visualize annotations with OpenCV, supporting both display and save modes with color-coded classes
- **Evaluation**: Evaluate detection/segmentation model outputs with COCO-standard metrics (mAP, mAP50, mAP75, AR) using pycocotools
- **Command-line Interface**: User-friendly CLI with `convert`, `visualize`, and `evaluate` subcommands
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
      - [Evaluation](#evaluation)
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

# LabelMe to YOLO
dataflow-cv convert labelme2yolo labelme_json/ classes.txt yolo_labels/

# LabelMe to COCO
dataflow-cv convert labelme2coco labelme_json/ classes.txt coco_annotations.json

# With RLE encoding
dataflow-cv convert labelme2coco labelme_json/ classes.txt coco_annotations.json --do-rle

# COCO to YOLO
dataflow-cv convert coco2yolo coco_annotations.json yolo_labels/

# COCO to LabelMe
dataflow-cv convert coco2labelme coco_annotations.json labelme_json/

# Convert YOLO predictions to COCO (with confidence scores)
dataflow-cv convert yolo2coco --prediction images/ yolo_preds/ classes.txt pred.json

# Enable verbose logging
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt coco_annotations.json --verbose

# Disable strict mode (skip invalid annotations instead of aborting)
dataflow-cv convert yolo2coco --no-strict images/ yolo_labels/ classes.txt coco_annotations.json
```

#### Visualization
```bash
# Visualize YOLO annotations
dataflow-cv visualize yolo images/ yolo_labels/ classes.txt --save visualized/

# Visualize LabelMe annotations
dataflow-cv visualize labelme images/ labelme_json/ --save visualized/

# Visualize COCO annotations
dataflow-cv visualize coco images/ coco_annotations.json --save visualized/

# Enable verbose logging for detailed debug output
dataflow-cv visualize yolo --verbose images/ yolo_labels/ classes.txt --save visualized/

# Run on headless server (no display window)
dataflow-cv visualize yolo --no-display images/ yolo_labels/ classes.txt --save visualized/
```

#### Evaluation

Evaluate object detection and instance segmentation model outputs using COCO-standard metrics (mAP, AP50, AP75, AR, per-class breakdown). Evaluation requires two COCO-format JSON files:

- **`anno.json`** (Ground Truth / GT): Reference annotations in COCO JSON format
- **`pred.json`** (Detection / DT): Model predictions in COCO JSON format with `score` field

##### Preparing Evaluation Data

If your annotations and predictions are in YOLO format, convert them to COCO JSON first:

```bash
# Step 1: Convert YOLO ground truth labels → COCO GT (anno.json)
# YOLO label format:  class_id cx cy w h  (5 tokens for detection)
#                     class_id x1 y1 ... xn yn  (odd tokens for segmentation)
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt anno.json

# Step 2: Convert YOLO predictions → COCO DT (pred.json)
# YOLO prediction format:  class_id cx cy w h confidence  (6 tokens for detection)
#                          class_id x1 y1 ... xn yn confidence  (even tokens for segmentation)
dataflow-cv convert yolo2coco --prediction images/ yolo_preds/ classes.txt pred.json
```

> **Important**: YOLO label files (GT) use odd token counts (5 for detection), while prediction files (DT) use even token counts (6 for detection) with a trailing `confidence` value. The `--prediction` flag tells the converter to parse prediction format and store confidence as the COCO `score` field. Mixed label/prediction files in the same directory are not supported — the flag applies dataset-wide.

##### Detection vs Segmentation — Format Requirements

| Field | Detection GT | Detection DT | Segmentation GT | Segmentation DT |
|-------|:-----------:|:-----------:|:---------------:|:---------------:|
| `bbox` | Required | Required | Required (for area) | Required (for area) |
| `score` | — | **Required** | — | **Required** |
| `segmentation` | Not required | Not required | **Required** | **Required** |
| `area` | Recommended | Recommended | **Required** | **Required** |
| `iscrowd` | Optional | — | Optional | — |

- **Object Detection** evaluates bounding box overlap using bbox IoU (`iouType='bbox'`). Only `bbox` and `score` are mandatory in DT.
- **Instance Segmentation** evaluates mask overlap using mask IoU (`iouType='segm'`). Both GT and DT must include `segmentation` (polygon or RLE), `area`, and `bbox`.

##### CLI Commands

```bash
# Evaluate object detection results (bbox IoU)
dataflow-cv evaluate detection anno.json pred.json

# Evaluate with verbose per-class breakdown
dataflow-cv evaluate detection --verbose anno.json pred.json

# Evaluate with additional P/R/F1 computation at IoU=0.5
dataflow-cv evaluate detection --prf1 --prf1-iou 0.5 anno.json pred.json

# Evaluate instance segmentation results (mask IoU)
dataflow-cv evaluate segmentation anno.json pred.json

# Save evaluation results as JSON
dataflow-cv evaluate detection --output results.json anno.json pred.json
```

##### End-to-End Workflow

```bash
# Complete pipeline: YOLO data → COCO conversion → evaluation
dataflow-cv convert yolo2coco images/ yolo_labels/ classes.txt anno.json
dataflow-cv convert yolo2coco --prediction images/ yolo_preds/ classes.txt pred.json
dataflow-cv evaluate detection --verbose --prf1 anno.json pred.json
```

### Python API

```python
from dataflow.convert import YoloAndCocoConverter
from dataflow.visualize import YOLOVisualizer
from dataflow.evaluate import DetectionEvaluator, compute_pr_f1

# Convert YOLO to COCO (label mode)
converter = YoloAndCocoConverter(source_to_target=True, verbose=True, strict_mode=True)
result = converter.convert(
    source_path="yolo_labels/",
    target_path="coco_annotations.json",
    class_file="classes.txt",
    image_dir="images/",
    do_rle=False
)

# Convert YOLO predictions to COCO (prediction mode)
converter = YoloAndCocoConverter(source_to_target=True, prediction=True)
result = converter.convert(
    source_path="yolo_preds/",
    target_path="pred.json",
    class_file="classes.txt",
    image_dir="images/"
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

# Evaluate detection results
evaluator = DetectionEvaluator(verbose=True)
result = evaluator.evaluate("gt.json", "pred.json")
print(f"AP: {result.metrics.ap:.3f}, AP50: {result.metrics.ap50:.3f}")

# Quick P/R/F1 at IoU=0.5
prf1 = compute_pr_f1("gt.json", "pred.json", iou_threshold=0.5)
print(f"F1: {prf1.overall.f1_score:.3f}")
```

See the `samples/` directory for complete examples:
- `samples/visualize/yolo_demo.py` - YOLO visualization example
- `samples/visualize/labelme_demo.py` - LabelMe visualization example
- `samples/visualize/coco_demo.py` - COCO visualization example
- `samples/convert/` - Conversion examples

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Detailed architecture, development guide, and known gotchas
- **[CHANGELOG.md](CHANGELOG.md)**: Version history and breaking changes
- **[specs/](specs/)**: Canonical specifications organized into three layers:
  - **`evaluate/`** — Evaluation metric contracts (IoU, matching, AP/mAP/AR, detection vs segmentation)
  - **`formats/`** — External format contracts (YOLO, LabelMe, COCO) and conversion rules
  - **`modules/`** — Internal module architecture, interface contracts, and dependency constraints

### Key Concepts

- **Format-Native Coordinates**: Coordinates stored in each format's native representation — YOLO normalized [0,1] center-based, LabelMe/COCO absolute pixels top-left. See `DatasetAnnotations.format` to determine semantics
- **Explicit Coordinate Transforms**: Converters handle all coordinate transformations between formats. No hidden normalization — lossy vs lossless behavior is explicitly documented
- **Strict Mode**: Validation errors raise exceptions (default). Disable in CLI with `--no-strict`, or in Python API with `strict_mode=False`
- **Verbose Logging**: Detailed debug logs saved to files when `--verbose` is used. The CLI prints "Verbose log saved to: <path>" after operations.
- **Headless Support**: Use `--no-display` for servers/Docker; use `--save` to output visualization images without a window
- **Keyboard Shortcuts**: During visualization, press `q` or `ESC` to exit early; `Enter`/`Space` to advance; any other key continues
- **Missing Image Handling**: Missing images are skipped with warnings, allowing processing to continue
- **RLE Mask Visualization**: COCO RLE masks are displayed with semi-transparent fills for better visibility
- **Color Management**: Each class ID gets a unique color from an HSV-based palette for consistent visualization
- **Evaluation Metrics**: COCO-standard 12-metric output (AP, AP50, AP75, AP-small/medium/large, AR@1/10/100, AR-small/medium/large) with optional per-class breakdown
- **Prediction Files**: YOLO prediction files (6 tokens for detection, even tokens for segmentation) differ from label files (5/odd tokens). Use `--prediction` flag for conversion
- **Specifications**: The `specs/` directory contains the canonical format, evaluate, and module specifications — the authoritative reference for expected behavior

## Development
For detailed developer guidance including advanced test commands, debugging, and architecture overview, see [CLAUDE.md](CLAUDE.md).

### Testing

362 tests, **75%** code coverage.

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=dataflow --cov-report=term

# Run specific test module
pytest tests/convert/test_yolo_and_coco.py
pytest tests/evaluate/test_evaluator.py
```

**Coverage by module:**

| Module | Coverage | Notes |
|--------|----------|-------|
| `dataflow/label/` | 68% | models (87%), coco_handler (74%), labelme_handler (72%), yolo_handler (57%) |
| `dataflow/convert/` | 84% | yolo_and_coco (89%), labelme_and_yolo (93%), coco_and_labelme (88%), rle (81%) |
| `dataflow/visualize/` | 81% | yolo_vis (97%), labelme_vis (100%), coco_vis (97%), base (80%) |
| `dataflow/evaluate/` | 88% | evaluator (100%), metrics (96%), result (99%), base (91%), utils (69%) |
| `dataflow/cli/` | 59% | main (96%), convert cmd (48%), evaluate cmd (24%), visualize cmd (84%), utils (86%) |
| `dataflow/util/` | 93% | logging (99%), file_util (84%) |
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
├── evaluate/        # Evaluation modules (pycocotools-based)
├── util/            # Logging and file operation utilities
└── cli/             # CLI entry point, commands, and validation
tests/               # Unit and integration tests (label, convert, visualize, evaluate, cli, util)
samples/             # Python API usage examples (visualize, convert, label, cli)
assets/              # Test data organized by format (det/seg) and annotation type
specs/               # Canonical specifications (evaluate/ + formats/ + modules/ layers)
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

- Thanks to the creators of YOLO, LabelMe, and COCO formats for establishing these annotation standards
- Built with OpenCV, NumPy, and Click
- Inspired by the need for seamless format conversion in multi-tool CV pipelines