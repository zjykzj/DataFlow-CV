# DataFlow-CV

> **Where Vibe Coding meets CV data.** 🌊
> Convert & visualize datasets. Built with the flow of Claude Code.

![Python Version](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.1.1-orange)
![Development Status](https://img.shields.io/badge/status-alpha-yellow)

A data processing library for computer vision datasets, focusing on format conversion and visualization between LabelMe, COCO, and YOLO formats. Provides both a CLI and Python API.

## Table of Contents

- [Features](#features)
- [Supported Formats](#supported-formats)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Development](#development)
- [License](#license)

## Features

- **Format Conversion**: Convert between LabelMe, COCO, and YOLO formats
- **Batch Conversion**: Process entire directories with progress display and error handling
- **Single-Image Visualization**: Visualize annotations on individual images
- **Batch Visualization**: Process entire directories with interactive navigation
- **Simple CLI**: Easy-to-use command-line interface with intuitive subcommands
- **Python API**: Programmatic access to all conversion and visualization functions

## Supported Formats

- **LabelMe**: JSON format for polygon/rectangle annotations
- **COCO**: JSON format for object detection
- **YOLO**: TXT format with normalized coordinates

## Quick Start

1. Install the library:
   ```bash
   # Regular installation
   pip install .

   # For development (editable installation)
   python setup.py develop
   ```

2. Convert a COCO annotation to YOLO format:
   ```bash
   dataflow convert coco2yolo image.jpg annotation.json output.txt --class-names classes.txt
   ```

3. Visualize a YOLO annotation:
   ```bash
   dataflow visualize yolo image.jpg label.txt classes.txt --show
   ```

See the full [Usage](#usage) section for more examples.

## Installation

```bash
# Install from source (regular installation)
pip install .

# For development (editable installation)
python setup.py develop

# Install with optional dependencies (pycocotools, torch, torchvision)
pip install .[full]

# Or install directly (after release)
# pip install dataflow-cv
```

## Usage

### Command Line Interface

#### Format Conversion

##### Single File Conversion
```bash
# COCO to YOLO
dataflow convert coco2yolo image.jpg annotation.json output.txt --class-names classes.txt

# YOLO to COCO
dataflow convert yolo2coco image.jpg label.txt classes.txt output.json

# LabelMe to COCO
dataflow convert labelme2coco labelme.json output.json

# COCO to LabelMe
dataflow convert coco2labelme annotation.json image.jpg output.json
```

##### Batch Conversion (Process Directories)
```bash
# COCO to YOLO (batch mode)
dataflow convert coco2yolo images/ annotations/ output/ --batch --class-names classes.txt
# Creates output/ directory with .txt files for each image

# YOLO to COCO (batch mode - per file)
dataflow convert yolo2coco images/ labels/ classes.txt output/ --batch
# Creates output/ directory with .json files for each image

# YOLO to COCO (batch mode - combined COCO file)
dataflow convert yolo2coco images/ labels/ classes.txt combined_coco.json --batch --combined
# Creates a single COCO JSON file with all images and annotations

# LabelMe to COCO (batch mode - per file)
dataflow convert labelme2coco annotations/ output/ --batch
# Creates output/ directory with .json files for each LabelMe file

# LabelMe to COCO (batch mode - combined COCO file)
dataflow convert labelme2coco annotations/ combined_coco.json --batch --combined
# Creates a single COCO JSON file with all LabelMe annotations

# COCO to LabelMe (batch mode)
dataflow convert coco2labelme images/ annotations/ output/ --batch
# Creates output/ directory with .json files for each image

# LabelMe to YOLO (batch mode)
dataflow convert labelme2yolo annotations/ output/ --batch --class-names classes.txt
# Creates output/ directory with .txt files for each LabelMe file

# YOLO to LabelMe (batch mode)
dataflow convert yolo2labelme images/ labels/ classes.txt output/ --batch
# Creates output/ directory with .json files for each image
```

#### Visualization

##### Single Image Visualization
```bash
# Visualize COCO annotations
dataflow visualize coco image.jpg annotation.json --show --save output.jpg

# Visualize YOLO annotations
dataflow visualize yolo image.jpg label.txt classes.txt --show

# Visualize LabelMe annotations
dataflow visualize labelme image.jpg annotation.json --show
```

##### Batch Visualization (Process Directories)
```bash
# Batch visualize COCO annotations with interactive navigation
dataflow visualize coco images/ annotations/ --batch --show --save output/
# Navigation: ← previous, → next, q quit

# Batch visualize YOLO annotations and save all to directory
dataflow visualize yolo images/ labels/ classes.txt --batch --save output/ --no-show

# Batch visualize LabelMe annotations with both display and save
dataflow visualize labelme images/ annotations/ --batch --show --save output/
```

### Python API

##### Single File Conversion
```python
import dataflow

# Format conversion
dataflow.convert.coco_to_yolo("image.jpg", "coco.json", "yolo.txt")
dataflow.convert.yolo_to_coco("image.jpg", "yolo.txt", ["cat", "dog"], "coco.json")

# Visualization
image = dataflow.visualize.visualize_coco("image.jpg", "annotation.json")
image = dataflow.visualize.visualize_yolo("image.jpg", "label.txt", ["cat", "dog"])
```

##### Batch Conversion
```python
import dataflow
from pathlib import Path

# Batch conversion functions are available for all formats
# Example: Batch COCO to YOLO
pairs = [
    ("image1.jpg", "annotation1.json"),
    ("image2.jpg", "annotation2.json"),
    ("image3.jpg", "annotation3.json")
]
dataflow.convert.batch_coco_to_yolo(pairs, ["cat", "dog"], "output_dir/")

# Example: Batch LabelMe to COCO (combined file)
labelme_files = ["labelme1.json", "labelme2.json", "labelme3.json"]
pairs = [(f, f) for f in labelme_files]  # Same file for input and annotation
dataflow.convert.batch_labelme_to_coco(pairs, "combined_coco.json")

# Batch utility functions
from dataflow.convert.batch import batch_process_conversion, find_matching_conversion_pairs

# Find matching image-annotation pairs
pairs = dataflow.convert.find_matching_conversion_pairs("images/", "annotations/", ".json")

# Process batch conversion
dataflow.convert.batch_process_conversion(
    pairs,
    dataflow.convert.coco_to_yolo,
    "output_dir/",
    needs_image=True,
    class_names=["cat", "dog"]
)
```

## Project Structure

```
dataflow/
├── __init__.py
├── cli.py                    # Command-line interface
├── config.py                 # Configuration management
├── convert/                  # Format conversion module
│   ├── __init__.py
│   ├── base.py              # Converter base class
│   ├── batch.py             # Batch conversion utilities
│   ├── labelme_to_coco.py
│   ├── coco_to_labelme.py
│   ├── labelme_to_yolo.py
│   ├── yolo_to_labelme.py
│   ├── coco_to_yolo.py
│   └── yolo_to_coco.py
└── visualize/                # Visualization module
    ├── __init__.py
    ├── base.py              # Visualizer base class
    ├── labelme_vis.py
    ├── coco_vis.py
    ├── yolo_vis.py
    └── batch.py             # Batch visualization utilities
```

## Requirements

- Python 3.8 or higher
- Linux environment (POSIX compatible)
- OpenCV (opencv-python) for visualization
- Pillow for image size detection
- NumPy for numerical operations
- Click for CLI

## Development

For detailed development instructions, see [CLAUDE.md](CLAUDE.md). Key commands:

- Run tests: `python -m tests.convert.test_convert`
- Build distribution: `python -m build`
- Install with optional dependencies: `pip install -e .[full]`

## License

[MIT License](LICENSE) © 2026 zjykzj