# DataFlow

A data processing library for computer vision datasets, focusing on format conversion and visualization.

## Features

- **Format Conversion**: Convert between LabelMe, COCO, and YOLO formats
- **Single-Image Visualization**: Visualize annotations on individual images
- **Simple CLI**: Easy-to-use command-line interface

## Supported Formats

- **LabelMe**: JSON format for polygon/rectangle annotations
- **COCO**: JSON format for object detection
- **YOLO**: TXT format with normalized coordinates

## Installation

```bash
# Install from source
pip install -e .

# Or install directly (after release)
# pip install dataflow-cv
```

## Usage

### Command Line Interface

#### Format Conversion

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

```python
import dataflow

# Format conversion
dataflow.convert.coco_to_yolo("image.jpg", "coco.json", "yolo.txt")
dataflow.convert.yolo_to_coco("image.jpg", "yolo.txt", ["cat", "dog"], "coco.json")

# Visualization
image = dataflow.visualize.visualize_coco("image.jpg", "annotation.json")
image = dataflow.visualize.visualize_yolo("image.jpg", "label.txt", ["cat", "dog"])
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
    └── yolo_vis.py
```

## Requirements

- Python 3.8 or higher
- Linux environment (POSIX compatible)

## License

[MIT License](LICENSE) © 2026 zjykzj