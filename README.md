# DataFlow-CV

> **Where Vibe Coding meets CV data.** 🌊

A data processing library for computer vision datasets.

## Overview

DataFlow-CV provides unified label file format conversion, visualization, and data processing functionality for computer vision datasets. It supports multiple annotation formats including LabelMe, YOLO, and COCO, with automatic detection of annotation types (object detection vs. instance segmentation). The library features completely lossless read/write functionality, preserving original annotation data to ensure bitwise identical output when reading and re-writing annotation files.

## Key Features

- **Multi-format Support**: LabelMe, YOLO, COCO annotation formats
- **Dual Task Support**: Both object detection and instance segmentation
- **Automatic Detection**: Intelligent format and type detection
- **Lossless Read/Write**: Preserve original annotation data for bitwise identical output
- **Cross-platform**: Full compatibility with Windows, Linux, macOS
- **Comprehensive Testing**: Extensive unit tests and examples
- **Visualization Module**: Interactive visualization with OpenCV, support for display and save modes

## Visualization Module

The visualization module provides annotation visualization for computer vision datasets, supporting three major formats: LabelMe, YOLO, and COCO. It features unified visualization for both object detection and instance segmentation annotations.

### Key Features

- **Multi-format Support**: LabelMe, YOLO, COCO visualization
- **Dual-task Support**: Object detection bounding boxes and instance segmentation polygons
- **RLE Mask Support**: COCO RLE mask decoding and visualization (requires `pycocotools`)
- **Automatic Color Management**: Consistent colors per category with high-contrast palette
- **Interactive Mode**: Show images with keyboard controls (Enter/Space to continue, 'q' or ESC to quit)
- **Save Mode**: Save visualizations as JPEG images with configurable quality
- **Batch Processing**: Process entire directories of images and annotations
- **Strict Error Handling**: Configurable strict mode for immediate error reporting

### Quick Example

```python
from dataflow.visualize import LabelMeVisualizer

# Create visualizer for LabelMe format
visualizer = LabelMeVisualizer(
    label_dir="path/to/labelme_annotations",
    image_dir="path/to/images",
    is_show=True,   # Display images
    is_save=False   # Don't save
)

# Run visualization
result = visualizer.visualize()
if result.success:
    print(f"Visualized {result.data['processed_count']} images")
else:
    print(f"Failed: {result.message}")
```

### Examples

See the `samples/visualize/` directory for complete working examples:
- `labelme_demo.py` - LabelMe format visualization
- `yolo_demo.py` - YOLO format visualization with save mode
- `coco_demo.py` - COCO format visualization

### Installation

```bash
pip install opencv-python  # Required
pip install pycocotools    # Optional, for COCO RLE support
```

## Development

### Installation for Development
```bash
# Install with development dependencies
pip install -e .[dev,coco]
```

### Running Tests
```bash
# Run all tests with coverage
pytest --cov=dataflow --cov-report=term-missing

# Run specific test module
pytest tests/label/test_labelme_handler.py
```

### Code Quality
```bash
# Format code with black
black dataflow tests samples

# Sort imports with isort
isort dataflow tests samples

# Run style checks with flake8
flake8 dataflow tests samples

# Type checking with mypy
mypy dataflow
```

See `CLAUDE.md` for more detailed guidance.

## License

[MIT License](LICENSE) © 2026 zjykzj