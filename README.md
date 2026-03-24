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
- **Verbose Logging System**: Two-level output control with detailed logs, progress feedback, and structured summaries

## Logging and Verbose Mode

DataFlow-CV includes a powerful verbose logging system that provides detailed insights into the processing workflow. The system offers two output modes:

### Verbose Mode Features

- **Two-level output control**:
  - Default mode (`verbose=False`): Shows only concise summary in console
  - Verbose mode (`verbose=True`): Shows summary in console + detailed logs to file

- **Automatic log file management**:
  - Logs are automatically saved to `./logs/log_YYYYMMDD_HHMMSS.log`
  - Log rotation prevents oversized files (10MB max, 5 backups)
  - UTF-8 encoding ensures cross-platform compatibility

- **Progress feedback**:
  - Text-based progress bars for batch operations
  - Real-time status updates
  - Processing statistics and timing information

- **Structured summary output**:
  - Beautifully formatted operation summaries
  - Key statistics and metrics
  - Clear success/failure indicators

### Using Verbose Mode

#### In Python Code

```python
from dataflow.visualize import YOLOVisualizer
from dataflow.convert import LabelMeAndYoloConverter

# Visualization with verbose mode
visualizer = YOLOVisualizer(
    label_dir="path/to/labels",
    image_dir="path/to/images",
    class_file="path/to/classes.txt",
    verbose=True,  # Enable verbose logging
    is_save=True,
    output_dir="path/to/output"
)
result = visualizer.visualize()

# Conversion with verbose mode
converter = LabelMeAndYoloConverter(
    source_to_target=True,
    verbose=True  # Enable verbose logging
)
result = converter.convert(
    source_path="path/to/labelme",
    target_path="path/to/yolo_output",
    class_file="path/to/classes.txt"
)
print(result.get_verbose_summary())  # Print detailed summary
```

#### Command Line Examples

All demo scripts in `samples/` support the `--verbose` command line option:

```bash
# Visualization with verbose logging
python samples/visualize/yolo_demo.py \
    --label_dir assets/test_data/det/yolo/labels \
    --image_dir assets/test_data/det/yolo/images \
    --class_file assets/test_data/det/yolo/classes.txt \
    --verbose

# Conversion with verbose logging
python samples/convert/labelme_to_yolo_demo.py \
    --source assets/test_data/det/labelme \
    --target outputs/yolo_output \
    --class_file assets/test_data/det/labelme/classes.txt \
    --verbose
```

### Log File Format

Log files contain comprehensive information including:
- Timestamps with millisecond precision
- Log level (DEBUG, INFO, WARNING, ERROR)
- Source file and line number
- Detailed processing steps
- Performance statistics
- Error traces and warnings

Example log entry:
```
2026-03-24 10:30:15 - convert.labelme_to_yolo - INFO - labelme_and_yolo.py:42 - Reading 150 image annotations
2026-03-24 10:30:15 - convert.labelme_to_yolo - DEBUG - labelme_and_yolo.py:56 - Category mapping: {'person': 0, 'car': 1}
```

### Progress Feedback Format

During batch operations, you'll see progress bars like:
```
[=========>.................................] 25.0% Processing image_038.jpg
[====================>......................] 50.0% Processing image_075.jpg
[=================================>.........] 75.0% Processing image_113.jpg
[==========================================>] 100.0% Complete!
```

### Summary Output Format

When verbose mode is enabled, structured summaries are displayed:
```
====================================================================
Visualization Operation Summary
====================================================================
• Module: YOLOVisualizer
• Duration: 12.34 seconds
• Input Label Directory: /path/to/labels
• Input Image Directory: /path/to/images
• Output Directory: /path/to/output
• Image Statistics:
  • Total: 150
  • Successful: 150
  • Failed: 0
  • Success Rate: 100.0%
• Total Objects: 1200
• Status: Success
====================================================================
```

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