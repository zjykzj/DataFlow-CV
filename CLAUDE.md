# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataFlow-CV is a Python library for computer vision dataset processing. The project focuses on providing unified label file format conversion, visualization, and data processing functionality.

## Platform Compatibility

DataFlow-CV is designed for full cross-platform compatibility (Windows, Linux, macOS).

## Key Design Principles

1. **Unified Interfaces**: All annotation handlers follow the same interface defined in `BaseAnnotationHandler`
2. **Type Safety**: Extensive use of Python dataclasses with type hints
3. **Cross-platform Compatibility**: Use `pathlib.Path` for all path operations, enforce UTF-8 encoding
4. **Error Handling**: Strict mode by default with informative error messages
5. **Testing**: Comprehensive unit tests with ≥90% coverage requirement

## Architecture Overview

### Core Data Model
- `DatasetAnnotations`: Top-level container for all annotations in a dataset
- `ImageAnnotation`: Annotations for a single image (width, height, objects)
- `ObjectAnnotation`: Single object annotation with optional `BoundingBox` and/or `Segmentation`
- `BoundingBox`: Normalized center coordinates (x, y, width, height)
- `Segmentation`: Polygon points (normalized) with optional RLE data
- `OriginalData`: Preserves original annotation data for lossless round-trip conversions

### Handler Pattern
- `BaseAnnotationHandler`: Abstract base class for reading/writing annotation formats
- Implementations: `LabelMeAnnotationHandler`, `YoloAnnotationHandler`, `CocoAnnotationHandler`
- Each handler reads annotations into `DatasetAnnotations` and writes them back
- Supports both object detection (bounding boxes) and instance segmentation (polygons)

### Conversion Module
- `BaseConverter`: Abstract base class for format-to-format conversion
- Implementations: `CocoAndLabelMeConverter`, `LabelMeAndYoloConverter`, `YoloAndCocoConverter`
- Uses source and target handlers to read, convert, and write annotations
- Maintains lossless round-trip via `OriginalData` preservation

### Visualization Module
- `BaseVisualizer`: Abstract base class for annotation visualization
- Implementations: `LabelMeVisualizer`, `YOLOVisualizer`, `COCOVisualizer`
- Supports interactive display (OpenCV) and save-to-file modes
- Automatic color management per category

### Utilities
- `FileOperations`: Safe file operations with cross-platform path handling
- `ColorManager`: Consistent color assignment for visualization
- `LoggingUtil`: Structured logging configuration

## Development Commands

### Setup
```bash
# Install package in development mode with all optional dependencies
pip install -e .[dev,coco]

# Or install core only
pip install -e .
```

### Testing
```bash
# Run all tests with coverage
pytest --cov=dataflow --cov-report=term-missing

# Run specific test module
pytest tests/label/test_labelme_handler.py

# Run tests with verbose output
pytest -v

# Generate HTML coverage report
pytest --cov=dataflow --cov-report=html
```

### Linting & Formatting
```bash
# Format code with black
black dataflow tests samples

# Sort imports with isort
isort dataflow tests samples

# Run flake8 for style checking
flake8 dataflow tests samples

# Type checking with mypy
mypy dataflow

# Run pylint
pylint dataflow
```

### Building & Distribution
```bash
# Build source and wheel distributions
python -m build

# Check package metadata
python setup.py check --metadata --strict
```

## Important Paths
- `dataflow/`: Main package source code
- `tests/`: Unit tests organized by module
- `samples/`: Example scripts and demonstration data
- `assets/`: Sample images and annotation files for testing
- `docs/`: Documentation (specifications, design notes)