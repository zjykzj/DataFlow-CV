# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataFlow-CV is a computer vision dataset processing library for format conversion and visualization between LabelMe, COCO, and YOLO annotation formats. It provides both a Python API and a command-line interface (CLI) with `convert` and `visualize` subcommands.

The project follows a modular architecture with clear separation between format handlers, converters, visualizers, and utilities. All coordinates are normalized (0-1 range) in the internal data model.

## Architecture Overview

### Core Data Model (`dataflow/label/models.py`)
- `DatasetAnnotations`: Top-level container for all annotations in a dataset
- `ImageAnnotation`: Annotations for a single image (width, height, objects)
- `ObjectAnnotation`: Single object annotation with optional `BoundingBox` and `Segmentation`
- `OriginalData`: Preserves original annotation data for lossless round-trip conversions
- `AnnotationFormat` enum: `LABELME`, `YOLO`, `COCO`

### Annotation Handlers (`dataflow/label/`)
- `BaseAnnotationHandler`: Abstract base class defining `read()`, `write()`, `validate()` interface
- `YoloHandler`: Handles YOLO format (both detection and segmentation)
- `LabelMeHandler`: Handles LabelMe JSON format
- `CocoHandler`: Handles COCO JSON format (supports RLE for segmentation)

Handlers are responsible for reading/writing annotation files and converting between external formats and the internal data model.

### Converters (`dataflow/convert/`)
- `BaseConverter`: Abstract base class for format conversion with verbose logging support
- `YoloAndCocoConverter`: Bidirectional conversion between YOLO and COCO
- `LabelMeAndYoloConverter`: Bidirectional conversion between LabelMe and YOLO
- `CocoAndLabelMeConverter`: Bidirectional conversion between COCO and LabelMe

Converters orchestrate the conversion process: read with source handler → convert annotations → write with target handler.

### Visualizers (`dataflow/visualize/`)
- `YOLOVisualizer`: Visualizes YOLO annotations with OpenCV
- `LabelMeVisualizer`: Visualizes LabelMe annotations
- `CocoVisualizer`: Visualizes COCO annotations

Visualizers support both display and save modes, with automatic detection of annotation type (detection vs segmentation).

### CLI Structure (`dataflow/cli/`)
- `main.py`: Entry point with global options (`--verbose`, `--log-dir`, `--strict`)
- `commands/convert.py`: `dataflow convert` subcommands:
  - `yolo2coco`, `yolo2labelme`, `coco2yolo`, `coco2labelme`, `labelme2yolo`, `labelme2coco`
- `commands/visualize.py`: `dataflow visualize` subcommands:
  - `yolo`, `labelme`, `coco`

### Utilities (`dataflow/util/`)
- `logging_util.py`: `LoggingOperations` and `VerboseLoggingOperations` for consistent logging
- `file_util.py`: `FileOperations` for file system operations

## Common Development Tasks

### Installation
```bash
# Regular installation
pip install .

# Editable installation (development)
pip install -e .
```

**Note**: With editable installation, use `python -m dataflow.cli` instead of the `dataflow-cv` command.

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=dataflow

# Run specific test module
pytest tests/convert/test_yolo_and_coco.py

# Run tests with verbose output
pytest -v
```

### Linting and Formatting
Development dependencies are defined in `pyproject.toml` project.optional-dependencies['dev']:
```bash
# Install development dependencies
pip install -e .[dev]

# Format code with black
black dataflow tests samples

# Sort imports with isort
isort dataflow tests samples

# Type checking with mypy
mypy dataflow

# Linting with flake8
flake8 dataflow tests samples

# Linting with pylint
pylint dataflow
```

### Test Structure
Tests are organized by module:
- `tests/label/`: Unit tests for annotation handlers
- `tests/convert/`: Unit tests for converters
- `tests/visualize/`: Unit tests for visualizers
- `tests/util/`: Unit tests for utilities
- `tests/cli/`: Integration tests for CLI commands

### Building and Publishing
The project uses setuptools for packaging. The GitHub workflow `.github/workflows/python-publish.yml` automates PyPI publishing.

## CLI Usage Examples

### Format Conversion
```bash
# YOLO to COCO
dataflow-cv convert yolo2coco yolov8-labels/ coco-annotations/ --image-dir images/ --class-file classes.txt

# COCO to YOLO
dataflow-cv convert coco2yolo annotations_trainval2017/ yolo-labels/ --image-dir images/ --class-file classes.txt

# LabelMe to YOLO
dataflow-cv convert labelme2yolo labelme-json/ yolo-labels/ --image-dir images/ --class-file classes.txt

# With verbose logging
dataflow-cv --verbose convert yolo2coco ... --log-dir ./logs
```

### Visualization
```bash
# Visualize YOLO annotations
dataflow-cv visualize yolo --label-dir yolov8-labels/ --image-dir images/ --class-file classes.txt --output-dir visualized/

# Visualize COCO annotations
dataflow-cv visualize coco --coco-json annotations_trainval2017/annotations.json --image-dir images/ --output-dir visualized/

# Visualize LabelMe annotations
dataflow-cv visualize labelme --label-dir labelme-json/ --image-dir images/ --output-dir visualized/
```

### Python API Examples
See `samples/` directory for comprehensive examples:
- `samples/visualize/yolo_demo.py`
- `samples/visualize/labelme_demo.py`
- `samples/visualize/coco_demo.py`
- `samples/convert/` contains conversion examples

## Important Patterns and Conventions

### Logging Configuration
- Use `LoggingOperations.get_logger()` for standard logging
- Use `VerboseLoggingOperations.get_verbose_logger()` for verbose mode (creates log files)
- When `--verbose` flag is used, log files are created in `logs/` directory with DEBUG details including filename/line numbers
- Console output includes timestamps in both modes

### Strict Mode
- Handlers and converters have `strict_mode` parameter (default: `True`)
- In strict mode, validation errors raise exceptions
- In non-strict mode, errors are logged but processing continues where possible
- CLI `--strict` flag controls this globally; `--skip-errors` sets strict=false

### Original Data Preservation
- The `OriginalData` system preserves original annotation coordinates for lossless round-trip conversions
- When converting between formats, original data is retained when available
- This enables converting A→B→A without precision loss

### Coordinate System
- Internal coordinates are normalized (0-1 range)
- Handlers convert between absolute pixel coordinates and normalized coordinates
- Bounding boxes use center-x, center-y, width, height (YOLO format)
- Segmentation polygons are lists of normalized (x, y) points

### Error Handling
- Operations return `AnnotationResult` or `ConversionResult` with success flag, messages, errors, warnings
- CLI commands raise `RuntimeCLIError` for user-facing errors
- Use `validate_inputs()` pattern in converters

## Notes for Contributors

- The project supports both object detection and instance segmentation annotations
- COCO RLE support requires `pycocotools` (optional dependency)
- All file paths should use `pathlib.Path` for cross-platform compatibility
- Chinese comments are present in some files; maintain bilingual clarity where appropriate
- Follow existing patterns for adding new format handlers or converters
- The actual LLM used for this project is DeepSeek-V3.2; do not use `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>` in commit messages
- When modifying logging behavior, ensure both console and file logging work correctly in verbose mode

## Related Documentation
- `CHANGELOG.md`: Version history and breaking changes
- `docs/specs/`: Detailed specifications for each module
- `assets/`: Sample data for testing and demonstrations