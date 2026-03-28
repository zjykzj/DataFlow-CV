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

# Run specific test class
pytest tests/label/test_yolo.py::TestYoloAnnotationHandler

# Run specific test method
pytest tests/label/test_yolo.py::TestYoloAnnotationHandler::test_read_detection

# Run tests with coverage report (HTML)
pytest --cov=dataflow --cov-report=html

# Run tests in parallel
pytest -n auto

# Run tests with markers (exclude slow tests)
pytest -m "not slow"

# Run tests and generate JUnit XML report (for CI)
pytest --junitxml=test-results.xml
```

### Debugging Conversion Issues

```bash
# Enable verbose logging for debugging
dataflow-cv convert yolo2coco --verbose --log-dir ./logs ...

# Check log files in logs/ directory
ls -la logs/

# Run conversion with strict mode disabled to see warnings
dataflow-cv --skip-errors convert yolo2coco ...

# Use Python API with verbose mode for detailed inspection
python -c "from dataflow.convert import YoloAndCocoConverter; converter = YoloAndCocoConverter(verbose=True); result = converter.convert(...); print(result)"
```

### Adding New Format Support

To add support for a new annotation format:

1. **Extend `BaseAnnotationHandler`** in `dataflow/label/`
2. **Add to `AnnotationFormat` enum** in `models.py`
3. **Create corresponding converter** in `dataflow/convert/`
4. **Add visualizer** in `dataflow/visualize/`
5. **Update CLI commands** in `dataflow/cli/commands/`

Refer to existing implementations (YOLO, LabelMe, COCO) for patterns.

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
dataflow-cv convert yolo2coco --verbose --log-dir ./logs ...
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

- **Normalized Coordinates**: All internal coordinates are 0-1 normalized
- **Conversion**: Handlers convert between normalized and absolute pixel coordinates
- **Bounding Boxes**: Center-x, center-y, width, height (YOLO format)
- **Segmentation**: Lists of normalized (x, y) points
- **Example**: Pixel coordinate (x=320, y=240) in 640×480 image → normalized (0.5, 0.5)
- **Preservation**: Original coordinates stored in `OriginalData` for lossless round-trip conversion

### Error Handling
- Operations return `AnnotationResult` or `ConversionResult` with success flag, messages, errors, warnings
- CLI commands raise `RuntimeCLIError` for user-facing errors
- Use `validate_inputs()` pattern in converters

### Data Flow Pattern

1. **Handlers** (`dataflow/label/`) read/write specific formats to/from the internal `DatasetAnnotations` model
2. **Converters** (`dataflow/convert/`) orchestrate: source handler → convert → target handler
3. **Visualizers** (`dataflow/visualize/`) display or save annotated images
4. **Utilities** (`dataflow/util/`) provide cross-cutting concerns (logging, file operations)


### Result Objects Pattern

- `AnnotationResult` and `ConversionResult` provide structured error handling
- Contains success flag, messages, errors, warnings lists
- Allows operations to continue in non-strict mode while collecting issues
- Used throughout handlers, converters, and visualizers

## Troubleshooting

### Editable Installation Quirk
- With editable installation (`pip install -e .`), use `python -m dataflow.cli` instead of `dataflow-cv` command
- This is because entry point scripts may not be properly linked in development mode

### Path Handling Issues
- Use `pathlib.Path` for cross-platform compatibility
- Ensure image directories exist before conversion
- Relative paths are resolved relative to the current working directory

### Log File Locations
- With `--verbose` flag, log files are created in `logs/` directory (default)
- Custom log directory can be specified with `--log-dir`
- Log files have timestamped names (e.g., `log_20260324_222035.log`)
- Files contain DEBUG details including filename/line numbers

### COCO RLE Support
- Requires optional dependency `pycocotools`
- Install with `pip install dataflow-cv[coco]` or `pip install pycocotools`
- Without it, COCO segmentation conversions will fall back to polygon format

## Dependency Management

### Core Dependencies
- `numpy>=1.24.0`
- `opencv-python>=4.6.0.66`
- `click>=7.0.0`

### Optional Dependencies
- `pycocotools>=2.0.0`: Required for COCO RLE segmentation support
  - Install with `pip install dataflow-cv[coco]` or `pip install pycocotools`
- Without pycocotools, COCO segmentation conversions fall back to polygon format

### Development Dependencies
- Defined in `pyproject.toml` project.optional-dependencies['dev']
- Install with `pip install -e .[dev]`
- Includes pytest, black, isort, flake8, mypy, pylint

## Notes for Contributors

- The project supports both object detection and instance segmentation annotations
- COCO RLE support requires `pycocotools` (optional dependency)
- All file paths should use `pathlib.Path` for cross-platform compatibility
- Chinese comments are present in some files; maintain bilingual clarity where appropriate
- Follow existing patterns for adding new format handlers or converters
- The actual LLM used for this project is DeepSeek-V3.2; use `Co-Authored-By: DeepSeek-V3.2 <noreply@deepseek.com>` in commit messages
- When modifying logging behavior, ensure both console and file logging work correctly in verbose mode

## Related Documentation

### Specifications
- `docs/specs/`: Detailed specifications for each module (in Chinese)
  - `specs_for_label.md`: Label module specification
  - `specs_for_convert.md`: Converter specification
  - `specs_for_visualize.md`: Visualizer specification

### Examples
- `samples/`: Comprehensive usage examples
  - `samples/visualize/`: Visualization demos
  - `samples/convert/`: Conversion examples
  - `samples/label/`: Handler usage examples
  - `samples/cli/`: CLI usage patterns

### Test Data
- `assets/`: Sample data for testing and demonstrations
  - `assets/test_data/`: Organized by format (det/seg) and annotation type

### Project History
- `CHANGELOG.md`: Version history and breaking changes