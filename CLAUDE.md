# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataFlow is a Python library for computer vision dataset processing, focusing on format conversion and visualization between LabelMe, COCO, and YOLO formats. It provides both a CLI and Python API. The project is in alpha stage (Development Status :: 3 - Alpha).

## Development Commands

### Installation
```bash
# Regular installation
pip install .

# For development (editable installation)
python setup.py develop
# Note: pip install -e . may not work due to setuptools compatibility issues

# Install with optional dependencies (pycocotools, torch, torchvision)
pip install .[full]
```

### Running Tests
Tests are organized as standalone scripts in the `tests/` directory. Run them directly:

```bash
# Convert module tests
python -m tests.convert.test_convert

# Visualization module tests
python -m tests.visualize.test_visualize

# CLI tests
python -m tests.cli.test_cli
```

Each test script returns exit code 0 on success, 1 on failure.

### Building Distribution
```bash
# Build wheel and source distribution
python -m build

# Clean build artifacts
rm -rf build/ dist/ *.egg-info/
```

## Architecture

### Core Modules
- **`dataflow/convert/`** – Format conversion between LabelMe, COCO, YOLO
  - `base.py` – `BaseConverter` with common utilities (bbox conversion, normalization, JSON/TXT I/O, batch utilities)
  - `batch.py` – Batch conversion utilities (`batch_process_conversion`, `batch_convert_with_combined_option`, `find_matching_conversion_pairs`)
  - Six concrete converters: `*_to_*.py` implementing specific format transformations
  - Six batch converters: `batch_*` functions for each conversion direction
  - All functions are imported in `__init__.py` for easy access via `dataflow.convert.*`
- **`dataflow/visualize/`** – Annotation visualization
  - `base.py` – `BaseVisualizer` with common drawing utilities and batch navigation
  - Three visualizers: `coco_vis.py`, `yolo_vis.py`, `labelme_vis.py`
  - `batch.py` – Batch processing utilities (`find_matching_pairs`, `batch_process_images`)
  - Uses OpenCV for drawing; imports `cv2` only when needed
- **`dataflow/cli.py`** – Click‑based CLI with two command groups:
  - `convert` – subcommands for each conversion direction (e.g., `coco2yolo`, `labelme2coco`)
  - `visualize` – subcommands for each format (`coco`, `yolo`, `labelme`)
- **`dataflow/config.py`** – Central configuration dictionary (`DEFAULT_CONFIG`) for visualization colors/thickness, conversion defaults, and paths.

### Design Patterns
- **Base classes** (`BaseConverter`, `BaseVisualizer`) provide shared functionality; concrete implementations inherit or compose them.
- **Configuration** is accessed via `get_config()` returning a copy of `DEFAULT_CONFIG`.
- **Error handling** in CLI commands catches exceptions, prints user‑friendly messages, and exits with code 1.
- **Image I/O** uses PIL (Pillow) preferentially, falls back to OpenCV if Pillow is unavailable.
- **Coordinate systems**:
  - COCO uses `[x1, y1, width, height]` (top‑left corner)
  - YOLO uses normalized `[xc, yc, width, height]` (center)
  - LabelMe uses polygon points or rectangle `[[x1, y1], [x2, y2]]`
- **Class names** are passed as lists; YOLO conversions require them for mapping category IDs.

### Important File Locations
- `pyproject.toml` – Project metadata, dependencies, entry‑point (`dataflow = dataflow.cli:main`)
- `README.md` – User documentation with CLI examples and Python API snippets
- `examples/` – Example scripts for conversion and visualization
  - `convert/basic_convert.py` – Single-file conversion examples
  - `convert/batch_convert.py` – Batch conversion examples
  - `visualize/basic_visualize.py` – Single-image visualization examples
- `tests/` – Mirror of the main package structure; each test creates temporary files and cleans up

## Working with the Codebase

### Adding a New Converter
1. Create `dataflow/convert/newformat_to_targetformat.py`
2. Implement conversion logic (use `BaseConverter` helpers)
3. Add the function to `dataflow/convert/__init__.py`
4. Add CLI command in `dataflow/cli.py` under the `@convert.group`
5. Write corresponding test in `tests/convert/test_convert.py`

### Adding a New Visualizer
1. Create `dataflow/visualize/newformat_vis.py`
2. Implement `visualize_*` function that returns an OpenCV image (BGR)
3. Add to `dataflow/visualize/__init__.py`
4. Add CLI command in `dataflow/cli.py` under `@visualize.group`
5. Write test in `tests/visualize/test_visualize.py`

### Adding Batch Support to a Converter
To add batch processing to an existing or new converter:

1. **Add batch utility functions** to `dataflow/convert/base.py` if needed:
   - `find_matching_conversion_pairs()`: For matching input-annotation pairs
   - `validate_conversion_directories()`: For directory validation
   - `get_image_size_from_source()`: For image dimension extraction

2. **Create batch function** in the converter file:
   - Name: `batch_[source]_to_[target]()` (e.g., `batch_coco_to_yolo`)
   - Parameters: File pairs list, required arguments, output path
   - Implementation: Process each pair with error handling and progress display
   - For COCO output: Support both per-file and combined modes

3. **Update converter `__init__.py`**:
   - Import the batch function
   - Add to `__all__` list

4. **Update CLI command** in `dataflow/cli.py`:
   - Add `@click.option('--batch', is_flag=True, ...)`
   - For COCO output: Add `@click.option('--combined', is_flag=True, ...)`
   - Implement batch mode logic:
     - Validate directories
     - Find matching pairs
     - Determine output mode (directory vs. file)
     - Call appropriate batch function
   - Maintain single-file mode compatibility

5. **Test batch functionality**:
   - Create test directories with multiple files
   - Test both per-file and combined output modes (if applicable)
   - Test error handling (skip invalid files)

### Batch Visualization Implementation
The batch visualization feature extends existing `visualize` subcommands with a `--batch` flag. Key components:

- **CLI Structure**: Add `@click.option('--batch', is_flag=True, help='Batch mode: process directories instead of single files')` to visualization commands
- **File Matching**: `find_matching_pairs()` from `dataflow/visualize/batch.py` matches images and annotations by basename (supports common image extensions)
- **Validation**: `validate_batch_directories()` ensures directories exist and contain relevant files
- **Navigation**: `BaseVisualizer.show_batch_navigation()` provides interactive controls (←/→ arrows, 'a'/'d' keys, 'q' to quit) with progress overlay
- **Batch Processing**: `batch_process_images()` orchestrates the batch loop, handling errors, saving outputs, and navigation
  - For COCO/LabelMe: Match `.jpg/.png` images with `.json` annotations
  - For YOLO: Match `.jpg/.png` images with `.txt` annotations, class names from single file
  - Save mode: If `--save` path is directory, outputs are saved as `{basename}_vis.jpg`
- **Error Handling**: Skip files with errors, continue processing others with warnings
- **Recent Improvements**: YOLO visualization fixes ensure correct bounding box drawing and batch navigation works smoothly

### Batch Conversion Implementation
The batch conversion feature extends all 6 conversion commands with `--batch` flag. Key components:

- **CLI Structure**: Add `@click.option('--batch', is_flag=True, help='Batch mode: process directories instead of single files')` to conversion commands. For COCO output formats, add `@click.option('--combined', is_flag=True, help='In batch mode, combine all annotations into a single COCO file')`
- **File Matching**: `find_matching_conversion_pairs()` from `dataflow/convert/batch.py` matches files based on conversion needs (images + annotations, or annotations only)
- **Validation**: `validate_conversion_directories()` ensures directories exist and contain relevant files
- **Batch Processing**: `batch_process_conversion()` orchestrates the batch loop with progress display and error skipping
  - For conversions with images: Match images and annotations by filename
  - For conversions without images: Process annotation files directly
  - Output modes: Per-file output (directory) or combined output (single file, COCO only)
- **Conversion-Specific Functions**: Each converter has a corresponding `batch_*` function (e.g., `batch_coco_to_yolo`, `batch_labelme_to_coco`)
- **Error Handling**: Skip files with errors, continue processing others with progress reporting
- **Flexible Arguments**: Support both image-based and image-free conversions with appropriate CLI parameter validation

### Configuration Updates
Modify `DEFAULT_CONFIG` in `config.py`. All settings are grouped under `visualization`, `conversion`, `paths`, or `batch`. Changes apply globally.

### Testing Philosophy
Tests are self‑contained scripts that create temporary images and annotation files, run the conversion/visualization, and verify outputs. They do not rely on external data. Keep tests independent and clean up after themselves.

## Notes
- The library is Linux‑oriented (assumes POSIX paths).
- OpenCV is required for visualization; Pillow is required for image size detection.
- Optional dependencies (`pycocotools`, `torch`, `torvision`) are only needed for extended functionality (marked as `full` extra).
- Recent improvements include batch conversion support for all format conversions, YOLO visualization fixes (correct bounding box drawing), and enhanced batch navigation.