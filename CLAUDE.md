# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataFlow is a Python library for computer vision dataset processing, focusing on format conversion and visualization between LabelMe, COCO, and YOLO formats. It provides both a CLI and Python API. The project is in alpha stage (Development Status :: 3 - Alpha).

## Development Commands

### Installation
```bash
# Install in development mode
pip install -e .

# Install with optional dependencies (pycocotools, torch, torchvision)
pip install -e .[full]
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
  - `base.py` – `BaseConverter` with common utilities (bbox conversion, normalization, JSON/TXT I/O)
  - Six concrete converters: `*_to_*.py` implementing specific format transformations
  - All converters are imported in `__init__.py` for easy access via `dataflow.convert.*`
- **`dataflow/visualize/`** – Annotation visualization
  - `base.py` – `BaseVisualizer` with common drawing utilities
  - Three visualizers: `coco_vis.py`, `yolo_vis.py`, `labelme_vis.py`
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
- `tests/` – Mirror of the main package structure; each test creates temporary files and cleans up

## Working with the Codebase

### Adding a New Converter
1. Create `dataflow/convert/newformat_to_targetformat.py`
2. Implement conversion logic (use `BaseConverter` helpers)
3. Add the function to `dataflow/convert/__init__.py`
4. Add CLI command in `dataflow/cli.py` under the `@convert_cmd.group`
5. Write corresponding test in `tests/convert/test_convert.py`

### Adding a New Visualizer
1. Create `dataflow/visualize/newformat_vis.py`
2. Implement `visualize_*` function that returns an OpenCV image (BGR)
3. Add to `dataflow/visualize/__init__.py`
4. Add CLI command in `dataflow/cli.py` under `@visualize.group`
5. Write test in `tests/visualize/test_visualize.py`

### Batch Visualization Implementation
The batch visualization feature extends existing `visualize` subcommands with a `--batch` flag:

1. **CLI Structure**: Add `@click.option('--batch', is_flag=True, help='Batch mode: process directories instead of single files')` to visualization commands
2. **File Matching**: Use `find_matching_pairs()` from `dataflow/visualize/batch.py` to match images and annotations by basename
3. **Navigation**: `BaseVisualizer.show_batch_navigation()` provides interactive controls (←/→ arrows, 'q' to quit)
4. **Batch Processing**:
   - For COCO/LabelMe: Match `.jpg/.png` images with `.json` annotations
   - For YOLO: Match `.jpg/.png` images with `.txt` annotations, class names from single file
   - Save mode: If `--save` path is directory, outputs are saved as `{basename}_vis.jpg`
5. **Error Handling**: Skip files with errors, continue processing others with warnings

### Configuration Updates
Modify `DEFAULT_CONFIG` in `config.py`. All settings are grouped under `visualization`, `conversion`, `paths`, or `batch`. Changes apply globally.

### Testing Philosophy
Tests are self‑contained scripts that create temporary images and annotation files, run the conversion/visualization, and verify outputs. They do not rely on external data. Keep tests independent and clean up after themselves.

## Notes
- The library is Linux‑oriented (assumes POSIX paths).
- OpenCV is required for visualization; Pillow is required for image size detection.
- Optional dependencies (`pycocotools`, `torch`, `torchvision`) are only needed for extended functionality (marked as `full` extra).