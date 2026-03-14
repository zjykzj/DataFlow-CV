# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataFlow is a Python library for computer vision dataset processing, focusing on format conversion and visualization between LabelMe, COCO, and YOLO formats. It provides both a CLI and Python API. The project is in alpha stage (Development Status :: 3 - Alpha). The `assets/` directory contains sample images and test data for demonstration purposes, including COCO format examples, segmentation examples, and sample images. Usage examples can be found in `samples/`.

## Git Commits

When creating git commits via Claude Code, avoid using the default "Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>" line. Instead, use the following format (optionally including a Co-Authored-By line for DeepSeek):

```bash
git commit -m "$(cat <<'EOF'
<type>(<scope>): <subject>

<body if needed>

Co-Authored-By: DeepSeek-V3.2 <noreply@deepseek.com>
EOF
)"
```

The Co-Authored-By line is optional and can be omitted if desired.

Follow conventional commit style:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `build`: Build system or external dependencies
- `test`: Adding missing tests or correcting existing tests
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, etc.)
- `perf`: Code change that improves performance
- `ci`: Changes to CI configuration files and scripts
- `chore`: Other changes that don't modify src or test files

The AI model used in this project is DeepSeek-V3.2 (128K context length), not Claude Opus.

## Common Development Commands

### Running Tests

The project uses a custom test runner (`tests/run_tests.py`) built on unittest. It supports discovery patterns and specific test selection.

```bash
# Run all tests
python tests/run_tests.py

# Run tests matching a pattern (e.g., test_coco_to_yolo.py)
python tests/run_tests.py --pattern test_coco_to_yolo.py

# Run tests from a specific directory
python tests/run_tests.py --dir tests/convert/

# Run specific test class
python tests/run_tests.py --test TestCocoToYoloConverter

# Run specific test method
python tests/run_tests.py --test TestCocoToYoloConverter.test_successful_conversion

# Verbose output
python tests/run_tests.py -v

# Quiet mode (minimal output)
python tests/run_tests.py -q
```

### Installation
```bash
# Regular installation from source
pip install .
# After installation, both dataflow and dataflow-cv commands are available

# Editable installation (development mode)
# Due to setuptools compatibility, use python setup.py develop (not pip install -e .)
python setup.py develop
# After editable installation, use python -m dataflow.cli instead of the dataflow command
# (both dataflow and dataflow-cv commands may not work with editable installation)
```

### Build System

The project uses setuptools with a `pyproject.toml` configuration. Distribution packages are built with `python -m build`.

```bash
# Build wheel and source distribution
python -m build

# Install from built wheel
pip install dist/dataflow_cv-*.whl
```

### Command Line Interface
Command line options: `-v/--verbose` is available as a local option for each subcommand. The `--overwrite` option has been removed and is not supported in the current version.

Both `dataflow` and `dataflow-cv` commands are available and can be used interchangeably (e.g., `dataflow-cv convert coco2yolo ...`).
```bash
# Show help
dataflow --help

# Convert COCO to YOLO (use --segmentation for polygon annotations)
dataflow convert coco2yolo annotations.json output_dir/
dataflow convert coco2yolo annotations.json output_dir/ --segmentation

# Convert YOLO to COCO
dataflow convert yolo2coco images/ labels/ classes.names output.json

# Convert COCO to LabelMe (use --segmentation for polygon annotations)
dataflow convert coco2labelme annotations.json output_dir/
dataflow convert coco2labelme annotations.json output_dir/ --segmentation

# Convert LabelMe to COCO
dataflow convert labelme2coco labels/ classes.names output.json

# Convert LabelMe to YOLO (use --segmentation for polygon annotations)
dataflow convert labelme2yolo labels/ output_dir/
dataflow convert labelme2yolo labels/ output_dir/ --segmentation

# Convert YOLO to LabelMe
dataflow convert yolo2labelme images/ labels/ classes.names output_dir/

# Visualize YOLO annotations (use --save to export images, --segmentation for strict segmentation mode)
dataflow visualize yolo images/ labels/ classes.names
dataflow visualize yolo images/ labels/ classes.names --save output_dir/
dataflow visualize yolo images/ labels/ classes.names --segmentation

# Visualize COCO annotations (use --save to export images, --segmentation for strict segmentation mode)
dataflow visualize coco images/ annotations.json
dataflow visualize coco images/ annotations.json --save output_dir/
dataflow visualize coco images/ annotations.json --segmentation

# Visualize LabelMe annotations (use --save to export images, --segmentation for strict segmentation mode)
dataflow visualize labelme images/ labels/
dataflow visualize labelme images/ labels/ --save output_dir/
dataflow visualize labelme images/ labels/ --segmentation

# Show configuration
dataflow config
```

### Python API Usage
```python
import dataflow

# COCO to YOLO conversion (pass segmentation=True for polygon annotations)
result = dataflow.coco_to_yolo("annotations.json", "output_dir")
result = dataflow.coco_to_yolo("annotations.json", "output_dir", segmentation=True)

# YOLO to COCO conversion
result = dataflow.yolo_to_coco("images/", "labels/", "classes.names", "output.json")

# COCO to LabelMe conversion (pass segmentation=True for polygon annotations)
result = dataflow.coco_to_labelme("annotations.json", "output_dir")
result = dataflow.coco_to_labelme("annotations.json", "output_dir", segmentation=True)

# LabelMe to COCO conversion
result = dataflow.labelme_to_coco("labels/", "classes.names", "output.json")

# LabelMe to YOLO conversion (pass segmentation=True for polygon annotations)
result = dataflow.labelme_to_yolo("labels/", "output_dir")
result = dataflow.labelme_to_yolo("labels/", "output_dir", segmentation=True)

# YOLO to LabelMe conversion
result = dataflow.yolo_to_labelme("images/", "labels/", "classes.names", "output_dir")

# Visualize YOLO annotations (save_dir is optional, segmentation=True for strict segmentation mode)
result = dataflow.visualize_yolo("images/", "labels/", "classes.names")
result = dataflow.visualize_yolo("images/", "labels/", "classes.names", save_dir="output_dir/")
result = dataflow.visualize_yolo("images/", "labels/", "classes.names", segmentation=True)

# Visualize COCO annotations (save_dir is optional, segmentation=True for strict segmentation mode)
result = dataflow.visualize_coco("images/", "annotations.json")
result = dataflow.visualize_coco("images/", "annotations.json", save_dir="output_dir/")
result = dataflow.visualize_coco("images/", "annotations.json", segmentation=True)

# Visualize LabelMe annotations (save_dir is optional, segmentation=True for strict segmentation mode)
result = dataflow.visualize_labelme("images/", "labels/")
result = dataflow.visualize_labelme("images/", "labels/", save_dir="output_dir/")
result = dataflow.visualize_labelme("images/", "labels/", segmentation=True)
```

### Version Compatibility

DataFlow-CV is compatible with the following minimum dependency versions:
- `numpy>=1.24.0` (December 2022, supports Python 3.8+)
- `opencv-python>=4.6.0.66` (October 2022 release)
- `Pillow>=8.0.0` (January 2021)
- `click>=7.0.0` (2018)

All tests pass with these versions, and the library uses only basic functionality from each dependency. The previous higher version requirements have been lowered to increase compatibility with older environments.

## Architecture and Design Patterns

### Task‑Based Structure
The library follows a **main‑task → sub‑task** pattern:
- **Main task**: A broad functional area (e.g., `convert`, `visualize`).
- **Sub‑task**: A specific operation within that area (e.g., `coco2yolo`, `yolo2coco`, `yolo`, `coco`, `labelme`).

Each sub‑task is implemented as an independent module with its own converter/visualizer class, test file, and example files.

### Converter Base Class
All format converters inherit from `BaseConverter` (`dataflow/convert/base.py`), which provides:
- Common validation utilities (`validate_input_path`, `validate_output_path`)
- File listing methods (`get_image_files`, `get_label_files`)
- Class‑file I/O (`read_classes_file`, `write_classes_file`)
- Batch‑conversion support (`batch_convert`)
- Logging and progress reporting

### Visualizer Base Class
All annotation visualizers inherit from `BaseVisualizer` (`dataflow/visualize/base.py`), which provides:
- Common drawing utilities (`draw_bounding_box`, `draw_polygon`)
- Color management (`get_color_for_class`)
- Image I/O (`read_image`, `save_image`, `display_image`)
- Window management and display resizing
- Logging and progress reporting

### Label Handlers and Unified Format

The `label/` module provides format-specific handlers (`YoloHandler`, `CocoHandler`, `LabelMeHandler`) that parse and serialize annotation files. Each handler produces a **unified format**—a common dictionary structure used by converters and visualizers. This enables consistent processing across different annotation formats.

**Unified format keys**:
- `image_path`: Path to the image file
- `image_id`: Unique identifier for the image
- `annotations`: List of annotation dictionaries, each containing:
  - `category_id`: Integer class ID
  - `category_name`: String class name
  - `bbox`: `[x_min, y_min, width, height]` (optional)
  - `segmentation`: List of polygon coordinate lists (optional)

Converters inherit from `LabelBasedConverter` (which itself extends `BaseConverter`) and use label handlers to read source annotations and write target formats. Visualizers inherit from `GenericVisualizer` (which extends `BaseVisualizer`) and use label handlers to load annotations for drawing.

### Configuration Management
Global settings are centralized in `Config` (`dataflow/config.py`). CLI options (verbose) update the config at runtime. Avoid hard‑coding file names, extensions, or default values; use the `Config` class instead.

Module‑specific configurations are provided by `ConvertConfig` (`dataflow/convert/config.py`) and `VisualizeConfig` (`dataflow/visualize/config.py`), which inherit from the global `Config` and add module‑specific defaults. The `dataflow config` command displays both global and module configurations.

### CLI Organization
The CLI is built with Click and follows a modular architecture:
- Root command (`dataflow`) with global options (`--version`, `--help`)
- Task‑level groups (`convert`, `visualize`) are dynamically imported from their respective modules (`dataflow.convert.cli`, `dataflow.visualize.cli`)
- Each module provides a `create_<module>_group()` function that returns a Click command group with its sub‑task commands (`coco2yolo`, `yolo2coco`, `yolo`, `coco`, `labelme`)

Each sub‑task command validates its arguments, creates the appropriate converter/visualizer, runs the operation, and prints a summary. Missing modules are handled gracefully with informative error messages.

### File Layout
```
dataflow/
├── __init__.py              # Package exports (coco_to_yolo, yolo_to_coco, coco_to_labelme, labelme_to_coco, yolo_to_labelme, labelme_to_yolo, visualize_*)
├── cli.py                   # Click CLI definition
├── config.py                # Config class
├── convert/                 # Format conversion module
│   ├── __init__.py          # Exports BaseConverter, all converter classes, and CLI functions
│   ├── base.py              # BaseConverter abstract class
│   ├── coco_and_yolo.py     # COCO ↔ YOLO converters (CocoToYoloConverter, YoloToCocoConverter)
│   ├── coco_and_labelme.py  # COCO ↔ LabelMe converters (CocoToLabelMeConverter, LabelMeToCocoConverter)
│   ├── yolo_and_labelme.py  # YOLO ↔ LabelMe converters (YoloToLabelMeConverter, LabelMeToYoloConverter)
│   ├── cli.py               # Convert module CLI commands
│   └── config.py            # Convert module configuration
├── label/                   # Label format handlers module
│   ├── __init__.py          # Exports YoloHandler, CocoHandler, LabelMeHandler
│   ├── yolo.py              # YOLO format handler
│   ├── coco.py              # COCO format handler
│   └── labelme.py           # LabelMe format handler
└── visualize/               # Annotation visualization module
    ├── __init__.py          # Exports BaseVisualizer, YoloVisualizer, CocoVisualizer, LabelMeVisualizer, GenericVisualizer, and CLI functions
    ├── base.py              # BaseVisualizer abstract class
    ├── generic.py           # Generic visualizer base class using label handlers
    ├── yolo.py              # YOLO annotation visualizer
    ├── coco.py              # COCO annotation visualizer
    ├── labelme.py           # LabelMe annotation visualizer
    ├── cli.py               # Visualize module CLI commands
    └── config.py            # Visualize module configuration

tests/
├── __init__.py
├── convert/                # Conversion tests
│   ├── __init__.py
│   ├── test_coco_to_yolo.py
│   ├── test_yolo_to_coco.py
│   ├── test_coco_to_labelme.py
│   ├── test_labelme_to_coco.py
│   ├── test_labelme_to_yolo.py
│   └── test_yolo_to_labelme.py
├── visualize/              # Visualization tests
│   ├── __init__.py
│   ├── test_yolo.py
│   ├── test_coco.py
│   ├── test_labelme.py
│   └── test_generic.py    # Generic visualizer tests
└── run_tests.py           # Test runner

samples/
├── __init__.py
├── example_usage.py       # Quick usage demonstration
├── template.py            # Example template for creating new examples
├── cli/                   # CLI usage examples
│   ├── __init__.py
│   ├── convert/
│   │   ├── cli_coco_to_yolo.py
│   │   ├── cli_yolo_to_coco.py
│   │   ├── cli_coco_to_labelme.py
│   │   ├── cli_labelme_to_coco.py
│   │   ├── cli_labelme_to_yolo.py
│   │   └── cli_yolo_to_labelme.py
│   └── visualize/
│       ├── cli_yolo.py
│       ├── cli_coco.py
│       └── cli_labelme.py
└── api/                   # Python API examples
    ├── __init__.py
    ├── convert/
    │   ├── api_coco_to_yolo.py
    │   ├── api_yolo_to_coco.py
    │   ├── api_coco_to_labelme.py
    │   ├── api_labelme_to_coco.py
    │   ├── api_labelme_to_yolo.py
    │   └── api_yolo_to_labelme.py
    └── visualize/
        ├── api_yolo.py
        ├── api_coco.py
        └── api_labelme.py
docs/                          # Data format documentation
├── README.md                 # Documentation index
├── yolo.md                   # YOLO format specification
├── labelme.md                # LabelMe format specification
└── coco.md                   # COCO format specification
```

## Writing Principles

1. **Task‑Sub‑Task Pattern**: Follow the `dataflow <main‑task> <sub‑task> [arguments]` structure. Each sub‑task should be a self‑contained operation with a clear purpose.

2. **Independent Implementation Files**: Each sub‑task’s converter/visualizer, test, and examples are kept in separate files:
   - Converter/Visualizer: `dataflow/<main‑task>/<sub‑task>.py` (e.g., `coco_to_yolo.py`, `yolo.py`)
   - Test: `tests/<main‑task>/test_<sub‑task>.py` (e.g., `test_coco_to_yolo.py`)
   - CLI example: `samples/cli/<main‑task>/cli_<sub‑task>.py`
   - API example: `samples/api/<main‑task>/api_<sub‑task>.py`

   This ensures maintainability and makes it easy to add, update, or remove individual components without affecting others.

3. **Reuse Base Infrastructure**: All new converters must inherit from `BaseConverter` (or `LabelBasedConverter`) and all new visualizers from `BaseVisualizer` (or `GenericVisualizer`). Use the label handlers (`YoloHandler`, `CocoHandler`, `LabelMeHandler`) for format-specific I/O. Leverage their utility methods. Do not duplicate file‑system operations, validation, or logging.

4. **Configuration‑Driven Defaults**: Use `Config` for all default values (file extensions, directory names, image dimensions). Allow CLI options to override these defaults where appropriate.

5. **Consistent Return Values**: Each `convert` or `visualize` method should return a dictionary with standardized keys (e.g., `images_processed`, `annotations_processed`) to enable uniform summary reporting.

6. **Error Handling with Logging**: Use the `self.logger` provided by the base class for warnings and errors. Raise `ValueError` for invalid inputs, but catch internal exceptions and log them appropriately.

7. **Batch‑First Design**: Converters should support both single‑file and batch conversion via the `batch_convert` method. The CLI calls the single‑file `convert` method; batch support is available through the Python API.

## Segmentation Support

DataFlow-CV supports both bounding box and polygon segmentation annotations across all formats:

**YOLO Segmentation Format**
- Detection format: `class_id x_center y_center width height` (normalized coordinates)
- Segmentation format: `class_id x1 y1 x2 y2 ...` (polygon vertices, normalized)
- YOLO segmentation files have the same `.txt` extension as detection files

**COCO Segmentation Format**
- Polygon coordinates in `segmentation` field (list of `[x1, y1, x2, y2, ...]`)
- Both single-polygon and multi-polygon annotations are supported

**LabelMe Segmentation Format**
- Rectangle shapes (`shape_type: "rectangle"`) for bounding box annotations
- Polygon shapes (`shape_type: "polygon"`) for segmentation annotations
- Each JSON file contains `shapes` array with annotation data

**Usage Examples**
```bash
# Convert COCO to YOLO with segmentation annotations
dataflow convert coco2yolo annotations.json output_dir/ --segmentation

# Visualize YOLO annotations in strict segmentation mode (only polygons)
dataflow visualize yolo images/ labels/ classes.names --segmentation

# Visualize COCO annotations in strict segmentation mode
dataflow visualize coco images/ annotations.json --segmentation

# Visualize LabelMe annotations in strict segmentation mode (only polygons)
dataflow visualize labelme images/ labels/ --segmentation
```

**Python API**
```python
# Convert COCO to YOLO with segmentation
result = dataflow.coco_to_yolo("annotations.json", "output_dir", segmentation=True)

# Visualize in strict segmentation mode
result = dataflow.visualize_yolo("images/", "labels/", "classes.names", segmentation=True)
result = dataflow.visualize_coco("images/", "annotations.json", segmentation=True)
result = dataflow.visualize_labelme("images/", "labels/", segmentation=True)
```

**Notes**
- Without the `--segmentation` flag, both bounding boxes and polygons are processed automatically
- With `--segmentation` flag:
  - **YOLO to LabelMe conversion**: All annotations are converted to polygons (detection annotations become polygons from bounding boxes)
  - **Other conversions (COCO to YOLO, LabelMe to YOLO, etc.)**: Only valid polygon annotations are processed (strict mode)
- YOLO segmentation format requires at least 3 points (6 coordinates)
- COCO segmentation polygons are automatically converted to YOLO normalized coordinates
- LabelMe format supports both rectangle (`shape_type: "rectangle"`) and polygon (`shape_type: "polygon"`) shapes
- In segmentation mode, LabelMe visualizer rejects rectangle shapes and only accepts polygon shapes

## Recent Enhancements

DataFlow-CV has recently been enhanced with several improvements:

- **Simplified COCO to YOLO API**: The `coco_to_yolo` function now requires only two parameters (`coco_json_path`, `output_dir`). The `classes_path` parameter is optional and will be auto-generated in the output directory when not provided.
- **Enhanced YOLO Visualizer**: Added debug logging for troubleshooting and improved class name extraction that merges classes from annotation files with those from class names files, supporting case-insensitive matching and fallback strategies.
- **Improved Color Distinction**: Visualizers now use golden ratio distribution in HSV color space to generate distinct colors for many classes, with additional variation in saturation and value for better visual separation.
- **Cross-Platform Robustness**: All platform-specific code has been eliminated; the library now uses only standard Python libraries and follows strict cross-platform development principles.
- **Command alias**: Added `dataflow-cv` as an alias to the existing `dataflow` command for better alignment with package name.

## Platform Compatibility

DataFlow-CV is designed for full cross-platform compatibility (Windows, Linux, macOS). Recent enhancements have eliminated platform-specific code and hardcoded Unix paths.

### Core Module Compatibility
The `dataflow` core module follows strict cross-platform principles:
- Uses only standard Python libraries with no platform-specific APIs
- File operations use `os.path.join()`, `pathlib.Path`, and `shutil` modules
- Temporary files are created with `tempfile.mkdtemp()` and `tempfile.mkstemp()`
- File encoding is always specified as `encoding='utf-8'`
- Path separators are handled via `os.path.sep` and `os.path.join()`

### Samples Module
All example scripts (`samples/`) now use `create_test_paths()` helper functions that generate platform-independent temporary paths. Hardcoded Unix paths (`/tmp/`, `/root/`, `/invalid/path/`) have been replaced with `tempfile`-generated paths.

### Tests Module
- Removed all `@unittest.skipIf(platform.system() == "Windows", ...)` decorators
- Platform-dependent permission tests (`os.chmod()`) have been refactored to test invalid paths instead
- All 193 tests pass on both Linux and Windows platforms

### Development Principles for Cross-Platform Code
When modifying or extending the codebase:

1. **File System Operations**
   - Use `os.path.join()` or `pathlib.Path` for path construction (never string concatenation)
   - Use `os.path.exists()`, `os.path.isdir()`, `os.path.isfile()` for path validation
   - Use `shutil.rmtree()` with `ignore_errors=True` for safe directory removal
   - Always specify `encoding='utf-8'` when reading or writing text files
   - Use `with` statements when opening files to ensure proper resource cleanup

2. **Temporary Files and Directories**
   - Always use `tempfile.mkdtemp()` for temporary directories
   - Use `tempfile.mkstemp()` for temporary files when needed
   - Clean up temporary resources with `try/finally` blocks

3. **Platform-Specific APIs**
   - Never use Windows-specific APIs (`win32`, `ntpath`, etc.)
   - Never use Linux-specific APIs (POSIX-only functions)
   - Use `os.path` and `pathlib` which abstract platform differences
   - Avoid any platform-specific modules (e.g., `win32`, `ntpath`, `posix`, `fcntl`). Use only standard Python libraries.

4. **File Permissions and Attributes**
   - Avoid `os.chmod()` unless absolutely necessary (behavior differs across platforms)
   - Test invalid paths or file access errors rather than permission errors
   - Use `os.access()` for portable permission checks when needed

5. **Path Validation**
   - Use `os.path.abspath()` to normalize paths
   - Handle both forward and backward slashes in user input
   - Validate paths exist before operations, provide clear error messages

### Verification
- All tests pass: `python tests/run_tests.py` (193 tests, 0 failures)
- Example scripts run correctly on all platforms
- No hardcoded platform-specific paths remain in the codebase

## Notes
- The AI model used in this project is DeepSeek-V3.2 (128K context length), not Claude Opus.
- The library is fully cross-platform compatible (Windows, Linux, macOS) with no platform-specific code.
- The project is in alpha; the API and CLI may change.
- Visualization modules for COCO, YOLO, and LabelMe formats are included.
- Label format handlers (YoloHandler, CocoHandler, LabelMeHandler) provide unified format conversion.
- LabelMe conversion to/from other formats is now fully implemented (coco2labelme, labelme2coco, labelme2yolo, yolo2labelme).