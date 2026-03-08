# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataFlow is a Python library for computer vision dataset processing, focusing on format conversion and visualization between LabelMe, COCO, and YOLO formats. It provides both a CLI and Python API. The project is in alpha stage (Development Status :: 3 - Alpha).

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

### Installation
```bash
# Install in development mode (editable)
pip install -e .

# Install with full optional dependencies (pycocotools, torch, torchvision)
pip install -e .[full]

# Install directly from source
python setup.py develop
```

### Running Tests
```bash
# Run all tests
python tests/run_tests.py

# Run specific test class
python tests/run_tests.py --test TestCocoToYoloConverter

# Run specific test method
python tests/run_tests.py --test TestCocoToYoloConverter.test_successful_conversion

# Verbose output
python tests/run_tests.py -v

# Quiet mode
python tests/run_tests.py -q
```

### Building Distribution
```bash
# Build wheel and source distribution
python -m build

# Install from built wheel
pip install dist/dataflow_cv-*.whl
```

### Command Line Interface
```bash
# Show help
dataflow --help

# Convert COCO to YOLO
dataflow convert coco2yolo annotations.json output_dir/

# Convert YOLO to COCO
dataflow convert yolo2coco images/ labels/ classes.names output.json

# Show configuration
dataflow config
```

### Python API Usage
```python
import dataflow

# COCO to YOLO conversion
result = dataflow.coco_to_yolo("annotations.json", "output_dir")

# YOLO to COCO conversion
result = dataflow.yolo_to_coco("images/", "labels/", "classes.names", "output.json")
```

## Architecture and Design Patterns

### Task‑Based Structure
The library follows a **main‑task → sub‑task** pattern:
- **Main task**: A broad functional area (e.g., `convert`).
- **Sub‑task**: A specific operation within that area (e.g., `coco2yolo`, `yolo2coco`).

Each sub‑task is implemented as an independent module with its own converter class, test file, and example files.

### Converter Base Class
All format converters inherit from `BaseConverter` (`dataflow/convert/base.py`), which provides:
- Common validation utilities (`validate_input_path`, `validate_output_path`)
- File listing methods (`get_image_files`, `get_label_files`)
- Class‑file I/O (`read_classes_file`, `write_classes_file`)
- Batch‑conversion support (`batch_convert`)
- Logging and progress reporting

### Configuration Management
Global settings are centralized in `Config` (`dataflow/config.py`). CLI options (verbose, overwrite) update the config at runtime. Avoid hard‑coding file names, extensions, or default values; use the `Config` class instead.

### CLI Organization
The CLI is built with Click and structured as a command group hierarchy:
- Root command (`dataflow`) with global options (`--verbose`, `--overwrite`)
- Task‑level group (`convert`) that contains sub‑task commands (`coco2yolo`, `yolo2coco`)

Each sub‑task command validates its arguments, creates the appropriate converter, runs the conversion, and prints a summary.

### File Layout
```
dataflow/
├── __init__.py              # Package exports (coco_to_yolo, yolo_to_coco)
├── cli.py                   # Click CLI definition
├── config.py                # Config class
└── convert/                 # Conversion module
    ├── __init__.py          # Exports BaseConverter, CocoToYoloConverter, YoloToCocoConverter
    ├── base.py              # BaseConverter abstract class
    ├── coco_to_yolo.py      # CocoToYoloConverter implementation
    └── yolo_to_coco.py      # YoloToCocoConverter implementation

tests/
├── convert/
│   ├── test_coco_to_yolo.py
│   └── test_yolo_to_coco.py
└── run_tests.py            # Custom test runner

samples/
├── cli/convert/            # CLI usage examples
└── api/convert/            # Python API examples
```

## Writing Principles

1. **Task‑Sub‑Task Pattern**: Follow the `dataflow <main‑task> <sub‑task> [arguments]` structure. Each sub‑task should be a self‑contained operation with a clear purpose.

2. **Independent Implementation Files**: Each sub‑task’s converter, test, and examples are kept in separate files:
   - Converter: `dataflow/<main‑task>/<sub‑task>.py` (e.g., `coco_to_yolo.py`)
   - Test: `tests/<main‑task>/test_<sub‑task>.py` (e.g., `test_coco_to_yolo.py`)
   - CLI example: `samples/cli/<main‑task>/cli_<sub‑task>.py`
   - API example: `samples/api/<main‑task>/api_<sub‑task>.py`

   This ensures maintainability and makes it easy to add, update, or remove individual converters without affecting others.

3. **Reuse Base Infrastructure**: All new converters must inherit from `BaseConverter` and leverage its utility methods. Do not duplicate file‑system operations, validation, or logging.

4. **Configuration‑Driven Defaults**: Use `Config` for all default values (file extensions, directory names, image dimensions). Allow CLI options to override these defaults where appropriate.

5. **Consistent Return Values**: Each `convert` method should return a dictionary with standardized keys (e.g., `images_processed`, `annotations_processed`) to enable uniform summary reporting.

6. **Error Handling with Logging**: Use the `self.logger` provided by the base class for warnings and errors. Raise `ValueError` for invalid inputs, but catch internal exceptions and log them appropriately.

7. **Batch‑First Design**: Converters should support both single‑file and batch conversion via the `batch_convert` method. The CLI calls the single‑file `convert` method; batch support is available through the Python API.

## Notes
- The AI model used in this project is DeepSeek-V3.2 (128K context length), not Claude Opus.
- The library is Linux‑oriented (assumes POSIX paths).
- The project is in alpha; the API and CLI may change.
- Visualization modules (LabelMe, COCO, YOLO) have been removed; focus is currently on conversion between COCO and YOLO formats.