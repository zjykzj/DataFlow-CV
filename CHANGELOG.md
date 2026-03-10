# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-03-11

### Added
- Complete LabelMe format conversion support with bidirectional conversion between LabelMe, COCO, and YOLO formats
- Added `coco2labelme`, `labelme2coco`, `labelme2yolo`, and `yolo2labelme` conversion commands
- Added LabelMe visualization module (`dataflow visualize labelme`) and Python API function `visualize_labelme()`
- Added comprehensive LabelMe conversion examples in `samples/` directory

### Changed
- Refactored conversion module structure for better maintainability and extensibility
- Updated module organization following task-sub-task pattern

### Fixed
- Fixed image path resolution in LabelMeVisualizer

### Documentation
- Updated README Installation section with detailed instructions
- Added LabelMe conversion examples and documentation

## [0.2.1] - 2026-03-10

### Changed
- Updated version from 0.2.0 to 0.2.1

## [0.2.0] - 2026-03-08

### Added
- Added YOLO and COCO visualization module with CLI support (`dataflow visualize yolo` and `dataflow visualize coco`)
- Added Python API convenience functions: `visualize_yolo()` and `visualize_coco()`
- Added comprehensive visualization examples in `samples/` directory

### Changed
- Refactored COCO-YOLO conversion to be batch-first with new interfaces
- Improved command-line interface options and error handling
- Cleaned up and consolidated codebase structure
- Updated version from 0.1.1 to 0.2.0

### Fixed
- Fixed CLI interface options and improved user experience

### Documentation
- Updated CLAUDE.md with detailed usage and architecture guidelines
- Added DeepSeek-V3.2 AI model information to git commit guidelines
- Updated git commit guidelines with proper AI model attribution

## [0.1.1] - 2026-03-07

### Added
- Added `setup.py` to enable editable installations via `python setup.py develop`
- Added CHANGELOG.md file

### Changed
- Updated version from 0.1.0 to 0.1.1
- Updated dependency versions:
  - `numpy>=2.0.0` (from >=1.19.0)
  - `opencv-python>=4.8.0` (from >=4.5.0)
  - `Pillow>=10.0.0` (from >=8.0.0)
  - `click>=8.1.0` (from >=8.0.0)
- Updated installation documentation in README.md and CLAUDE.md:
  - Added `python setup.py develop` for editable installation
  - Changed `pip install -e .` to `pip install .` for regular installation
- Updated `pyproject.toml`:
  - Fixed license format to use SPDX expression
  - Removed deprecated license classifier
  - Limited setuptools version to <70 for compatibility

### Fixed
- Fixed build warnings related to license format and classifiers
- Fixed license configuration in pyproject.toml (use `{text = "MIT"}` format)
- Added custom develop command to enable `python setup.py develop` for editable installation
- Note: `pip install -e .` may not work due to setuptools compatibility issues; use `python setup.py develop` instead
  - With editable installation, use `python -m dataflow.cli` instead of `dataflow` command

## [0.1.0] - 2026-03-06

### Added
- Initial release with format conversion between LabelMe, COCO, and YOLO formats
- Batch conversion support for all conversion directions
- Single-image and batch visualization
- Command-line interface with `convert` and `visualize` subcommands
- Python API for programmatic access