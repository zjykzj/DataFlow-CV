# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-03-14

### Added
- **Cross-platform compatibility**: Full Windows, Linux, macOS support with no platform-specific code
- **Enhanced documentation**: Updated CLI and Python API examples with segmentation support flags
- **Simplified development**: Consolidated dependency files and version constraints
- **Command alias**: Added `dataflow-cv` command alias for the existing `dataflow` command

### Changed
- **Dependency versions**: Lowered minimum versions for better compatibility with older environments
- **CLI architecture**: Modularized CLI structure for better maintainability
- **Platform compatibility**: Enhanced cross-platform support and removed all Unix-specific paths

### Fixed
- **Windows compatibility**: Fixed platform-specific code and path handling for Windows
- **Documentation**: Updated README.md and CLAUDE.md with latest features and cross-platform guidelines

### Documentation
- Added cross-platform development principles to documentation
- Updated examples with `--segmentation` flag usage for all visualization commands
- Enhanced CLAUDE.md with recent improvements and development guidelines

## [0.4.0] - 2026-03-13

### ⚠️ Breaking Changes
- **API change**: `coco_to_yolo()` function signature changed from `(coco_json_path, classes_path, output_dir)` to `(coco_json_path, output_dir)` (auto-generates class.names)
- **API change**: `labelme_to_yolo()` function signature changed from `(label_dir, output_dir)` to `(label_dir, classes_path, output_dir)`
- Unified YOLO conversion behavior across COCO and LabelMe converters

### Added
- YOLO visualizer debug logging for class extraction and color assignment
- Enhanced color distinction for many classes in visualization
- CLI option `-v` as shorthand for `--version` when used alone
- Documentation updates: architecture clarification and development guidelines

### Changed
- Unified COCO and LabelMe to YOLO conversion behavior
- Updated YOLO detection annotation to LabelMe rectangle conversion
- Improved test parameter calls and directory expectations for LabelMe to YOLO conversion

### Fixed
- YOLO detection annotations correctly convert to LabelMe rectangles
- Test parameter calls and directory expectations in LabelMe to YOLO tests

### Documentation
- Updated CLAUDE.md with `-v` dual-purpose note
- Added "Development" section to README.md linking to CLAUDE.md
- Updated CLAUDE.md with architecture details about label handlers and unified format

## [0.3.1] - 2026-03-11

### Added
- Added PyPI version badge and GitHub Actions build status badge to README.md
- Added GitHub Actions workflow for PyPI publishing (python-publish.yml)
- Added detailed YOLO, LabelMe, and COCO format documentation in docs/ directory

### Changed
- Updated GitHub workflow name from "Upload Python Package" to "Publish"
- Updated version from 0.3.0 to 0.3.1

### Fixed
- Fixed color assignment in LabelMe visualization to assign distinct colors per class (previously all annotations used same color)

### Documentation
- Added Documentation section to table of contents in README.md
- Updated README.md and CLAUDE.md with latest project structure and segmentation support
- Added generic.py to visualize module documentation
- Added all converter test files to project structure

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