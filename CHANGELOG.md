# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Fixed
- None

## [0.1.0] - 2026-03-06

### Added
- Initial release with format conversion between LabelMe, COCO, and YOLO formats
- Batch conversion support for all conversion directions
- Single-image and batch visualization
- Command-line interface with `convert` and `visualize` subcommands
- Python API for programmatic access