"""
CLI module for DataFlow-CV.

This module provides a command-line interface for the DataFlow-CV library,
offering convenient access to visualization and format conversion functionality
through a unified command-line interface.

The CLI follows a `<main-task> <sub-task>` structure:
- `dataflow-cv visualize <format>`: Visualize annotations in YOLO, COCO, or LabelMe format
- `dataflow-cv convert <direction>`: Convert between six annotation format combinations

Example usage:
    $ dataflow-cv visualize yolo path/to/yolo/labels --image-dir path/to/images
    $ dataflow-cv convert yolo2coco path/to/yolo path/to/output.json

Key features:
- Thin wrapper over existing API (no business logic duplication)
- Consistent error handling and logging
- Comprehensive help system
- Cross-platform compatibility
"""

from .main import cli

__all__ = ["cli"]