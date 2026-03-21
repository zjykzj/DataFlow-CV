# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataFlow-CV is a Python library for computer vision dataset processing. The project focuses on providing unified label file format conversion, visualization, and data processing functionality.

## Platform Compatibility

DataFlow-CV is designed for full cross-platform compatibility (Windows, Linux, macOS).

## Development Specifications

The project follows the detailed development specification in `specs_for_label.md`. This document outlines:
- Overall architecture and module organization
- Detailed design of the `dataflow/label` module (LabelMe, YOLO, COCO handlers)
- Design of the `dataflow/util` module (file and logging operations)
- Technical implementation requirements and testing strategy
- Example code structure and development plan

When implementing features, always refer to `specs_for_label.md` for design guidance and API specifications.

## Key Design Principles

1. **Unified Interfaces**: All annotation handlers follow the same interface defined in `BaseAnnotationHandler`
2. **Type Safety**: Extensive use of Python dataclasses with type hints
3. **Cross-platform Compatibility**: Use `pathlib.Path` for all path operations, enforce UTF-8 encoding
4. **Error Handling**: Strict mode by default with informative error messages
5. **Testing**: Comprehensive unit tests with ≥90% coverage requirement