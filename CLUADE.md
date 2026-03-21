# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataFlow-CV is a Python library for computer vision dataset processing. The project focuses on providing unified label file format conversion, visualization, and data processing functionality.

## Platform Compatibility

DataFlow-CV is designed for full cross-platform compatibility (Windows, Linux, macOS).

## Key Design Principles

1. **Unified Interfaces**: All annotation handlers follow the same interface defined in `BaseAnnotationHandler`
2. **Type Safety**: Extensive use of Python dataclasses with type hints
3. **Cross-platform Compatibility**: Use `pathlib.Path` for all path operations, enforce UTF-8 encoding
4. **Error Handling**: Strict mode by default with informative error messages
5. **Testing**: Comprehensive unit tests with ≥90% coverage requirement