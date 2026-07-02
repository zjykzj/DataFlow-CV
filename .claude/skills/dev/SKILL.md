---
name: dev
description: Run development commands — install, test, lint, typecheck, and manual CLI verification. Use when the user asks to run tests, lint, type check, verify the CLI, check test coverage, or ensure samples/tests are compatible with the codebase.
allowed-tools: Bash
---

# Development Commands

## Installation

```bash
pip install -e .                    # Editable install (recommended for dev)
pip install -e .[dev]               # With test/lint deps
```

Project-specific extras and CLI entry points are documented in CLAUDE.md.

## Testing

```bash
pytest                              # All tests
pytest -x -q                        # Stop on first failure, quiet
pytest tests/label/test_yolo.py     # Single module
pytest tests/label/test_yolo.py::TestYoloAnnotationHandler::test_read_detection  # Single test
pytest --cov=dataflow --cov-report=html  # Coverage report
pytest -n auto                      # Parallel (requires pytest-xdist)
```

## Linting

```bash
black dataflow tests samples        # Format
isort dataflow tests samples        # Imports
flake8 dataflow tests samples       # Lint
mypy dataflow                       # Type check
```

Manual CLI verification commands, test structure, and test data locations are documented in CLAUDE.md.
