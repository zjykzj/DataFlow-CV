---
name: dev
description: Run development commands — install, test, lint, typecheck, and manual CLI verification. Use when the user asks to run tests, lint, type check, or verify the CLI.
allowed-tools: Bash
---

# Development Commands

## Installation

```bash
pip install -e .                    # Editable install (recommended for dev)
pip install -e .[dev]               # With test/lint deps
pip install -e .[coco]              # With pycocotools for RLE support
```

With editable install, use `python -m dataflow.cli` instead of `dataflow-cv`.

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

## Manual CLI Verification

```bash
# Test conversion (label)
dataflow-cv convert yolo2coco --verbose images/ yolo_labels/ classes.txt /tmp/out.json

# Test conversion (prediction)
dataflow-cv convert yolo2coco --prediction --verbose images/ yolo_labels/ classes.txt /tmp/pred.json

# Test visualization (non-display)
dataflow-cv visualize yolo --no-display --verbose images/ yolo_labels/ classes.txt --save /tmp/viz/

# Test evaluation
dataflow-cv evaluate detection --verbose --prf1 assets/test_data/evaluate/gt_coco.json assets/test_data/evaluate/dt_coco.json
```

## Test Structure

```
tests/
├── conftest.py      # Shared fixtures (project_root, test_data_dir, etc.)
├── label/           # Handler unit tests (read/write/validate per format)
├── convert/         # Converter unit tests + integration tests
├── visualize/       # Visualizer unit tests
├── evaluate/        # Evaluator unit tests + metric computation tests
├── util/            # Utility unit tests (LogManager, format helpers)
└── cli/             # CLI tests (convert, visualize, evaluate — 440 tests total)
```

Test data lives in `assets/test_data/`, organized by format (det/seg) and annotation type. Evaluate test data lives under `assets/test_data/evaluate/`.
