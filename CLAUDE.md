# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataFlow-CV is a computer vision dataset processing library for format conversion, visualization, and evaluation between YOLO, LabelMe, and COCO annotation formats. It provides both a Python API and a command-line interface (CLI) with `convert`, `visualize`, and `evaluate` subcommands.

The project follows a modular architecture with clear separation between format handlers, converters, visualizers, evaluators, and utilities. Each handler stores coordinates in its format's native representation — see `DatasetAnnotations.format` to determine coordinate semantics.

## Specifications

The `specs/` directory contains the **canonical specifications** — the single source of truth for SDD Agent development. It is organized into three layers:

```
specs/
├── evaluate/                       # WHAT — evaluation metric contracts
│   ├── index.md                    # Evaluate layer overview + conventions
│   ├── spec_evaluate_fundamentals.md  # IoU, matching rules, TP/FP/FN
│   ├── spec_evaluate_metrics.md       # P/R/F1, AP/mAP/AR, scale stratification
│   └── spec_evaluate_tasks.md         # Detection vs segmentation, COCO 12 metrics
│
├── formats/                        # WHAT — external data format contracts
│   ├── index.md                    # Formats layer overview
│   ├── spec_yolo_format.md         # YOLO .txt format authority
│   ├── spec_labelme_format.md      # LabelMe .json format authority
│   ├── spec_coco_format.md         # COCO .json format authority
│   └── spec_conversion.md          # Conversion rules (coordinate transforms, category mapping)
│
└── modules/                        # HOW — internal module architecture & interface contracts
    ├── index.md                    # Modules layer overview + dependency diagram
    ├── spec_label.md               # Label module (data models + handler interface)
    ├── spec_convert.md             # Convert module (pipeline, converters, RLE)
    ├── spec_visualize.md           # Visualize module (rendering pipeline, ColorManager, interaction)
    ├── spec_evaluate.md            # Evaluate module (evaluation pipeline, API, data models)
    └── spec_cli.md                 # CLI module (command signatures, exit codes, decorators)
```

### Architecture Constraint (from specs/modules/index.md)

```
┌──────────────────────────────────────────────────────────────┐
│                           CLI                                 │
│  (calls Convert, Visualize & Evaluate public APIs)            │
└──────┬─────────────────────┬──────────────────┬──────────────┘
       │                     │                  │
       ▼                     ▼                  ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│   Convert    │    │    Visualize     │    │   Evaluate   │
│  (pipeline)  │    │  (rendering)     │    │  (metrics)   │
└──────┬───────┘    └───────┬──────────┘    └──────┬───────┘
       │                    │                      │
       │    ZERO CROSS-     │    ZERO CROSS-       │
       │    DEPENDENCY      │    DEPENDENCY        │
       │                    │                      │
       ▼                    ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│                         Label                                 │
│  Data Models + Handlers (read/write/validate)                 │
└──────────────────────────────────────────────────────────────┘
```

**Hard constraints:**
1. **Convert ↔ Visualize**: Zero dependency. They do not import from each other.
2. **Evaluate ↔ Convert**: Zero dependency. They do not import from each other.
3. **Evaluate ↔ Visualize**: Zero dependency. They do not import from each other.
4. **Convert → Label**: Converters import handlers and models only through public interfaces.
5. **Visualize → Label**: Visualizers import handlers and models only through public interfaces.
6. **Evaluate → Label**: Evaluators import COCO handler and models only through public interfaces.
7. **CLI → Convert/Visualize/Evaluate**: CLI commands only call converter/visualizer/evaluator public APIs. CLI must NOT import label handlers or pycocotools directly.

### Specs vs CLAUDE.md

- **Specs** define "what is correct" — the behavioral contract. They are **living documents**: when specs are insufficient or unreasonable, update them first before touching code.
- **CLAUDE.md** describes "how the code works" — architecture, patterns, known gotchas. Evolves with the codebase.
- For SDD Agent development, specs are the **compliance benchmark**; CLAUDE.md is the **development context**.
- The full SDD Agent development workflow is documented in [`specs/SDD_AGENT.md`](specs/SDD_AGENT.md).

## Git Commits

When creating git commits, use the following format:

```bash
git commit -m "$(cat <<'EOF'
<type>(<scope>): <subject>

<body if needed>

Co-Authored-By: DeepSeek-V4.0 <noreply@deepseek.com>
EOF
)"
```

The Co-Authored-By line is optional and can be omitted.

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

The AI model used in this project is DeepSeek-V4.0, not Claude Opus.

## Architecture

### Format Ordering Convention

When YOLO, LabelMe, and COCO appear together in any listing (docs, enums, imports, CLI help, tables, `__all__`, etc.), they **must** follow this order:

```
YOLO → LabelMe → COCO
```

**Rationale**: Progression from simplest (YOLO: plain `.txt`, per-image) to medium (LabelMe: `.json` per-image) to most complex (COCO: single `.json` for entire dataset). This reduces cognitive load for newcomers and provides a consistent reading experience across the entire codebase.

**Where this applies:**
- `AnnotationFormat` enum member order in `models.py`
- Docstrings and comments listing all three formats
- Import order and `__all__` ordering in `__init__.py` files
- CLI subcommand registration order (controls `--help` output)
- Spec documents (both Formats and Modules layers)
- README and other user-facing documentation

**Where this does NOT apply:**
- Format-specific files where self-reference is natural (e.g., `spec_coco_format.md` comparison tables start with COCO)
- Conversion direction descriptions that follow an actual pipeline sequence

### Data Flow Pipeline
```
Source Format (YOLO/LabelMe/COCO) → Handler.read() → DatasetAnnotations → Converter → Target Handler.write() → Target Format
```

### Core Data Model (`dataflow/label/models.py`)

- `DatasetAnnotations`: Top-level container (images list, categories dict, `format` field, dataset_info). `format` defines coordinate semantics.
- `ImageAnnotation`: Per-image data (width, height, objects list, image_path, image_id)
- `ObjectAnnotation`: Single annotation (class_id, class_name, optional `BoundingBox` and `Segmentation`, is_crowd flag)
- `BoundingBox`: Coordinates in **format-native** representation. See `DatasetAnnotations.format`:
  - `YOLO`: `(x, y)` = center, `(width, height)` = normalized [0,1]
  - `LABELME`: `(x, y)` = top-left, `(width, height)` = absolute pixels
  - `COCO`: `(x, y)` = top-left, `(width, height)` = absolute pixels
- `Segmentation`: Polygon points in native coordinates (normalized for YOLO, absolute pixels for LabelMe/COCO)

### Annotation Handlers (`dataflow/label/`)

- `BaseAnnotationHandler`: Abstract base with `read()`, `write()`, `validate()` methods. Takes `strict_mode` (default True) and `verbose` (default False) kwargs.
- `YoloHandler(label_dir, class_file, image_dir, **kwargs)`: Reads/writes `.txt` files (one per image). Supports detection (`class_id x y w h`) and segmentation (`class_id x1 y1 x2 y2 ...`).
- `LabelMeHandler(label_dir, class_file=None, **kwargs)`: Reads/writes per-image `.json` files.
- `CocoHandler(annotation_file, do_rle=False, **kwargs)`: Reads/writes a single COCO JSON file. Supports polygon and RLE segmentation.

### Converters (`dataflow/convert/`)

All converters follow the pattern: **validate → read source → convert annotations → write target**. They support `strict_mode` and `verbose` kwargs.

- `YoloAndCocoConverter(source_to_target)`: YOLO ↔ COCO. Supports `do_rle` for RLE encoding in COCO output.
- `LabelMeAndYoloConverter(source_to_target)`: LabelMe ↔ YOLO. Copies images between directories.
- `CocoAndLabelMeConverter(source_to_target)`: COCO ↔ LabelMe.

Converters set `self._source_annotations_for_target` before write, which is used by `create_target_handler` for image copying (LabelMe→YOLO). Must be cleaned up via try/finally to prevent state leakage if write raises.

### RLE Conversion (`dataflow/convert/rle_converter.py`)

Handles polygon-to-RLE and RLE-to-polygon conversion using pycocotools. **Critical**: RLE `counts` bytes must use `latin1` (not UTF-8) encoding for JSON serialization — latin1 provides lossless 1:1 byte-to-character mapping while UTF-8 crashes on arbitrary binary RLE data.

### Visualizers (`dataflow/visualize/`)

- `YOLOVisualizer`: YOLO annotation visualization
- `LabelMeVisualizer`: LabelMe annotation visualization
- `CocoVisualizer`: COCO annotation visualization

All extend `BaseVisualizer` which provides `ColorManager` (HSV-based palette, max 1000 colors), image loading/drawing, progress bars, and both display (`is_show`) and save (`is_save`) modes.

**Display behavior**: Uses a single persistent OpenCV window (created once, reused across images) with fixed position. The window auto-sizes to match each image's dimensions. Keyboard controls: `Enter`/`Space` to advance to next image, `q`/`ESC` to exit early, any other key to continue.

### Evaluators (`dataflow/evaluate/`)

Evaluation of object detection and instance segmentation results, wrapping pycocotools' `COCOeval`.

- `DetectionEvaluator(verbose, strict_mode, logger)`: Detection evaluation using bbox IoU (`iouType='bbox'`). Input: GT COCO JSON + DT COCO JSON (with `score` field).
- `SegmentationEvaluator(verbose, strict_mode, logger)`: Segmentation evaluation using mask IoU (`iouType='segm'`). Input: GT COCO JSON + DT COCO JSON (with `segmentation` and `score` fields).
- `compute_pr_f1(gt, dt, iou_threshold, confidence_threshold, iou_type)`: Single-threshold P/R/F1 using manual greedy matching. Independent of full COCOeval pipeline for speed.
- `EvaluationResult`: Structured container with 12 COCO metrics, per-class breakdown (verbose mode), and stats.

**Key data models**: `EvaluationMetrics` (12 standard metrics), `PerClassMetrics` (per-category TP/FP/FN/AP/P/R/F1), `PRF1Result` (single-threshold P/R/F1).

**Verbose mode**: When `verbose=True`, computes per-class metrics and outputs a per-class breakdown table. Uses `VerboseLoggingOperations` for file logging (same pattern as Convert and Visualize).

**Input formats**: Accepts `str/Path` (COCO JSON file path), `dict` (COCO dict), or `DatasetAnnotations` (Label module). All normalized to `pycocotools.COCO` objects internally.

**pycocotools dependency**: pycocotools is a required runtime dependency for evaluation. Guard with `try/except ImportError` and raise a clear error message if not installed.

### Prediction File Conversion

YOLO prediction files (model output) differ from YOLO label files (ground truth):

| Format | Detection | Segmentation | Token Count |
|--------|-----------|-------------|-------------|
| **Label** (GT) | `class_id cx cy w h` | `class_id x1 y1 ... xn yn` | Odd (>0) |
| **Prediction** (DT) | `class_id cx cy w h confidence` | `class_id x1 y1 ... xn yn confidence` | Even (>0) |

Use `yolo2coco --prediction` to convert prediction files. The confidence value is preserved as the COCO `score` field. For segmentation predictions, polygon format is used by default (pycocotools handles polygon→RLE internally during evaluation).

### CLI Structure (`dataflow/cli/`)

- `main.py`: Entry point `cli` group with global `--version`/`-v` flag
- `commands/convert.py`: 6 subcommands — `yolo2coco`, `yolo2labelme`, `labelme2yolo`, `labelme2coco`, `coco2yolo`, `coco2labelme`. `yolo2coco` supports `--prediction` for model output.
- `commands/visualize.py`: 3 subcommands — `yolo`, `labelme`, `coco`
- `commands/evaluate.py`: 2 subcommands — `detection`, `segmentation`
- `commands/utils.py`: Shared decorators (`add_common_options`, `add_visualize_options`), validators, and `FormattedCommand` (custom Click Command with aligned argument display in --help)
- `commands/exceptions.py`: Exception hierarchy with distinct exit codes:
  - `ParameterError` (exit 1), `InputError` (exit 2), `OutputError` (exit 3), `RuntimeCLIError` (exit 4), `SystemError` (exit 5)
  - All extend `click.ClickException` for clean CLI error display

Common CLI options: `--verbose` (enable file logging), `--no-strict` (disable strict mode for convert), `--display/--no-display` (control visualization window for visualize).

### Utilities (`dataflow/util/`)

- `logging_util.py`: `LoggingOperations` (console logging) and `VerboseLoggingOperations` (console + file logging). `get_verbose_logger()` returns `(logger, log_file_path)` tuple.
- `file_util.py`: `FileOperations` for file I/O (read/write lines, copy, glob). `read_lines()` uses `rstrip()` to preserve leading whitespace.

## Critical Implementation Details

### Coordinate Systems

**Native format storage:** Each handler stores coordinates in its format's native representation. There is no unified normalized internal model.

| Format | Bbox origin | Coordinate space | Validated by |
|--------|------------|-----------------|-------------|
| YOLO | Center | Normalized (0-1) | `_validate_normalized_coordinate()` |
| LabelMe | Top-left | Absolute pixels | `_validate_absolute_coordinate()` |
| COCO | Top-left | Absolute pixels | `_validate_absolute_coordinate()` |

**Coordinate transforms** happen exclusively in converters (`dataflow/convert/`), not in handlers or visualizers. Transform logic is in each converter's `convert_annotations()` method.

### RLE Serialization

pycocotools `mask.encode()` returns binary `counts` bytes. For JSON serialization:
- **Write path** (`coco_handler.py`, `rle_converter.py`): `counts_bytes.decode("latin1")` → string for JSON
- **Read path** (`coco_handler.py`): `counts_str.encode("latin1")` → bytes for `mask.decode()`

Never use UTF-8 for RLE counts — it cannot represent all 256 byte values and will cause `UnicodeDecodeError` crashes.

### Visualizer Rendering Pipeline

Visualizers convert all annotations to `RenderAnnotation` (absolute pixel integers) during `load_annotations()`. Drawing methods receive pre-computed absolute pixel coordinates — no coordinate math happens in the draw path.

### Validation Behavior

- **Strict mode** (default): Validation errors immediately raise exceptions / return error results.
- **Non-strict mode**: Errors are collected as warnings; processing continues where possible. CLI now supports `--no-strict`.
- **Image errors**: Missing/unreadable images are always treated as warnings regardless of strict mode.
- **Coordinate validation**: Format-aware — YOLO coords checked in [0,1], LabelMe/COCO coords checked finite and non-negative.

## Development Commands

### Installation
```bash
pip install -e .                    # Editable install (recommended for dev)
pip install -e .[dev]               # With test/lint deps
pip install -e .[coco]              # With pycocotools for RLE support
```

With editable install, use `python -m dataflow.cli` instead of `dataflow-cv`.

### Testing
```bash
pytest                              # All tests
pytest -x -q                        # Stop on first failure, quiet
pytest tests/label/test_yolo.py     # Single module
pytest tests/label/test_yolo.py::TestYoloAnnotationHandler::test_read_detection  # Single test
pytest --cov=dataflow --cov-report=html  # Coverage report
pytest -n auto                      # Parallel (requires pytest-xdist)
```

### Linting
```bash
black dataflow tests samples        # Format
isort dataflow tests samples        # Imports
flake8 dataflow tests samples       # Lint
mypy dataflow                       # Type check
```

### Manual CLI Verification
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

## Known Gotchas

1. **COCO↔YOLO precision loss**: Cross-format conversion between normalized (YOLO) and absolute pixel (LabelMe/COCO) is inherently lossy (±1 px). The converter's `convert_annotations()` performs the transform explicitly — see `spec_conversion.md` for details.
2. **RLE encoding**: Always `latin1`, never `utf-8`, for byte↔string round-trips.
3. **LabelMe imageData**: Not required on read; valid external files may omit it.
4. **Converter state**: `_source_annotations_for_target` must be cleared in a `finally` block to prevent stale state on exceptions.
5. **COCO image_id fallback**: When `img.image_id` is not a digit string, use a dedicated image counter (not the annotation counter).
6. **Progress bar at 100%**: Guard against `filled == width` causing `"." * -1`.
7. **Visualization keyboard control**: The OpenCV window captures keyboard input. Press `Enter`/`Space` to advance, `q`/`ESC` to exit. The window manager close button (X) may not work reliably — always use keyboard to close.
8. **Single persistent window**: The visualizer creates one window (named by format) and reuses it for all images. Window auto-sizes to each image's dimensions. Fixed window position prevents flickering across images.
9. **BoundingBox semantics depend on format**: The same `BoundingBox` class is used for all formats. Always check `DatasetAnnotations.format` to interpret `(x, y, width, height)` correctly.
10. **DT predictions require `score`**: COCO prediction JSON must include `"score"` field in every annotation. The evaluate module validates this — missing scores cause errors in strict mode.
11. **DT predictions require `area`**: pycocotools `COCOeval` requires `area` on both GT and DT annotations. When generating DT JSON, ensure area is populated (typically `bbox.width * bbox.height`).
12. **Segmentation evaluation with mask IoU**: pycocotools automatically converts polygon → RLE internally during `COCOeval.evaluate()`. Pre-converting to RLE in pred.json is unnecessary — polygon format is recommended.
13. **YOLO prediction format**: YOLO prediction files use 6 tokens (detection) or even tokens (segmentation) vs 5/odd for labels. Use `--prediction` flag with `yolo2coco` to convert model outputs. Mixed label/prediction files are not supported — the flag applies dataset-wide.

## Bug Report

A comprehensive code review identified bugs across the codebase, documented in `~/.claude/plans/bug-p0-p1-p2-glowing-hellman.md`. Most have been fixed; remaining issues are tracked in the plan file. Refer to the plan file when working on related code areas.

## Test Structure

```
tests/
├── label/          # Handler unit tests (read/write/validate per format)
├── convert/        # Converter unit tests + integration tests
├── visualize/      # Visualizer unit tests
├── evaluate/       # Evaluator unit tests + metric computation tests
├── util/           # Utility unit tests
└── cli/            # CLI integration tests (convert, visualize, evaluate)
```

Test data lives in `assets/test_data/`, organized by format (det/seg) and annotation type. Evaluate test data lives under `assets/test_data/evaluate/`.
