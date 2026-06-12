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
- For SDD development, specs are the **compliance benchmark**; CLAUDE.md is the **development context**.
- The SDD methodology is documented in [`specs/SDD_METHODOLOGY.md`](specs/SDD_METHODOLOGY.md) (general, reusable) and [`specs/SDD_GUIDE.md`](specs/SDD_GUIDE.md) (DataFlow-CV specific).

## Version Bump Checklist

When bumping the project version, **all three** locations must be updated:

| # | File | Field |
|---|------|-------|
| 1 | `pyproject.toml` | `version = "X.Y.Z"` |
| 2 | `dataflow/__init__.py` | `__version__ = "X.Y.Z"` |
| 3 | `CHANGELOG.md` | `## [X.Y.Z] - YYYY-MM-DD` section header |

Verify with: `grep -rn '"X\.Y\.Z"' dataflow/ pyproject.toml` (exclude `CHANGELOG.md` which retains old entries).

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

The Co-Authored-By line is **mandatory** for all commits. The only exception is when the AI model actually powering this session is NOT DeepSeek-V4.0 — in that case, the engineer must confirm the correct model and update the Co-Authored-By line accordingly.

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

**Batch path (used by Convert→COCO, Evaluate):**
```
Source Format → Handler.read() → DatasetAnnotations → Converter.convert_annotations() → Target Handler.write() → Target Format
```

**Streaming path (used by Visualize, Convert→YOLO/LabelMe):**
```
Source Format → Handler.iter_images() → ImageAnnotation → _convert_single_image() → Target Handler.write_one() → Target Format
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

- `BaseAnnotationHandler`: Abstract base with `read()`, `write()`, `write_one()`, `validate()`, `iter_images()` methods. Takes `strict_mode` (default True) and `verbose` (default False) kwargs.
  - `read()`: Batch load — returns `DatasetAnnotations` (all data in memory). Used by Convert (batch path), Evaluate.
  - `iter_images()`: Streaming iterator — yields `ImageAnnotation` one at a time. Used by Visualize, Convert (streaming path). Lower memory, faster first-image latency.
  - `write()`: Batch write — writes entire `DatasetAnnotations` to target format (e.g., COCO JSON).
  - `write_one()`: Single-image write — writes one `ImageAnnotation` to target format. Used by Convert streaming path for per-file output (YOLO `.txt`, LabelMe `.json`). COCO handler raises `NotImplementedError` (always batch).
- `YoloHandler(label_dir, class_file, image_dir, **kwargs)`: Reads/writes `.txt` files (one per image). Supports detection (`class_id x y w h`) and segmentation (`class_id x1 y1 x2 y2 ...`).
- `LabelMeHandler(label_dir, class_file=None, **kwargs)`: Reads/writes per-image `.json` files.
- `CocoHandler(annotation_file, do_rle=False, prediction=False, **kwargs)`: Reads/writes a single COCO JSON file. Supports polygon and RLE segmentation. `write_one()` raises `NotImplementedError` — COCO is always batch output. When `prediction=True`, `write()` outputs a plain JSON list of annotation dicts (prediction format, Variant B per `spec_coco_format.md` §10.1) instead of a full COCO dict.

### Converters (`dataflow/convert/`)

**Two pipelines**, auto-selected by `BaseConverter.convert()` based on target format:

| Pipeline | Method | When Used |
|----------|--------|-----------|
| **Batch** | `_batch_convert()` → `handler.read()` → `convert_annotations()` → `handler.write()` | COCO target (single JSON) |
| **Streaming** | `stream_convert()` → `handler.iter_images()` → `_convert_single_image()` → `handler.write_one()` | YOLO/LabelMe target (per-file) |

**Template method hierarchy:**
- `convert()`: Concrete in `BaseConverter` — auto-dispatches: COCO target → `_batch_convert()`, else → `stream_convert()`. Subclasses should NOT override.
- `_batch_convert()`: Concrete template method in `BaseConverter` — orchestrates read → `convert_annotations()` → write with try/finally state cleanup. Subclasses should NOT override.
- `_convert_single_image(image_ann, **kwargs)`: Abstract — per-image coordinate transform (single source of truth). Uses shared utilities from `convert/utils.py`.
- `convert_annotations(source_annotations, kwargs)`: Abstract — batch coordinate transform. Base implementation raises `NotImplementedError` (no safe pass-through). Canonical delegation: calls `_convert_single_image()` per image.
- `stream_convert(source_path, target_path, **kwargs)`: Concrete template method in `BaseConverter` — streaming pipeline. Sets `self._source_path` for handlers that need it during `create_target_handler()`.
- `_ensure_categories_for_streaming()`: Pre-loads categories before streaming iteration. Base implementation checks `handler.categories`; COCO subclasses use `read_coco_categories()` from utils.
- `_post_batch_convert(result, source_handler, kwargs)`: Optional post-processing hook (default no-op). Overridden to add RLE accuracy warnings when `do_rle=True` AND segmentation data exists.

**Shared coordinate transforms** (`convert/utils.py`):

| Function | Direction | Used By |
|----------|-----------|---------|
| `yolo_to_absolute_pixel(bbox, seg, w, h)` | YOLO normalized center → absolute px top-left | `YoloAndCocoConverter` (YOLO→COCO), `LabelMeAndYoloConverter` (YOLO→LabelMe) |
| `absolute_pixel_to_yolo(bbox, seg, w, h)` | Absolute px top-left → YOLO normalized center | `YoloAndCocoConverter` (COCO→YOLO), `LabelMeAndYoloConverter` (LabelMe→YOLO) |
| `read_coco_categories(json_path)` | Read COCO categories without full dataset load | `YoloAndCocoConverter`, `CocoAndLabelMeConverter` (_ensure_categories_for_streaming) |

These are pure functions — stateless, no handler interaction. They replaced ~80 lines of duplicated coordinate math previously spread across two converter files.

- `YoloAndCocoConverter(source_to_target, prediction=False)`: YOLO ↔ COCO. `prediction=True` for model output. Supports `do_rle`.
  - YOLO→COCO: batch (COCO single JSON)
  - COCO→YOLO: streaming (per-file .txt)
- `LabelMeAndYoloConverter(source_to_target)`: LabelMe ↔ YOLO. Copies images between directories.
  - Both directions: streaming (per-file output)
- `CocoAndLabelMeConverter(source_to_target)`: COCO ↔ LabelMe.
  - COCO→LabelMe: streaming (per-file .json)
  - LabelMe→COCO: batch (COCO single JSON)

**State management**: `_source_annotations_for_target` stores categories for target handler creation. Must be cleared via `try/finally` in batch path. Streaming path avoids stale state by extracting categories upfront via `_ensure_categories_for_streaming()`.

### RLE Conversion (`dataflow/convert/rle_converter.py`)

Handles polygon-to-RLE and RLE-to-polygon conversion using pycocotools. **Critical**: RLE `counts` bytes must use `latin1` (not UTF-8) encoding for JSON serialization — latin1 provides lossless 1:1 byte-to-character mapping while UTF-8 crashes on arbitrary binary RLE data.

### Visualizers (`dataflow/visualize/`)

- `YOLOVisualizer(label_dir, image_dir, class_file, **kwargs)`: YOLO annotation visualization
- `LabelMeVisualizer(label_dir, image_dir, class_file=None, **kwargs)`: LabelMe annotation visualization
- `CocoVisualizer(annotation_file, image_dir, **kwargs)`: COCO annotation visualization

All extend `BaseVisualizer` which provides `ColorManager` (HSV-based palette, max 1000 colors), image loading/drawing, counter-based progress, and both display (`is_show`) and save (`is_save`) modes.

**Streaming pipeline** (no batch accumulation):
```
visualize()
├── handler = _create_handler()
├── for image_ann in handler.iter_images():
│   ├── render_data = _convert_to_render_data(image_ann)  ← per-image coordinate conversion
│   └── _visualize_single_image(image_path, render_data)   ← display or save
```

**Template method hierarchy:**
- `_create_handler()`: Abstract — creates format-specific Label handler (lazy, not in `__init__`)
- `_convert_to_render_data(image_ann)`: Abstract — converts single ImageAnnotation (format-native coords) to RenderData (absolute pixel coords)
- `_visualize_single_image(image_path, render_data)`: Concrete — load image, draw annotations, display/save

**Display behavior**: Uses a single persistent OpenCV window (created once, reused across images) with fixed position. The window auto-sizes to match each image's dimensions. Keyboard controls: `Enter`/`Space` to advance to next image, `q`/`ESC` to exit early, any other key to continue.

### Evaluators (`dataflow/evaluate/`)

Evaluation of object detection and instance segmentation results, wrapping pycocotools' `COCOeval`.

- `DetectionEvaluator(verbose, strict_mode, logger)`: Detection evaluation using bbox IoU (`iouType='bbox'`). Input: GT COCO JSON + DT COCO JSON (with `score` field).
- `SegmentationEvaluator(verbose, strict_mode, logger)`: Segmentation evaluation using mask IoU (`iouType='segm'`). Input: GT COCO JSON + DT COCO JSON (with `segmentation` and `score` fields).
- `compute_pr_f1(gt, dt, iou_threshold, confidence_threshold, iou_type, method)`: Single-threshold P/R/F1 using manual greedy matching. Supports macro (default) and micro averaging for overall P/R/F1. Supports both bbox IoU and mask IoU (via pycocotools `mask` module). Independent of full COCOeval pipeline for speed.
- `EvaluationResult`: Structured container with 12 COCO metrics, per-class breakdown (verbose mode), and stats.

**Key data models**: `EvaluationMetrics` (12 standard metrics), `PerClassMetrics` (per-category TP/FP/FN/AP/P/R/F1), `PRF1Result` (single-threshold P/R/F1).

**Verbose mode**: When `verbose=True`, computes per-class metrics and outputs a per-class breakdown table. Uses `VerboseLoggingOperations` for file logging (same pattern as Convert and Visualize).

**DT input formats**: Accepts `str/Path` (COCO JSON file — full dict or plain list), `list` (in-memory annotation dicts), `dict` (COCO dict), or `DatasetAnnotations` (Label module). List-format DT is loaded via `coco_gt.loadRes()`, which copies images and categories from GT. GT input always requires a full COCO dict.

**pycocotools dependency**: pycocotools is a required runtime dependency for evaluation. Guard with `try/except ImportError` and raise a clear error message if not installed.

### Prediction File Conversion

YOLO prediction files (model output) differ from YOLO label files (ground truth):

| Format | Detection | Segmentation | Token Count |
|--------|-----------|-------------|-------------|
| **Label** (GT) | `class_id cx cy w h` | `class_id x1 y1 ... xn yn` | Odd (>0) |
| **Prediction** (DT) | `class_id cx cy w h confidence` | `class_id x1 y1 ... xn yn confidence` | Even (>0) |

Use `yolo2coco --prediction` to convert prediction files. **Output format**: plain JSON list of annotation dicts (Variant B per `spec_coco_format.md` §10.1) — each entry contains `image_id`, `category_id`, `bbox`, `area`, `score`, and optionally `segmentation`. The list format matches standard model inference output (Detectron2, MMDetection) and is loaded by pycocotools' `loadRes()`. For segmentation predictions, polygon format is used by default (pycocotools handles polygon→RLE internally during evaluation).

COCO prediction files exist in **two variants** (see `spec_coco_format.md` §10):
- **Variant A**: Full COCO dict (`images`, `annotations`, `categories`) — produced by `yolo2coco` (label/annotation mode)
- **Variant B**: Plain JSON list of annotation dicts — produced by `yolo2coco --prediction` (prediction mode), also the most common output from model inference frameworks

### CLI Structure (`dataflow/cli/`)

- `main.py`: Entry point `cli` group with global `--version`/`-v` flag
- `commands/convert.py`: 6 subcommands — `yolo2coco`, `yolo2labelme`, `labelme2yolo`, `labelme2coco`, `coco2yolo`, `coco2labelme`. `yolo2coco` supports `--prediction` for model output.
- `commands/visualize.py`: 3 subcommands — `yolo`, `labelme`, `coco`
- `commands/evaluate.py`: 2 subcommands — `detection`, `segmentation`. Both support `--prf1` (P/R/F1), `--prf1-iou`, `--prf1-conf`, and `--prf1-method` (macro|micro) options.
- `commands/utils.py`: Shared decorators (`add_common_options`, `add_visualize_options`), validators, and `FormattedCommand` (custom Click Command with aligned argument display in --help)
- `commands/exceptions.py`: Exception hierarchy with distinct exit codes:
  - `ParameterError` (exit 1), `InputError` (exit 2), `OutputError` (exit 3), `RuntimeCLIError` (exit 4), `SystemError` (exit 5)
  - All extend `click.ClickException` for clean CLI error display

Common CLI options: `--verbose` (enable file logging), `--no-strict` (disable strict mode for convert), `--display/--no-display` (control visualization window for visualize).

### Utilities (`dataflow/util/`)

- `logging_util.py`: `LoggingOperations` (console logging) and `VerboseLoggingOperations` (console + file logging). `get_verbose_logger()` returns `(logger, log_file_path)` tuple. Also provides `logging_error_or_raise()` (shared error-handling utility used by all base classes) and `detect_image_error()` (identifies image-related errors that should always be warnings).
- `file_util.py`: `FileOperations` for file I/O (read/write lines, copy, glob). `read_lines()` uses `rstrip()` to preserve leading whitespace.

## Critical Implementation Details

### Coordinate Systems

**Native format storage:** Each handler stores coordinates in its format's native representation. There is no unified normalized internal model.

| Format | Bbox origin | Coordinate space | Validated by |
|--------|------------|-----------------|-------------|
| YOLO | Center | Normalized (0-1) | `_validate_normalized_coordinate()` |
| LabelMe | Top-left | Absolute pixels | `_validate_absolute_coordinate()` |
| COCO | Top-left | Absolute pixels | `_validate_absolute_coordinate()` |

**Coordinate transforms** happen exclusively in converters (`dataflow/convert/`), not in handlers or visualizers. The single source of truth is `_convert_single_image()`, which delegates to shared utilities in `convert/utils.py` (`yolo_to_absolute_pixel()` / `absolute_pixel_to_yolo()`). The batch `convert_annotations()` delegates to `_convert_single_image()` in a loop. Visualizers do per-image coordinate conversion to `RenderData` in `_convert_to_render_data()` (format-native → absolute pixels for OpenCV drawing).

### RLE Serialization

pycocotools `mask.encode()` returns binary `counts` bytes. For JSON serialization:
- **Write path** (`coco_handler.py`, `rle_converter.py`): `counts_bytes.decode("latin1")` → string for JSON
- **Read path** (`coco_handler.py`): `counts_str.encode("latin1")` → bytes for `mask.decode()`

Never use UTF-8 for RLE counts — it cannot represent all 256 byte values and will cause `UnicodeDecodeError` crashes.

### Visualizer Rendering Pipeline

Visualizers convert annotations per-image to `RenderAnnotation` (absolute pixel integers) during `_convert_to_render_data()`. Drawing methods receive pre-computed absolute pixel coordinates — no coordinate math happens in the draw path. The first image appears as soon as the first annotation file is parsed (streaming).

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
6. **Progress bar at 100%**: Guard against `filled == width` causing `"." * -1`. Note: Visualize module now uses counter-based progress (streaming), so this only applies to batch-mode progress bars in utility code.
7. **Visualization keyboard control**: The OpenCV window captures keyboard input. Press `Enter`/`Space` to advance, `q`/`ESC` to exit. The window manager close button (X) may not work reliably — always use keyboard to close.
8. **Single persistent window**: The visualizer creates one window (named by format) and reuses it for all images. Window auto-sizes to each image's dimensions. Fixed window position prevents flickering across images.
9. **BoundingBox semantics depend on format**: The same `BoundingBox` class is used for all formats. Always check `DatasetAnnotations.format` to interpret `(x, y, width, height)` correctly.
10. **DT predictions require `score`**: COCO prediction JSON must include `"score"` field in every annotation. The evaluate module validates this — missing scores cause errors in strict mode.
11. **DT predictions require `area`**: pycocotools `COCOeval` requires `area` on both GT and DT annotations. When generating DT JSON, ensure area is populated (typically `bbox.width * bbox.height`).
12. **Segmentation evaluation with mask IoU**: pycocotools automatically converts polygon → RLE internally during `COCOeval.evaluate()`. Pre-converting to RLE in pred.json is unnecessary — polygon format is recommended.
13. **YOLO prediction format**: YOLO prediction files use 6 tokens (detection) or even tokens (segmentation) vs 5/odd for labels. Use `--prediction` flag with `yolo2coco` to convert model outputs. Mixed label/prediction files are not supported — the flag applies dataset-wide.
14. **`convert_annotations()` must be overridden**: The base class raises `NotImplementedError` — there is no safe pass-through default. Every concrete converter must explicitly implement coordinate transformation.
15. **Shared coordinate transforms are in `convert/utils.py`**: `yolo_to_absolute_pixel()` and `absolute_pixel_to_yolo()` are the canonical implementations. New YOLO-involving converters must use these, not reimplement the math.
16. **`write_one()` is part of the handler contract**: `BaseAnnotationHandler` declares it as abstract. COCO handler raises `NotImplementedError`; YOLO/LabelMe handlers write per-file. New handlers must implement it.
17. **RLE warning is conditional**: Only added when `do_rle=True` AND the source dataset actually contains segmentation data (`handler.is_seg`). Detection-only datasets with `--do-rle` no longer produce misleading warnings.
18. **`_log_error` delegates to `logging_util.logging_error_or_raise()`**: All three base classes use the same shared utility. When modifying error behavior, update the utility, not individual base classes.
19. **COCO prediction file formats**: Prediction files exist in two variants. Variant A (full COCO dict) is the annotation-like format from `yolo2coco`. Variant B (plain JSON list) is the standard model-inference format from `yolo2coco --prediction`, Detectron2, MMDetection, etc. The Evaluate module accepts both via `_load_dt()`; the Convert module produces Variant A for labels and Variant B for predictions. See `spec_coco_format.md` §10.
20. **`CocoAnnotationHandler.prediction` parameter**: When `True`, `write()` outputs list format (Variant B). When `False` (default), outputs full COCO dict (Variant A). The parameter is propagated from `YoloAndCocoConverter(prediction=True)` → `create_target_handler()` → `CocoAnnotationHandler`.
21. **Prediction `score` field in list format**: In list-format prediction output, `score` is always included (not gated by `confidence < 1.0`). No `id` field is written — pycocotools `loadRes()` auto-assigns IDs.
22. **Prediction conversion is `yolo2coco` only**: Only `yolo2coco` supports `--prediction`. `labelme2coco` has no prediction mode — LabelMe format has no structural label vs prediction distinction (unlike YOLO's 5/odd token vs 6/even token difference), so there is no alternative prediction source format to convert from. When preparing evaluation data, if your predictions are not in YOLO format, they are likely already in COCO-compatible list format (e.g., Detectron2, MMDetection output).
23. **P/R/F1 macro vs micro averaging**: `compute_pr_f1(method="macro")` (default) computes overall P/R as the mean of per-class values — each category has equal weight. `method="micro"` computes overall P/R from summed TP/FP/FN across all categories — categories with more annotations have more weight. Per-class results are identical in both modes. `result.method` records which was used. `result.overall.tp/fp/fn` are always the summed totals.
24. **Mask IoU for segmentation PRF1**: `compute_pr_f1(iou_type='segm')` uses pycocotools `mask` module (`mask.frPyObjects()` + `mask.merge()` for polygon→RLE, `mask.iou()` for batched IoU computation). Both polygon and RLE input formats are supported. Image dimensions are fetched from `coco_gt.loadImgs()` for polygon→RLE conversion. Crowd annotations are handled via `mask.iou()`'s built-in `iscrowd` parameter.

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
