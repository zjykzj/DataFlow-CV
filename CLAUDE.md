# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DataFlow-CV is a computer vision dataset processing library for format conversion and visualization between LabelMe, COCO, and YOLO annotation formats. It provides both a Python API and a command-line interface (CLI) with `convert` and `visualize` subcommands.

The project follows a modular architecture with clear separation between format handlers, converters, visualizers, and utilities. All coordinates are normalized (0-1 range) in the internal data model.

## Specifications

The `specs/` directory contains the **canonical specifications** — the single source of truth for SSD Agent development. It is organized into two layers:

```
specs/
├── formats/                   # WHAT — external data format contracts
│   ├── index.md               # Formats layer overview
│   ├── spec_yolo_format.md    # YOLO .txt format authority
│   ├── spec_coco_format.md    # COCO .json format authority
│   ├── spec_labelme_format.md # LabelMe .json format authority
│   └── spec_conversion.md     # Conversion rules (coordinate transforms, category mapping)
│
└── modules/                   # HOW — internal module architecture & interface contracts
    ├── index.md               # Modules layer overview + dependency diagram
    ├── spec_label.md          # Label module (data models + handler interface)
    ├── spec_convert.md        # Convert module (pipeline, converters, RLE)
    ├── spec_visualize.md      # Visualize module (rendering pipeline, ColorManager, interaction)
    └── spec_cli.md            # CLI module (command signatures, exit codes, decorators)
```

### Architecture Constraint (from specs/modules/index.md)

```
┌──────────────────────────────────────────────┐
│                    CLI                        │
│  (calls Convert & Visualize public APIs)      │
└──────┬─────────────────────┬─────────────────┘
       │                     │
       ▼                     ▼
┌──────────────┐    ┌──────────────────┐
│   Convert    │    │    Visualize     │
│  (pipeline)  │    │  (rendering)     │
└──────┬───────┘    └───────┬──────────┘
       │                    │
       │    ZERO CROSS-     │
       │    DEPENDENCY      │
       │                    │
       ▼                    ▼
┌──────────────────────────────────────────────┐
│                   Label                       │
│  Data Models + Handlers (read/write/validate) │
└──────────────────────────────────────────────┘
```

**Hard constraints:**
1. **Convert ↔ Visualize**: Zero dependency. They do not import from each other.
2. **Convert → Label**: Converters import handlers and models only through public interfaces.
3. **Visualize → Label**: Visualizers import handlers and models only through public interfaces.
4. **CLI → Convert/Visualize**: CLI commands only call converter/visualizer public APIs. CLI must NOT import label handlers directly.

### Specs vs CLAUDE.md

- **Specs** define "what is correct" — the behavioral contract. Stable; change only when requirements change.
- **CLAUDE.md** describes "how the code works" — architecture, patterns, known gotchas. Evolves with the codebase.
- For SSD Agent development, specs are the **compliance benchmark**; CLAUDE.md is the **development context**.

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

### Data Flow Pipeline
```
Source Format (YOLO/LabelMe/COCO) → Handler.read() → DatasetAnnotations → Converter → Target Handler.write() → Target Format
```

### Core Data Model (`dataflow/label/models.py`)

- `DatasetAnnotations`: Top-level container (images list, categories dict, dataset_info)
- `ImageAnnotation`: Per-image data (width, height, objects list, image_path, image_id)
- `ObjectAnnotation`: Single annotation (class_id, class_name, optional `BoundingBox` and `Segmentation`, is_crowd flag)
- `BoundingBox`: **Center-based** normalized coordinates (x, y, width, height). Internally, x/y are the center of the box, not top-left.
- `Segmentation`: List of normalized (x, y) polygon points
- `OriginalData`: Stores raw annotation data keyed by `AnnotationFormat` enum for lossless A→B→A round-trips

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

### CLI Structure (`dataflow/cli/`)

- `main.py`: Entry point `cli` group with global `--version`/`-v` flag
- `commands/convert.py`: 6 subcommands — `yolo2coco`, `yolo2labelme`, `coco2yolo`, `coco2labelme`, `labelme2yolo`, `labelme2coco`
- `commands/visualize.py`: 3 subcommands — `yolo`, `labelme`, `coco`
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

**Internal model:** All coordinates are **0-1 normalized**. `BoundingBox.x`/`BoundingBox.y` are **center** coordinates (YOLO convention), NOT top-left.

**Key methods on BoundingBox:**
- `xyxy(img_w, img_h)` → `(x1, y1, x2, y2)` — top-left to bottom-right in absolute pixels. **Use this for COCO bbox conversion.**
- `xywh_abs(img_w, img_h)` → `(cx, cy, w, h)` — center-x, center-y in absolute pixels. Do NOT use for COCO output; COCO expects top-left.

**Format expectations:**
| Format | Bbox origin | Coordinate space |
|--------|------------|-----------------|
| YOLO | Center | Normalized (0-1) |
| COCO | Top-left | Absolute pixels |
| LabelMe | Varies (polygon/rectangle) | Absolute pixels |

### RLE Serialization

pycocotools `mask.encode()` returns binary `counts` bytes. For JSON serialization:
- **Write path** (`coco_handler.py`, `rle_converter.py`): `counts_bytes.decode("latin1")` → string for JSON
- **Read path** (`coco_handler.py`): `counts_str.encode("latin1")` → bytes for `mask.decode()`

Never use UTF-8 for RLE counts — it cannot represent all 256 byte values and will cause `UnicodeDecodeError` crashes.

### Original Data Preservation

`OriginalData` stores the exact raw bytes/strings from the source format, enabling lossless A→B→A round-trips:

- **YOLO**: Stores tokenized line items as `[class_id_str, x_float, y_float, ...]`. Items are stored with numeric types (float for coords) to prevent TypeError during coordinate extraction.
- **LabelMe**: Stores raw JSON dicts. `imageData` is NOT a required field on read (it's optional base64 image data), though it may appear in some files.
- **COCO**: Stores raw annotation dicts from the source JSON. When writing, the original data path preserves exact bbox/segmentation values (not recomputed from the internal model) to maintain lossless precision.

The `OriginalDataManager.extract_original_coordinates()` extracts bbox and segmentation points from original data by format.

### Validation Behavior

- **Strict mode** (default): Validation errors immediately raise exceptions / return error results.
- **Non-strict mode**: Errors are collected as warnings; processing continues where possible. CLI now supports `--no-strict`.
- **Image errors**: Missing/unreadable images are always treated as warnings regardless of strict mode.
- **Coordinate validation**: In non-strict mode, invalid coordinates cause the entire annotation line to be skipped (not just the failing coordinate check).

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
# Test conversion
dataflow-cv convert yolo2coco --verbose assets/test_data/ images/ yolo_labels/ classes.txt /tmp/out.json

# Test visualization (non-display)
dataflow-cv visualize yolo --no-display --verbose images/ yolo_labels/ classes.txt --save /tmp/viz/
```

## Known Gotchas

1. **COCO bbox**: Must convert from internal center-based coords to COCO top-left using `BoundingBox.xyxy()` → `[x1, y1, x2-x1, y2-y1]`. Using `xywh_abs()` directly produces offset bboxes.
2. **RLE encoding**: Always `latin1`, never `utf-8`, for byte↔string round-trips.
3. **YOLO OriginalData items**: First element is str (class_id), rest are float. Always `list()`-copy before mutating.
4. **LabelMe imageData**: Not required on read; valid external files may omit it.
5. **Converter state**: `_source_annotations_for_target` must be cleared in a `finally` block to prevent stale state on exceptions.
6. **COCO image_id fallback**: When `img.image_id` is not a digit string, use a dedicated image counter (not the annotation counter).
7. **Progress bar at 100%**: Guard against `filled == width` causing `"." * -1`.
8. **Visualization keyboard control**: The OpenCV window captures keyboard input. Press `Enter`/`Space` to advance, `q`/`ESC` to exit. The window manager close button (X) may not work reliably — always use keyboard to close.
9. **Single persistent window**: The visualizer creates one window (named by format) and reuses it for all images. Window auto-sizes to each image's dimensions. Fixed window position prevents flickering across images.

## Bug Report

A comprehensive code review identified bugs across the codebase, documented in `~/.claude/plans/bug-p0-p1-p2-glowing-hellman.md`. Most have been fixed; remaining issues are tracked in the plan file. Refer to the plan file when working on related code areas.

## Test Structure

```
tests/
├── label/          # Handler unit tests (includes lossless roundtrip tests)
├── convert/        # Converter unit tests + integration tests
├── visualize/      # Visualizer unit tests
├── util/           # Utility unit tests
└── cli/            # CLI integration tests
```

Test data lives in `assets/test_data/`, organized by format (det/seg) and annotation type.
