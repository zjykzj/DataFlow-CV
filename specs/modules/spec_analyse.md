# Analyse Module Specification

> **Version:** v1.0 | **Last Updated:** 2026-07-09
> **Status:** Draft — initial specification
> **Layer:** Modules
> **Dependencies:** Label module (handlers + models) + Logging module (LogManager)

## 1. Module Overview

The Analyse module (`dataflow/analyse/`) provides dataset introspection and preparation tools. It depends **only** on the Label module — it does not import from Convert, Visualize, Evaluate, or CLI.

### 1.1 Key Design: Format-Agnostic with Auto-Detection

Unlike Convert and Visualize which use format-specific subclasses, Analyse uses **operation-specific** concrete classes (`StatsAnalyser`, `SplitAnalyser`) that work across all annotation formats via the handler interface. The format is automatically detected from the label path, eliminating the need for the user to specify it explicitly.

### 1.2 Module Contract

- **Input**: A label path (directory or file) pointing to annotations in any supported format (YOLO, LabelMe, COCO)
- **Processing**: Auto-detect format → create handler → read annotations → compute statistics or split
- **Output**: `AnalysisResult` (structured container for statistics or split results)
- **Dependency**: Label module only (for handlers and data models), standard library only

```
dataflow/analyse/
├── __init__.py             # Public API exports
├── base.py                 # BaseAnalyser + AnalysisResult + StatsResult + SplitResult
├── stats.py                # StatsAnalyser
├── split.py                # SplitAnalyser
├── log_templates.py        # Pure formatting functions
└── utils.py                # Format detection, handler factory, class file parsing
```

## 2. Data Models (`base.py`)

### 2.1 `AnalysisResult`

Top-level return type shared by all analysers:

```python
@dataclass
class AnalysisResult:
    success: bool = True
    data: Optional[Any] = None          # StatsResult or SplitResult
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    log_path: Optional[str] = None

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        self.success = False

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)
```

### 2.2 `StatsResult`

Container for dataset statistics:

```python
@dataclass
class StatsResult:
    total_files: int                    # Number of label files (or images in COCO)
    total_annotations: int              # Total annotation objects across all images
    per_class: Dict[str, int]           # class_name → count, ordered by class_file or discovery order
    format: str                         # "yolo" | "labelme" | "coco"
    categories: Dict[int, str] = field(default_factory=dict)  # class_id → class_name
```

**Ordering contract**:
- If `--class-file` is provided: `per_class` is ordered by the class order in the file, with classes not in the file appended at the end (sorted alphabetically)
- If `--class-file` is NOT provided: `per_class` is ordered by count descending, then alphabetically for ties
- COCO format: categories are read from the JSON's `categories` array, maintaining their original order. If `--class-file` is also provided, the class-file order takes precedence

### 2.3 `SplitResult`

Container for train/test split results:

```python
@dataclass
class SplitResult:
    train_count: int                    # Number of images in train set
    val_count: int                      # Number of images in validation set
    train_dir: Path                     # Path to train output directory
    val_dir: Path                       # Path to validation output directory
    ratio: float                        # Train ratio (e.g., 0.8)
    seed: int                           # Random seed used
    format: str                         # "yolo" | "labelme" | "coco"
```

## 3. Core Classes

### 3.1 `BaseAnalyser` (`base.py`)

Shared logging infrastructure for all analysers. Not an abstract class — concrete analysers extend it to inherit logging helpers.

```python
class BaseAnalyser:
    """Shared logging infrastructure for dataset analysers."""

    def __init__(self, log_config: Optional[Any] = None):
        from ..util.logging import LogConfig, LogManager
        if log_config is None:
            log_config = LogConfig(name="analyse")
        self._log_manager = LogManager(log_config)
        self.logger = self._log_manager.logger

    def _log_info(self, message: str) -> None: ...
    def _log_warning(self, message: str) -> None: ...
    def _log_error(self, message: str) -> None:   # Logs error, does NOT raise (read-only operations)
```

**Error handling philosophy**: Unlike Convert (which raises in strict mode) and Evaluate (which always raises), Analyse is a **read-only** operation — it never modifies input data. Therefore `_log_error()` logs at ERROR level but does NOT raise. Errors are accumulated in `AnalysisResult.errors`.

### 3.2 `StatsAnalyser` (`stats.py`)

Computes dataset statistics for any supported annotation format.

```python
class StatsAnalyser(BaseAnalyser):
    """Compute dataset statistics.

    Constructor: ``StatsAnalyser(log_config=None)``
    """

    def analyse(
        self,
        label_path: Path,
        class_file: Optional[Path] = None,
    ) -> AnalysisResult:
        """Compute statistics for the dataset at ``label_path``.

        Args:
            label_path: Path to labels — directory (YOLO/LabelMe) or JSON file (COCO)
            class_file: Optional classes.txt for name mapping and ordering

        Returns:
            AnalysisResult with StatsResult in ``.data``
        """
```

**Pipeline**:

```
1. detect_format(label_path)     → "yolo" | "labelme" | "coco"
2. create_handler(label_path, class_file, logger)
3. handler.read()                → DatasetAnnotations
4. Count:
   a. total_files = len(dataset.images)
   b. total_annotations = sum(len(img.objects) for img in dataset.images)
   c. per_class = tally object counts by class_name
5. Order per_class:
   a. If class_file given: reorder to match class_file order, append unknowns
   b. If no class_file: sort by count descending
6. Log formatted table via log_templates
7. Return AnalysisResult(success=True, data=StatsResult(...))
```

### 3.3 `SplitAnalyser` (`split.py`)

Splits a dataset into train and validation subsets.

```python
class SplitAnalyser(BaseAnalyser):
    """Split dataset into train/val subsets.

    Constructor: ``SplitAnalyser(log_config=None)``
    """

    def analyse(
        self,
        label_path: Path,
        output_dir: Path,
        ratio: float = 0.8,
        seed: int = 42,
        class_file: Optional[Path] = None,
    ) -> AnalysisResult:
        """Split the dataset at ``label_path`` into train and val.

        Args:
            label_path: Path to labels — directory (YOLO/LabelMe) or JSON file (COCO)
            output_dir: Output root directory (train/ and val/ created inside)
            ratio: Proportion of data for training (default 0.8)
            seed: Random seed for reproducibility
            class_file: Optional classes.txt (required for YOLO)

        Returns:
            AnalysisResult with SplitResult in ``.data``
        """
```

**Pipeline**:

```
1. detect_format(label_path)     → "yolo" | "labelme" | "coco"
2. create_handler(label_path, class_file, logger)
3. handler.read()                → DatasetAnnotations
4. Shuffle images:
   a. random.Random(seed).shuffle(dataset.images)
5. Split by ratio:
   a. split_idx = int(len(images) * ratio)
   b. train_images = images[:split_idx]
   c. val_images = images[split_idx:]
6. Create two DatasetAnnotations (train, val) from image subsets
7. Write outputs:
   a. COCO: handler.write() for train.json and val.json in output_dir/
   b. YOLO/LabelMe: iterate images, handler.write_one() per image
      into output_dir/train/ and output_dir/val/
8. Copy class_file to both train/ and val/ (if provided)
9. Return AnalysisResult(success=True, data=SplitResult(...))
```

**Split behavior contract**:

| Property | Behavior |
|----------|----------|
| Random seed | `random.Random(seed)` for deterministic shuffling |
| Ratio | `train_count = int(total_images * ratio)`, remainder to val |
| Minimum size | If ratio results in empty train or val set, warn but proceed (single-image datasets) |
| Image copying | Images are NOT copied in v1 — only labels are split |
| Class file | Copied to both output directories if provided |

### 3.4 Constructor Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `log_config` | Optional[LogConfig] | No | None | Logging configuration. If None, default `LogConfig(name="analyse")` is used. |

## 4. Public API

### 4.1 `StatsAnalyser.analyse(label_path, class_file) → AnalysisResult`

Compute dataset statistics. Auto-detects the annotation format from `label_path`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `label_path` | Path | Yes | Path to labels (directory or file) |
| `class_file` | Path | No | Classes.txt for name mapping and output ordering |

### 4.2 `SplitAnalyser.analyse(label_path, output_dir, ratio, seed, class_file) → AnalysisResult`

Split dataset into train and validation subsets.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `label_path` | Path | Yes | — | Path to labels (directory or file) |
| `output_dir` | Path | Yes | — | Output root directory |
| `ratio` | float | No | 0.8 | Train proportion |
| `seed` | int | No | 42 | Random seed |
| `class_file` | Path | No | None | Classes.txt (required for YOLO) |

### 4.3 Module-Level Utilities (`utils.py`)

```python
def detect_format(label_path: Path) -> str:
    """Auto-detect annotation format from path.

    Detection rules (checked in order):
    1. If label_path is a file ending in .json → "coco"
    2. If label_path is a directory:
       a. List non-hidden files, exclude subdirectories
       b. If any file is .json:
          - Open first .json, check for "shapes" key → "labelme"
          - Check for "images" key → "coco" (unusual for dir, but handle)
          - Otherwise → error
       c. If any file is .txt → "yolo"
       d. If directory is empty → error
    3. Neither → raise ValueError

    Returns:
        "yolo" | "labelme" | "coco"
    """

def create_handler(
    label_path: Path,
    format: str,
    class_file: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> BaseAnnotationHandler:
    """Create the appropriate handler for the detected format.

    Args:
        label_path: Path to labels
        format: "yolo" | "labelme" | "coco"
        class_file: Classes.txt path (used for YOLO)
        logger: Logger to pass to the handler

    Returns:
        Configured BaseAnnotationHandler instance with strict_mode=False
    """

def load_class_names(class_file: Path) -> Dict[int, str]:
    """Parse classes.txt → {class_id: class_name}.

    Format: one class name per line, 0-indexed.
    Blank lines and lines starting with # are skipped.
    """
```

## 5. Input Format Contract

### 5.1 Format Detection

The module auto-detects the annotation format by inspecting the label path:

| Label Path Type | Detection Method | Result |
|----------------|-----------------|--------|
| File ending in `.json` | Read JSON, check for top-level keys | COCO if `"images"` key present |
| Directory with `.txt` files | File extension check | YOLO |
| Directory with `.json` files | Read first `.json`, check for `"shapes"` key | LabelMe |
| Empty directory | — | Error: "No annotation files found" |
| Ambiguous (mixed extensions) | — | Error: "Cannot determine format" |

### 5.2 Handler Configuration

All handlers are created with `strict_mode=False` — analysis operations are read-only and should be lenient with imperfect data. Skipped files produce warnings, not errors.

### 5.3 Class File Contract

- YOLO format: `class_file` is **strongly recommended**. Without it, class names display as `"class_0"`, `"class_1"`, etc.
- LabelMe format: `class_file` is **optional** — class names are read from JSON files. If provided, it controls output ordering only.
- COCO format: `class_file` is **optional** — categories are read from the JSON's `categories` array. If provided, it controls output ordering only.

## 6. Logging Contract

See [`spec_logging.md`](spec_logging.md) for the full `LogManager` contract. Analyse-specific:

**Constructor**: `BaseAnalyser.__init__(log_config=None)`

- If `log_config` is None, a default `LogConfig(name="analyse")` is created
- The analyser creates a `LogManager` from the config

### 6.1 Stats Output Format

```
══════════════════════════════════════
Analyse: Dataset Statistics
  Source: yolo_labels/
  Format: yolo (auto-detected)

── Summary ──
  Total files:       50
  Total annotations: 95
  Categories:        3

── Per-Class ──
┌────────────┬───────┐
│ Class      │ Count │
├────────────┼───────┤
│ cat        │    42 │
│ dog        │    38 │
│ person     │    15 │
└────────────┴───────┘

── Result ──
  Status: ✓ Success

  Log saved to: logs/analyse_stats_20260709_100000.log
══════════════════════════════════════
```

### 6.2 Split Output Format

```
══════════════════════════════════════
Analyse: Train/Test Split
  Source: yolo_labels/
  Format: yolo (auto-detected)
  Ratio:  0.8
  Seed:   42

── Split ──
  Train: 40 images → output/train/
  Val:   10 images → output/val/

── Result ──
  Status: ✓ Success

  Log saved to: logs/analyse_split_20260709_100000.log
══════════════════════════════════════
```

### 6.3 Log Templates

Log templates are in `dataflow/analyse/log_templates.py`:

- `format_analyse_header(operation, label_path, format)` — header block
- `format_stats_result(total_files, total_annotations, per_class, categories)` — stats table
- `format_split_result(train_count, val_count, train_dir, val_dir, ratio, seed)` — split summary
- `format_analyse_result(status, log_path)` — final result block

## 7. Dependency Contract

```
Analyse module imports FROM:
├── dataflow.label.models             (DatasetAnnotations, ImageAnnotation, ObjectAnnotation, AnnotationFormat)
├── dataflow.label.base               (BaseAnnotationHandler)
├── dataflow.label.yolo_handler       (YoloAnnotationHandler)
├── dataflow.label.labelme_handler    (LabelMeAnnotationHandler)
├── dataflow.label.coco_handler       (CocoAnnotationHandler)
├── dataflow.util.logging             (LogConfig, LogManager, format_table, format_divider,
│                                       format_section, format_kv, format_result_block)
├── pathlib                           (Path)
├── random                            (Random — for split shuffling)
└── json                              (for format detection)

Analyse module does NOT import FROM:
├── dataflow.convert.*                (FORBIDDEN — zero cross-dependency)
├── dataflow.visualize.*              (FORBIDDEN — zero cross-dependency)
├── dataflow.evaluate.*               (FORBIDDEN — zero cross-dependency)
├── dataflow.cli.*                    (FORBIDDEN — CLI depends on Analyse, not vice versa)
└── pycocotools                       (NOT NEEDED — analysis is format-agnostic)
```

## 8. Architecture Position

```
┌──────────────────────────────────────────────────────────────┐
│                           CLI                                 │
│  (passes LogConfig to modules; click.echo() for terminal UI)  │
└──┬──────────┬──────────────┬──────────────┬──────────────────┘
   │          │              │              │
   ▼          ▼              ▼              ▼
┌──────────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐
│   Analyse    │ │Convert │ │Visualize │ │Evaluate  │
│(introspection)│ │(pipeline)│ │(rendering)│ │(metrics) │
│  LogManager  │ │LogMgr  │ │LogMgr    │ │LogMgr    │
└──────┬───────┘ └──┬─────┘ └────┬─────┘ └────┬─────┘
       │            │            │            │
       │   ZERO CROSS-DEPENDENCY │            │
       │            │            │            │
       ▼            ▼            ▼            ▼
┌──────────────────────────────────────────────────────────────┐
│                         Label                                 │
│  Data Models + Handlers (read/write/validate)                 │
│  (receive logger from calling module)                         │
└──────────────────────────────────────────────────────────────┘
       │            │            │            │
       └────────────┼────────────┼────────────┘
                    │            │
                    ▼            ▼
┌──────────────────────────────────────────────────────────────┐
│                    util/logging.py                             │
│  LogManager + format helpers (shared infrastructure)           │
└──────────────────────────────────────────────────────────────┘
```

**Hard constraints:**

1. **Analyse ↔ Convert**: Zero dependency. They do not import from each other.
2. **Analyse ↔ Visualize**: Zero dependency. They do not import from each other.
3. **Analyse ↔ Evaluate**: Zero dependency. They do not import from each other.
4. **Analyse → Label**: Analysers import handlers and models from the Label module only through public interfaces.
5. **CLI → Analyse**: CLI commands only call analyser public APIs. CLI must NOT import label handlers directly.
6. **Logging ownership**: All log output is produced by the Analyse module, not CLI. CLI passes `LogConfig` to analyser constructors and uses `click.echo()` for terminal UI.

## 9. Error Handling Contract

### 9.1 Error Propagation

All errors are accumulated in `AnalysisResult.errors`. The analyser never raises exceptions for data issues — it is a read-only operation.

```
Format detection fails     → AnalysisResult(success=False, errors=[...])
Handler.read() fails       → AnalysisResult(success=False, errors=[...])
Split output write fails   → AnalysisResult(success=False, errors=[...])
```

### 9.2 Specific Error Scenarios

| Error | Behavior |
|-------|----------|
| Label path not found | `AnalysisResult(success=False, errors=["Label path not found: ..."])` |
| Cannot determine format | `AnalysisResult(success=False, errors=["Cannot determine annotation format from: ..."])` |
| Empty dataset (no annotations) | `AnalysisResult(success=True, data=StatsResult(total_files=0, ...))` — not an error |
| Class file not found | `AnalysisResult(success=False, errors=["Class file not found: ..."])` |
| Split ratio out of range | `AnalysisResult(success=False, errors=["Ratio must be between 0 and 1, got: ..."])` |
| Output directory not writable | `AnalysisResult(success=False, errors=["Cannot create output directory: ..."])` |
| Skipped invalid files (non-strict) | Warning in `warnings` list; analysis continues |

### 9.3 Non-Strict Mode

All analyser operations run in **non-strict mode by default** (handlers created with `strict_mode=False`):

- Corrupted/invalid annotation files are **skipped** with a warning
- Missing images are **ignored** (files counted from label files, not images)
- The operation continues with valid data
- Skipped counts are reported in the warnings list

## 10. CLI Integration (Planned)

### 10.1 Command Signatures

```
dataflow-cv analyse stats [OPTIONS] LABEL_PATH
dataflow-cv analyse split [OPTIONS] LABEL_PATH OUTPUT_DIR
```

### 10.2 Options

| Option | Applies To | Type | Default | Description |
|--------|-----------|------|---------|-------------|
| `--verbose` | both | Flag | False | Enable verbose log output |
| `--class-file`, `-c` | both | Path | None | Classes.txt for name mapping |
| `--output`, `-o` | stats | Path | None | Save stats as JSON |
| `--ratio`, `-r` | split | Float | 0.8 | Train proportion |
| `--seed`, `-s` | split | Int | 42 | Random seed |

## 11. Summary of Public API

| API | Location | Purpose |
|-----|----------|---------|
| `StatsAnalyser(log_config)` | `stats.py` | Dataset statistics computation |
| `analyser.analyse(label_path, class_file) → AnalysisResult` | `stats.py` | Run statistics |
| `SplitAnalyser(log_config)` | `split.py` | Train/val dataset splitting |
| `analyser.analyse(label_path, output_dir, ratio, seed, class_file) → AnalysisResult` | `split.py` | Run split |
| `detect_format(label_path) → str` | `utils.py` | Auto-detect annotation format |
| `create_handler(label_path, format, class_file, logger) → BaseAnnotationHandler` | `utils.py` | Handler factory |
| `load_class_names(class_file) → Dict[int, str]` | `utils.py` | Parse classes.txt |
