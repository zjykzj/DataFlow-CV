# Analyse Module Specification

> **Version:** v1.1 | **Last Updated:** 2026-07-13
> **Status:** Draft — adding filter functionality
> **Layer:** Modules
> **Dependencies:** Label module (handlers + models) + Logging module (LogManager)

## 1. Module Overview

The Analyse module (`dataflow/analyse/`) provides dataset introspection and preparation tools. It depends **only** on the Label module — it does not import from Convert, Visualize, Evaluate, or CLI.

### 1.1 Key Design: Format-Agnostic with Auto-Detection

Unlike Convert and Visualize which use format-specific subclasses, Analyse uses **operation-specific** concrete classes (`StatsAnalyser`, `SplitAnalyser`, `FilterAnalyser`) that work across all annotation formats via the handler interface. The format is automatically detected from the label path, eliminating the need for the user to specify it explicitly.

### 1.2 Module Contract

- **Input**: A label path (directory or file) pointing to annotations in any supported format (YOLO, LabelMe, COCO)
- **Processing**: Auto-detect format → create handler → read annotations → compute statistics, split, or filter
- **Output**: `AnalysisResult` (structured container for statistics, split, or filter results)
- **Dependency**: Label module only (for handlers and data models), standard library only

```
dataflow/analyse/
├── __init__.py             # Public API exports
├── base.py                 # BaseAnalyser + AnalysisResult + StatsResult + SplitResult + FilterResult
├── stats.py                # StatsAnalyser
├── split.py                # SplitAnalyser
├── filter.py               # FilterAnalyser
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
    data: Optional[Any] = None          # StatsResult, SplitResult, or FilterResult
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
- If `--class-file` is provided: `per_class` is ordered by the class order in the file, with classes not in the file appended at the end (sorted alphabetically). Sort options (`--sort-by`, `--descending`) are ignored.
- If `--class-file` is NOT provided: `per_class` is ordered by `--sort-by` + `--descending/--ascending`, ties broken alphabetically. Default: class_id ascending.

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

### 2.4 `FilterResult`

Container for category-based annotation filtering results.

```python
@dataclass
class FilterResult:
    total_files: int                              # Total label files processed
    total_files_with_annotations: int             # Files that still have annotations after filtering
    total_annotations_before: int                 # Annotation count before filtering
    total_annotations_after: int                  # Annotation count after filtering
    kept_categories: List[CategoryMapping]        # Categories retained (in new class file order)
    removed_categories: List[RemovedCategory]     # Categories removed
    missing_categories: List[str]                 # Categories in new class file but not in source
    output_dir: Path                              # Output directory
    format: str                                   # "yolo" | "labelme" | "coco"

@dataclass
class CategoryMapping:
    """A category that was kept during filtering."""
    new_id: int          # New class ID (line index in new_classes.txt)
    old_id: int          # Original class ID in source data
    name: str            # Class name

@dataclass
class RemovedCategory:
    """A category that was removed during filtering."""
    old_id: int          # Original class ID in source data
    name: str            # Class name
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
        image_dir: Optional[Path] = None,
        sort_by: str = "id",
        descending: bool = False,
    ) -> AnalysisResult:
        """Compute statistics for the dataset at ``label_path``.

        Args:
            label_path: Path to labels — directory (YOLO/LabelMe) or JSON file (COCO)
            class_file: Optional classes.txt for name mapping and ordering (overrides sort options)
            image_dir: Optional image directory (needed for YOLO format to locate image dimensions)
            sort_by: ``"id"`` (default, class_id ascending) or ``"count"`` (annotation count)
            descending: When True, reverse sort direction

        Returns:
            AnalysisResult with StatsResult in ``.data``
        """
```

**Pipeline**:

```
1. detect_format(label_path)     → "yolo" | "labelme" | "coco"
2. create_handler(format, label_path, class_file, image_dir, logger)
3. handler.read()                → DatasetAnnotations
4. Count:
   a. total_files = len(dataset.images)
   b. total_annotations = sum(len(img.objects) for img in dataset.images)
   c. per_class = tally object counts by class_name
5. Order per_class:
   a. If class_file given: reorder to match class_file order, append unknowns alphabetically
   b. If no class_file: sort by ``sort_by`` + ``descending`` (default: class_id ascending), ties alphabetically
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
        image_dir: Optional[Path] = None,
    ) -> AnalysisResult:
        """Split the dataset at ``label_path`` into train and val.

        Args:
            label_path: Path to labels — directory (YOLO/LabelMe) or JSON file (COCO)
            output_dir: Output root directory (train/ and val/ created inside)
            ratio: Proportion of data for training (default 0.8)
            seed: Random seed for reproducibility
            class_file: Optional classes.txt (required for YOLO)
            image_dir: Optional image directory (needed for YOLO format to locate image dimensions)

        Returns:
            AnalysisResult with SplitResult in ``.data``
        """
```

**Pipeline**:

```
1. detect_format(label_path)     → "yolo" | "labelme" | "coco"
2. create_handler(format, label_path, class_file, image_dir, logger)
3. handler.read()                → DatasetAnnotations
4. Shuffle images:
   a. random.Random(seed).shuffle(dataset.images)
5. Split by ratio:
   a. split_idx = int(len(images) * ratio)
   b. If split_idx == 0 and len(images) >= 2 → silently adjust to 1
   c. If split_idx == len(images) and len(images) >= 2 → silently adjust to len(images)-1
   d. train_images = images[:split_idx]
   e. val_images = images[split_idx:]
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
| Minimum size | If ratio results in empty train or val set and dataset has >= 2 images, silently adjust boundaries (min 1 per side). Single-image datasets produce empty val set with a warning. |
| Image copying | Images are NOT copied in v1 — only labels are split |
| Class file | Copied to both output directories if provided |

### 3.4 `FilterAnalyser` (`filter.py`)

Filters a dataset's annotations by category, retaining only specified classes and re-mapping class IDs.

```python
class FilterAnalyser(BaseAnalyser):
    """Filter dataset annotations by category.

    Constructor: ``FilterAnalyser(log_config=None)``
    """

    def analyse(
        self,
        label_path: Path,
        original_class_file: Path,
        new_class_file: Path,
        output_dir: Path,
        image_dir: Optional[Path] = None,
    ) -> AnalysisResult:
        """Filter annotations, keeping only categories listed in new_class_file.

        Args:
            label_path: Path to labels — directory (YOLO/LabelMe) or JSON file (COCO)
            original_class_file: Source class file (required — defines source class names)
            new_class_file: Target class file (required — defines which categories to keep
                and their new order/IDs)
            output_dir: Output root directory
            image_dir: Optional image directory for YOLO format
                (auto-detected if omitted)

        Returns:
            AnalysisResult with FilterResult in ``.data``
        """
```

**Pipeline**:

```
1. Validate args (both class files must exist)
2. Load original class mapping: load_class_names(original_class_file) → {id: name}
3. Load new class mapping:   load_class_names(new_class_file)   → {id: name}
4. Build filter mapping:
   a. For each (new_id, name) in new_class_file:
      - Find old_id where original_classes[old_id] == name
      - If not found → record in missing_categories, emit WARNING
      - If found → record mapping: old_id → (new_id, name)
   b. For each (old_id, name) in original_class_file not in kept set:
      - Record in removed_categories
5. detect_format(label_path) → "yolo" | "labelme" | "coco"
6. create_handler(format, label_path, original_class_file, image_dir, logger)
7. Read source: handler.read() → DatasetAnnotations
8. Filter annotations per image:
   a. For each ObjectAnnotation in each ImageAnnotation:
      - If obj.class_id is in filter mapping → keep, remap class_id to new_id
      - Otherwise → discard
   b. Count before/after totals
9. Create filtered DatasetAnnotations with new categories dict
   (from new_class_file only)
10. Write outputs:
    a. COCO: handler.write() → single JSON in output_dir/
    b. YOLO/LabelMe: handler.write_one() per image into output_dir/
11. Copy new_class_file to output_dir as "classes.txt"
12. Return AnalysisResult(success=True, data=FilterResult(...))
```

**Filter mapping contract**:

| Scenario | Behavior |
|----------|----------|
| `new_class_file` contains a subset of original classes | Only those classes are kept, IDs remapped to new file's line order |
| `new_class_file` contains a class NOT in original | WARNING emitted, class recorded in `missing_categories`, not in output |
| `new_class_file` reorders classes | Output follows new_class_file order (e.g., original `[dog, cat, bird]`, new `[bird, dog]` → output has `bird=0, dog=1`) |
| `new_class_file` is identical to original | All classes kept, order unchanged (no-op filter, ID mapping is identity) |
| Image has zero annotations after filtering | File still written (empty YOLO `.txt` / LabelMe `shapes: []` / no COCO annotation entries) |

**Read/write paths**:

| Source → Target (all same-format) | Path |
|-----------------------------------|------|
| YOLO → YOLO | Streaming: `iter_images()` → `write_one()` per image |
| LabelMe → LabelMe | Streaming: `iter_images()` → `write_one()` per image |
| COCO → COCO | Batch: `read()` → `write()` single JSON |

Filter is **same-format only** — the output format matches the source format. Cross-format filtering is not supported; use Convert for format changes.

### 3.5 Constructor Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `log_config` | Optional[LogConfig] | No | None | Logging configuration. If None, default `LogConfig(name="analyse")` is used. |

## 4. Public API

### 4.1 `StatsAnalyser.analyse(label_path, class_file, image_dir, sort_by, descending) → AnalysisResult`

Compute dataset statistics. Auto-detects the annotation format from `label_path`.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `label_path` | Path | Yes | — | Path to labels (directory or file) |
| `class_file` | Path | No | None | Classes.txt for name mapping and output ordering |
| `image_dir` | Path | No | None | Image directory (needed for YOLO format to locate image dimensions) |
| `sort_by` | str | No | `"id"` | Sort key: `"id"` (class_id) or `"count"` (annotation count). Ignored when `class_file` is provided. |
| `descending` | bool | No | False | When True, reverse sort direction |

### 4.2 `SplitAnalyser.analyse(label_path, output_dir, ratio, seed, class_file, image_dir) → AnalysisResult`

Split dataset into train and validation subsets.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `label_path` | Path | Yes | — | Path to labels (directory or file) |
| `output_dir` | Path | Yes | — | Output root directory |
| `ratio` | float | No | 0.8 | Train proportion |
| `seed` | int | No | 42 | Random seed |
| `class_file` | Path | No | None | Classes.txt (required for YOLO) |
| `image_dir` | Path | No | None | Image directory (needed for YOLO format to locate image dimensions) |

### 4.3 `FilterAnalyser.analyse(label_path, original_class_file, new_class_file, output_dir, image_dir) → AnalysisResult`

Filter annotations by category, keeping only classes listed in `new_class_file`.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `label_path` | Path | Yes | — | Path to labels (directory or file) |
| `original_class_file` | Path | Yes | — | Source classes.txt defining all categories in the source dataset |
| `new_class_file` | Path | Yes | — | Target classes.txt defining which categories to keep and their new order |
| `output_dir` | Path | Yes | — | Output directory |
| `image_dir` | Path | No | None | Image directory (needed for YOLO format to locate image dimensions) |

### 4.4 Module-Level Utilities (`utils.py`)

```python
def detect_format(label_path: Path) -> str:
    """Auto-detect annotation format from path.

    Detection rules (checked in order):
    1. If label_path is a file ending in .json:
       - Open file, check keys in order:
         a. ``"images"`` → ``"coco"``
         b. ``"shapes"`` → ``"labelme"``
         c. ``"annotations"`` → ``"coco"`` (both ``"images"`` and ``"annotations"`` individually indicate COCO)
         d. otherwise → error
    2. If label_path is a directory:
       a. List non-hidden files, exclude ``classes.txt``
       b. If mixed .txt and .json → error (ambiguous)
       c. If .txt only → ``"yolo"``
       d. If .json only → open first, check for ``"shapes"`` key → ``"labelme"``;
          if ``"images"`` key → error (COCO is always a single file)
       e. If empty → error
    3. Neither → raise ValueError

    Returns:
        "yolo" | "labelme" | "coco"
    """

def create_handler(
    label_path: Path,
    format: str,
    class_file: Optional[Path] = None,
    image_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> BaseAnnotationHandler:
    """Create the appropriate handler for the detected format.

    Args:
        label_path: Path to labels
        format: "yolo" | "labelme" | "coco"
        class_file: Classes.txt path (required for YOLO)
        image_dir: Image directory for YOLO (auto-detected if omitted)
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
| File ending in `.json` | Read JSON, check in order: ``"images"`` → COCO, ``"shapes"`` → LabelMe, ``"annotations"`` → COCO. Both ``"images"`` and ``"annotations"`` individually indicate COCO. | COCO or LabelMe |
| Directory with `.txt` files only | File extension check | YOLO |
| Directory with `.json` files only | Read first `.json`, check for ``"shapes"`` key | LabelMe |
| Directory with ``"images"``-key JSON | — | Error: "COCO is a single file — point to it directly" |
| Empty directory | — | Error: "No annotation files found" |
| Mixed extensions or ambiguous | — | Error with diagnostic message |

### 5.2 Handler Configuration

All handlers are created with `strict_mode=False` — analysis operations are read-only and should be lenient with imperfect data. Skipped files produce warnings, not errors.

### 5.3 Class File Contract

**For stats/split operations:**

- YOLO format: `class_file` is **optional** — if not provided, a temporary classes.txt is auto-generated by scanning label files for observed class IDs, producing placeholder names like ``"class_0"``, ``"class_1"``, etc. Provide it explicitly for real class names.
- LabelMe format: `class_file` is **optional** — class names are read from JSON files. If provided, it controls output ordering only.
- COCO format: `class_file` is **optional** — categories are read from the JSON's `categories` array. If provided, it controls output ordering only.

**For filter operation:**

- `original_class_file` is **required** — defines the source dataset's full category set (class_id → class_name). This is the ground truth for which categories exist in the source data.
- `new_class_file` is **required** — defines which categories to keep and their new order. Each line's position determines the new class_id. Classes in this file that do not appear in the original file trigger a WARNING and are recorded in `FilterResult.missing_categories`.
- Both files follow the same format: one class name per line, 0-indexed, blank lines and `#`-prefixed lines skipped.

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

### 6.3 Filter Output Format

```
══════════════════════════════════════
Analyse: Category Filter
  Source:         yolo_labels/
  Original class: classes.txt (10 categories)
  New class:      new_classes.txt (6 categories)
  Format:         yolo (auto-detected)

── Category Comparison ──
  Kept (remapped):     6 categories
    [0] dog        (was: class_id=2)
    [1] cat        (was: class_id=0)
    [2] bird       (was: class_id=4)
    [3] fish       (was: class_id=7)
    [4] horse      (was: class_id=1)
    [5] sheep      (was: class_id=3)
  Removed:             4 categories
    class_id=5  "car"
    class_id=6  "truck"
    class_id=8  "person"
    class_id=9  "bicycle"
  Not found in source: 0 categories

── Filter Summary ──
  Total files:              50
  Files with annotations:   48
  Annotations before:       240
  Annotations after:        156
  Output:                   48 files → output/

── Result ──
  Status: ✓ Success

  Log saved to: logs/analyse_filter_20260713_100000.log
══════════════════════════════════════
```

### 6.4 Log Templates

Log templates are in `dataflow/analyse/log_templates.py`:

- `format_analyse_header(operation, label_path, format_name)` — header block
- `format_stats_result(total_files, total_annotations, per_class)` — stats table
- `format_split_result(train_count, val_count, train_dir, val_dir, ratio, seed)` — split summary
- `format_filter_result(total_files, total_files_with_annotations, annotations_before, annotations_after, kept_categories, removed_categories, missing_categories, output_dir)` — filter comparison + summary
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
| New class file has no valid entries | `AnalysisResult(success=False, errors=["No valid class names in new class file: ..."])` |
| No matching categories found | `AnalysisResult(success=False, errors=["No matching categories between source and new class file"])` |
| New class file has entries not in source | WARNING per missing class; class recorded in `FilterResult.missing_categories` |
| All annotations filtered out | Not an error — result reports `total_annotations_after=0` with warning |

### 9.3 Non-Strict Mode

All analyser operations run in **non-strict mode by default** (handlers created with `strict_mode=False`):

- Corrupted/invalid annotation files are **skipped** with a warning
- Missing images are **ignored** (files counted from label files, not images)
- The operation continues with valid data
- Skipped counts are reported in the warnings list

## 10. CLI Integration

### 10.1 Command Signatures

```
dataflow-cv analyse stats [OPTIONS] LABEL_PATH
dataflow-cv analyse split [OPTIONS] LABEL_PATH OUTPUT_DIR
dataflow-cv analyse filter [OPTIONS] LABEL_PATH ORIGINAL_CLASS_FILE NEW_CLASS_FILE OUTPUT_DIR
```

### 10.2 Options

| Option | Applies To | Type | Default | Description |
|--------|-----------|------|---------|-------------|
| `--verbose` | all three | Flag | False | Enable verbose log output |
| `--log-dir` | all three | Path | `./logs` | Log file output directory |
| `--class-file`, `-c` | stats, split | Path | None | Classes.txt for name mapping |
| `--image-dir` | all three | Path | None | Image directory for YOLO (auto-detected if omitted: tries ``labels/images/``, ``dataset/images/``, ``dataset_parent/images/``) |
| `--sort-by` | stats | Choice | `"id"` | `"id"` or `"count"` |
| `--descending/--ascending` | stats | Flag | `--ascending` | Sort direction |
| `--ratio`, `-r` | split | Float | 0.8 | Train proportion |
| `--seed`, `-s` | split | Int | 42 | Random seed |

### 10.3 Filter Command

```
dataflow-cv analyse filter [OPTIONS] LABEL_PATH ORIGINAL_CLASS_FILE NEW_CLASS_FILE OUTPUT_DIR
```

| Argument / Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `LABEL_PATH` (argument) | Path | *(required)* | Directory (YOLO/LabelMe) or COCO JSON file |
| `ORIGINAL_CLASS_FILE` (argument) | Path | *(required)* | Source classes.txt — defines which categories exist in the source data |
| `NEW_CLASS_FILE` (argument) | Path | *(required)* | Target classes.txt — defines which categories to keep and their new order/IDs |
| `OUTPUT_DIR` (argument) | Path | *(required)* | Output root directory |
| `--image-dir` | Path | None | Image directory for YOLO format (auto-detected if omitted) |

Plus shared options: `--verbose`, `--log-dir`.

**Behavior by format**:

| Format | Read | Write | Output |
|--------|------|-------|--------|
| YOLO | `iter_images()` streaming | `write_one()` streaming per `.txt` | Individual `.txt` files in `OUTPUT_DIR/` |
| LabelMe | `iter_images()` streaming | `write_one()` streaming per `.json` | Individual `.json` files in `OUTPUT_DIR/` |
| COCO | `read()` batch | `write()` batch single JSON | Single `.json` file in `OUTPUT_DIR/` |

The filtered `classes.txt` (from `NEW_CLASS_FILE`) is copied to `OUTPUT_DIR/`.

## 11. Summary of Public API

| API | Location | Purpose |
|-----|----------|---------|
| `StatsAnalyser(log_config)` | `stats.py` | Dataset statistics computation |
| `analyser.analyse(label_path, class_file, image_dir) → AnalysisResult` | `stats.py` | Run statistics |
| `SplitAnalyser(log_config)` | `split.py` | Train/val dataset splitting |
| `analyser.analyse(label_path, output_dir, ratio, seed, class_file, image_dir) → AnalysisResult` | `split.py` | Run split |
| `FilterAnalyser(log_config)` | `filter.py` | Category-based annotation filtering |
| `analyser.analyse(label_path, original_class_file, new_class_file, output_dir, image_dir) → AnalysisResult` | `filter.py` | Run filter |
| `detect_format(label_path) → str` | `utils.py` | Auto-detect annotation format |
| `create_handler(label_path, format, class_file, image_dir, logger) → BaseAnnotationHandler` | `utils.py` | Handler factory |
| `load_class_names(class_file) → Dict[int, str]` | `utils.py` | Parse classes.txt |
