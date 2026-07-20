# Analyse Module Specification

> **Version:** v2.0 | **Last Updated:** 2026-07-20
> **Status:** Stable
> **Layer:** Modules
> **Dependencies:** Label module (handlers + models) + Logging module (LogManager)

## 1. Module Overview

The Analyse module (`dataflow/analyse/`) provides dataset introspection and preparation tools. It depends **only** on the Label module — it does not import from Convert, Visualize, Evaluate, or CLI.

### 1.1 Key Design: Format-Agnostic with Auto-Detection

Unlike Convert and Visualize which use format-specific subclasses, Analyse uses **operation-specific** concrete classes (`StatsAnalyser`, `SplitAnalyser`, `FilterAnalyser`, `PartitionAnalyser`) that work across all annotation formats via the handler interface. The format is automatically detected from the label path, eliminating the need for the user to specify it explicitly.

`StatsAnalyser` supports **multiple label paths** — statistics from each path are computed independently and merged into a single aggregated result. All paths must share the same format and, when a class file is provided, the same category list. A `--recursive` flag enables recursive traversal of subdirectories for YOLO and LabelMe formats.

`PartitionAnalyser` supports **N-way dataset partitioning** — splitting a dataset into N equal(ish) parts. Three modes: labels-only, images-only, or labels+images together (labels drive the partition, images follow by stem matching). With `--shuffle` for random distribution and `--move` for storage-constrained scenarios where copying isn't feasible. Supports YOLO and LabelMe formats only (not COCO).

### 1.2 Module Contract

- **Input**: One or more label paths (directories or files) pointing to annotations in supported formats (YOLO, LabelMe). For partition: also supports raw image directories without annotations. All paths must be the same format. An optional `--recursive` flag enables recursive subdirectory traversal for YOLO and LabelMe formats.
- **Processing**: Auto-detect format → create handler(s) → read annotations → compute statistics, split, filter, or partition → merge (stats only)
- **Output**: `AnalysisResult` (structured container for statistics, split, filter, or partition results)
- **Dependency**: Label module only (for handlers and data models), standard library only

```
dataflow/analyse/
├── __init__.py             # Public API exports
├── base.py                 # BaseAnalyser + AnalysisResult + StatsResult + SplitResult + FilterResult + PartitionResult
├── stats.py                # StatsAnalyser
├── split.py                # SplitAnalyser
├── filter.py               # FilterAnalyser
├── partition.py            # PartitionAnalyser
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
    data: Optional[Any] = None          # StatsResult, SplitResult, FilterResult, or PartitionResult
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
    source_paths: List[Path] = field(default_factory=list)   # Paths that contributed to this result
```

**Ordering contract**:
- If `--class-file` is provided: `per_class` is ordered by the class order in the file, with classes not in the file appended at the end (sorted alphabetically). Sort options (`--sort-by`, `--descending`) are ignored.
- If `--class-file` is NOT provided: `per_class` is ordered by `--sort-by` + `--descending/--ascending`, ties broken alphabetically. Default: class_id ascending.

### 2.3 `SplitResult`

Container for train/test split results:

```python
@dataclass
class SplitResult:
    train_count: int                    # Number of files in train set
    val_count: int                      # Number of files in validation set
    train_dir: Path                     # Path to train output directory
    val_dir: Path                       # Path to validation output directory
    ratio: float                        # Train ratio (e.g., 0.8)
    seed: int                           # Random seed used
    format: str                         # "yolo" | "labelme" | "" (empty for images-only)
    mode: str                           # "labels" | "images" | "both"
    move: bool                          # Whether move mode was used
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

### 2.5 `PartitionResult`

Container for N-way dataset partition results.

```python
@dataclass
class PartitionResult:
    num_partitions: int                           # Number of partitions (N)
    partition_sizes: List[int]                    # File count per partition, e.g. [20000, 20000, 20002]
    partition_dirs: List[Path]                    # Path to each partition directory
    total_files: int                              # Total files processed
    seed: int                                     # Random seed (meaningful only when shuffle=True, default 42)
    shuffle: bool                                 # Whether shuffle was applied
    mode: str                                     # "images" | "labels" | "both"
    format: str                                   # "yolo" | "labelme" | "" (empty for images-only mode)
    move: bool                                    # Whether move mode was used
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

Computes dataset statistics for any supported annotation format. Supports multiple label paths (merged into a single result) and recursive subdirectory traversal.

```python
class StatsAnalyser(BaseAnalyser):
    """Compute dataset statistics.

    Constructor: ``StatsAnalyser(log_config=None)``
    """

    def analyse(
        self,
        label_paths: List[Path],
        class_file: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        sort_by: str = "id",
        descending: bool = False,
        recursive: bool = False,
    ) -> AnalysisResult:
        """Compute statistics for the datasets at ``label_paths``.

        Args:
            label_paths: One or more paths to labels — directories
                (YOLO/LabelMe) or JSON files (COCO).  All paths must
                be the same format.
            class_file: Optional classes.txt for name mapping and
                ordering (overrides sort options).  When provided,
                categories in the data that are NOT in this file
                cause an ERROR.
            image_dir: Optional image directory (needed for YOLO
                format to locate image dimensions).
            sort_by: ``"id"`` (default, class_id ascending) or
                ``"count"`` (annotation count).
            descending: When True, reverse sort direction.
            recursive: When True, recursively traverse subdirectories
                for YOLO and LabelMe formats.  Ignored for COCO
                (single-file format).

        Returns:
            AnalysisResult with StatsResult in ``.data``
        """
```

**Pipeline** (per path, then merge):

```
For each label_path:
  1. detect_format(label_path)     → "yolo" | "labelme" | "coco"
     - All paths must yield the same format; first path sets the
       expected format; subsequent paths with different formats
       produce an ERROR.
  2. create_handler(fmt, label_path, class_file, image_dir, logger,
       skip_image_loading=True, recursive=recursive)
  3. handler.read()                → DatasetAnnotations
  4. Count:
     a. total_files += len(dataset.images)
     b. total_annotations += sum(len(img.objects) for img in dataset.images)
     c. per_class = merge tally by class_name across all paths
  5. If class_file provided:
     a. Strict validation: any class_name in data NOT in class_file → ERROR
        (reports the specific path and unknown class names)
     b. Reorder per_class to match class_file order, drop unknowns
        (none should exist after validation)

After all paths:
  6. If class_file NOT provided:
     a. Sort per_class by sort_by + descending (default: class_id ascending),
        ties broken alphabetically
  7. Log formatted table via log_templates
  8. Return AnalysisResult(success=True, data=StatsResult(...))
```

**Merge contract**:
- `total_files` and `total_annotations`: simple sum across paths
- `per_class`: merge by class_name (sum counts for matching names)
- `categories`: all paths share the same class_file → use class_file definitions
- `format`: all paths must be identical; first path's format is authoritative

### 3.3 `SplitAnalyser` (`split.py`)

Splits a dataset into train and validation subsets. Supports three modes (auto-detected from provided inputs). Only YOLO and LabelMe formats are supported (not COCO).

```python
class SplitAnalyser(BaseAnalyser):
    """Split dataset into train/val subsets.

    Constructor: ``SplitAnalyser(log_config=None)``
    """

    def analyse(
        self,
        output_dir: Path,
        ratio: float = 0.8,
        seed: int = 42,
        label_dir: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        class_file: Optional[Path] = None,
        move: bool = False,
    ) -> AnalysisResult:
        """Split the dataset into train and val subsets.

        At least one of label_dir or image_dir must be provided.

        Args:
            output_dir: Output root directory (train/ and val/ created inside).
            ratio: Proportion of data for training (default 0.8).
            seed: Random seed for reproducible shuffling.
            label_dir: Optional label directory (YOLO or LabelMe).
            image_dir: Optional image directory.
            class_file: Optional classes.txt (copied to output dirs).
            move: When True, move source files instead of copying.

        Returns:
            AnalysisResult with SplitResult in ``.data``
        """
```

**Pipeline** (file-level — no handler for YOLO/LabelMe labels mode):

```
1. Validate: at least one of label_dir/image_dir required
   (both → error)
2. Validate ratio in (0, 1)
3. Determine mode:
   a. label_dir + image_dir → "both"
   b. label_dir only → "labels"
   c. image_dir only → "images"
4. Detect format from label_dir (labels/both mode):
   a. YOLO → .txt extension
   b. LabelMe → .json extension
   c. COCO → error (not supported)
5. Collect items:
   a. labels mode: list *.txt (YOLO) or *.json (LabelMe), sorted by stem
   b. images mode: list image files via _collect_image_files()
   c. both mode: list label files + pre-index image files by stem
6. Shuffle items with random.Random(seed)
7. Split by ratio:
   a. split_idx = int(len(items) * ratio)
   b. If split_idx == 0 and len(items) >= 2 → silently adjust to 1
   c. If split_idx == len(items) and len(items) >= 2 → silently adjust to len(items)-1
   d. train_items = items[:split_idx]
   e. val_items = items[split_idx:]
8. Copy/move files to output_dir/train/ and output_dir/val/:
   a. labels mode: copy label files directly (no annotation parsing)
   b. images mode: copy image files directly
   c. both mode: labels/ and images/ subdirectories under train/ and val/
      Labels drive split; images matched by stem
9. Copy class_file to both output dirs (if provided)
10. Warn on unmatched images (both mode, copy only — move self-resolves)
11. Return AnalysisResult(success=True, data=SplitResult(...))
```

**Split behavior contract**:

| Property | Behavior |
|----------|----------|
| Modes | Three auto-detected: labels-only, images-only, both |
| Formats | YOLO and LabelMe only (COCO → error) |
| Labels-only | Pure file-level — no annotation parsing. List files by extension, copy/move to train/val |
| Images-only | Pure file-level — scan image dir by extension, copy/move to train/val |
| Both | Labels drive split; images matched by file stem. Output: `train/labels/` + `train/images/` (same for val) |
| Shuffle | Always on — `random.Random(seed).shuffle(items)` before splitting (ML train/val split must be randomized) |
| Random seed | `random.Random(seed)` for deterministic shuffling |
| Ratio | `train_count = int(total * ratio)`, remainder to val |
| Minimum size | If ratio results in empty train or val set and dataset has >= 2 items, silently adjust boundaries (min 1 per side) |
| Image copying | Images ARE copied when --image-dir is provided (both mode) or images-only mode |
| Move mode | `--move` relocates files instead of copying (requires confirmation via CLI) |
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

### 3.5 `PartitionAnalyser` (`partition.py`)

Partitions a dataset into N roughly-equal subsets. Supports three modes: labels-only, images-only, or labels+images (labels drive the partition, images follow by stem matching). Supports YOLO and LabelMe formats only — COCO is rejected (single JSON files should use `SplitAnalyser`).

```python
class PartitionAnalyser(BaseAnalyser):
    """Partition dataset into N roughly-equal subsets.

    Constructor: ``PartitionAnalyser(log_config=None)``
    """

    def analyse(
        self,
        output_dir: Path,
        num: int,
        label_dir: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        shuffle: bool = False,
        seed: int = 42,
        class_file: Optional[Path] = None,
        move: bool = False,
    ) -> AnalysisResult:
        """Partition the dataset into ``num`` subsets.

        At least one of ``label_dir`` or ``image_dir`` must be provided.

        Args:
            output_dir: Output root directory. ``part_1/`` through
                ``part_N/`` are created inside.
            num: Number of partitions (>= 2).
            label_dir: Optional label directory (YOLO .txt or LabelMe .json).
            image_dir: Optional image directory.
            shuffle: When True, randomly shuffle before partitioning.
                Default False (sequential partitioning).
            seed: Random seed for ``shuffle=True`` reproducibility.
            class_file: Optional classes.txt for label mode.
            move: When True, move source files instead of copying
                (destructive — CLI requires confirmation).

        Returns:
            AnalysisResult with PartitionResult in ``.data``
        """
```

**Pipeline**:

```
1. Validate inputs:
   a. num >= 2 (else error)
   b. At least one of label_dir / image_dir provided (else error)
2. Determine mode:
   a. label_dir only     → mode = "labels"
   b. image_dir only     → mode = "images"
   c. both specified     → mode = "both"
3. Collect items:
   Labels mode / Both mode:
     a. detect_format(label_dir) → "yolo" | "labelme"
        - If "coco" → ERROR: "partition does not support COCO format"
     b. create_handler(format, label_dir, class_file, image_dir, logger)
     c. handler.read() → DatasetAnnotations
     d. items = list(dataset.images), sorted by image_id
   Images-only mode:
     a. Collect image files from image_dir (glob common extensions:
        .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp), sorted by name
     b. items = list of (image_path, stem) tuples
4. Partition:
   a. If shuffle: random.Random(seed).shuffle(items)
   b. total = len(items)
   c. base_size = total // num, remainder = total % num
   d. First (num - remainder) partitions: base_size items each
      Last remainder partitions: base_size + 1 items each
5. Write outputs per partition i (1..N):
   a. Create part_i/ under output_dir
   b. Labels mode / Both mode:
      - Write labels via handler.write_one() (streaming, YOLO/LabelMe)
      - Both mode: match images by stem, copy/move to part_i/images/
   c. Images-only mode:
      - Copy/move image files to part_i/
   d. If class_file provided: copy to part_i/
6. Log formatted output via log_templates
7. Return AnalysisResult(success=True, data=PartitionResult(...))
```

**Three modes**:

| Mode | Input | Behavior |
|------|-------|----------|
| **labels** | ``label_dir`` only | Partition label files, write per partition |
| **images** | ``image_dir`` only | Partition image files, copy/move per partition |
| **both** | both specified | Labels drive partition; images matched by stem, placed in ``part_i/images/`` alongside ``part_i/labels/`` |

**Both-mode stem matching**: For each label's ``image_id`` (file stem, e.g. ``image001``), search ``image_dir`` for a matching image file (e.g. ``image001.jpg``). Label files without matching images → WARNING, image file skipped. Images without matching labels → WARNING, skipped.

**Partition sizing contract**:

| Property | Behavior |
|----------|----------|
| Algorithm | ``base = total // num``, remainder ``r = total % num``. First ``num-r`` partitions get ``base``, last ``r`` get ``base+1`` |
| Sequential (default) | Items processed in sorted order; first N items → part_1, next N → part_2, etc. |
| Shuffle | ``random.Random(seed).shuffle(items)`` before partitioning |
| Seed | Only meaningful when ``shuffle=True`` |

**Output structure**:

```
# Labels-only mode (labels written directly into part_i/, consistent with split):
OUTPUT_DIR/
  part_1/
    image001.txt, image002.txt, ...
  part_2/
    ...
  part_N/
    ...
  classes.txt          # if --class-file provided, copied to each part_i/

# Images-only mode:
OUTPUT_DIR/
  part_1/
    image001.jpg, image002.jpg, ...
  part_2/
    ...
  part_N/
    ...

# Both mode (labels in labels/, images in images/ for clean self-contained structure):
OUTPUT_DIR/
  part_1/
    images/
      image001.jpg, image002.jpg, ...
    labels/
      image001.txt, image002.txt, ...
    classes.txt
  part_2/
    ...
  part_N/
    ...
```

**Move mode contract**:

| Property | Behavior |
|----------|----------|
| Scope | ``--move`` affects both labels and images (when specified) |
| Method | ``shutil.move()`` per file |
| Confirmation | CLI layer uses ``click.confirm()`` before proceeding; Python API skips confirmation |
| Failure | Single file move failure → WARNING, continues (no partial-state abort) |
| Source dirs | Not deleted — only matched files are moved out |

### 3.6 Constructor Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `log_config` | Optional[LogConfig] | No | None | Logging configuration. If None, default `LogConfig(name="analyse")` is used. |

## 4. Public API

### 4.1 `StatsAnalyser.analyse(label_paths, class_file, image_dir, sort_by, descending, recursive) → AnalysisResult`

Compute dataset statistics for one or more label paths. Auto-detects the annotation format from the first path; all subsequent paths must match. Supports recursive subdirectory traversal.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `label_paths` | List[Path] | Yes | — | One or more paths to labels (directories or files). All must be the same format. |
| `class_file` | Path | No | None | Classes.txt for name mapping and output ordering. When provided, **strict validation** is enforced — any class in the data not present in this file causes an ERROR. |
| `image_dir` | Path | No | None | Image directory (needed for YOLO format to locate image dimensions) |
| `sort_by` | str | No | `"id"` | Sort key: `"id"` (class_id) or `"count"` (annotation count). Ignored when `class_file` is provided. |
| `descending` | bool | No | False | When True, reverse sort direction |
| `recursive` | bool | No | False | When True, recursively find label files in subdirectories (YOLO/LabelMe only). Ignored for COCO. |

### 4.2 `SplitAnalyser.analyse(output_dir, ratio, seed, label_dir, image_dir, class_file, move) → AnalysisResult`

Split dataset into train and validation subsets. At least one of ``label_dir`` or ``image_dir`` must be provided.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `output_dir` | Path | Yes | — | Output root directory. Receives train/ and val/ subdirectories. |
| `ratio` | float | No | 0.8 | Train proportion (0 < ratio < 1) |
| `seed` | int | No | 42 | Random seed for reproducible shuffling |
| `label_dir` | Path | No | None | Label directory (YOLO or LabelMe). At least one of `label_dir` / `image_dir` required. |
| `image_dir` | Path | No | None | Image directory. At least one of `label_dir` / `image_dir` required. |
| `class_file` | Path | No | None | Classes.txt (copied to output directories) |
| `move` | bool | No | False | Move source files instead of copying |

### 4.3 `FilterAnalyser.analyse(label_path, original_class_file, new_class_file, output_dir, image_dir) → AnalysisResult`

Filter annotations by category, keeping only classes listed in `new_class_file`.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `label_path` | Path | Yes | — | Path to labels (directory or file) |
| `original_class_file` | Path | Yes | — | Source classes.txt defining all categories in the source dataset |
| `new_class_file` | Path | Yes | — | Target classes.txt defining which categories to keep and their new order |
| `output_dir` | Path | Yes | — | Output directory |
| `image_dir` | Path | No | None | Image directory (needed for YOLO format to locate image dimensions) |

### 4.4 `PartitionAnalyser.analyse(output_dir, num, label_dir, image_dir, shuffle, seed, class_file, move) → AnalysisResult`

Partition dataset into N roughly-equal subsets. Supports YOLO and LabelMe for label mode; images-only mode has no format restriction. COCO is rejected for label mode.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `output_dir` | Path | Yes | — | Output root directory |
| `num` | int | Yes | — | Number of partitions (>= 2) |
| `label_dir` | Path | No | None | Label directory (YOLO or LabelMe). At least one of `label_dir` / `image_dir` required. |
| `image_dir` | Path | No | None | Image directory. At least one of `label_dir` / `image_dir` required. |
| `shuffle` | bool | No | False | When True, randomly shuffle before partitioning |
| `seed` | int | No | 42 | Random seed for shuffle reproducibility |
| `class_file` | Path | No | None | Classes.txt (label mode only) |
| `move` | bool | No | False | Move source files instead of copying (destructive) |

### 4.5 Module-Level Utilities (`utils.py`)

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
    skip_image_loading: bool = False,
    recursive: bool = False,
) -> BaseAnnotationHandler:
    """Create the appropriate handler for the detected format.

    Args:
        label_path: Path to labels
        format: "yolo" | "labelme" | "coco"
        class_file: Classes.txt path (required for YOLO; auto-generated
            if omitted by scanning .txt files for class IDs)
        image_dir: Image directory for YOLO (auto-detected if omitted)
        logger: Logger to pass to the handler
        skip_image_loading: If True and format is YOLO, skip image file
            I/O (use placeholder dimensions) — for read-only stats
        recursive: If True, handler uses rglob for recursive file
            discovery (YOLO/LabelMe only)

    Returns:
        Configured BaseAnnotationHandler instance with strict_mode=False
    """

def _auto_generate_class_file(label_dir: Path, recursive: bool = False) -> Path:
    """Generate a temporary classes.txt from observed class IDs in .txt files.

    Scans all .txt files (recursively if recursive=True), collects unique
    class IDs, and creates a temporary classes.txt with class_<id> names.
    Uses parse_yolo_class_id() for float-tolerant parsing.
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

**Strict class validation (stats only)**:

When `class_file` is provided to `StatsAnalyser.analyse()`, after counting all annotations, the analyser validates that every class name observed in the data appears in the class file. Classes in the data but NOT in the class file produce an **ERROR** (not a warning). The error message includes:

- The specific label path where the unknown classes were found
- The list of unknown class names

This strict validation ensures data quality — when you define an expected category list, any deviation is a hard failure. When `class_file` is NOT provided, no validation occurs (any class names are accepted).

All paths in a multi-path stats run share the same `class_file`. Validation runs against the merged per-class tally, reporting all unknown names with their source paths.

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

**Single path** (backward-compatible):

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
┌────────────┬────┬───────┐
│ Class      │ ID │ Count │
├────────────┼────┼───────┤
│ cat        │ 0  │    42 │
│ dog        │ 1  │    38 │
│ person     │ 2  │    15 │
├────────────┼────┼───────┤
│ Total (3)  │    │    95 │
└────────────┴────┴───────┘

── Result ──
  Status: ✓ Success

  Log saved to: logs/analyse_stats_20260709_100000.log
══════════════════════════════════════
```

**Multi-path** (two or more source paths):

```
══════════════════════════════════════
Analyse: Dataset Statistics
  Sources:        train/labels/, val/labels/    (2 paths)
  Class file:     classes.txt (80 categories)
  Format:         yolo (auto-detected)

── Path Breakdown ──
  train/labels/    5000 files, 12450 annotations
  val/labels/      1200 files,  2980 annotations

── Summary ──
  Total files:       6200
  Total annotations: 15430
  Categories:        80

── Per-Class ──
┌────────────┬────┬───────┐
│ Class      │ ID │ Count │
├────────────┼────┼───────┤
│ person     │ 0  │  3240 │
│ car        │ 1  │  2891 │
│ ...        │ .. │   ... │
├────────────┼────┼───────┤
│ Total (80) │    │ 15430 │
└────────────┴────┴───────┘

── Result ──
  Status: ✓ Success

  Log saved to: logs/analyse_stats_20260709_100000.log
══════════════════════════════════════
```

The "Path Breakdown" section is only shown when ``len(label_paths) > 1``. When `--recursive` is used, each path is annotated with ``(recursive)`` in the breakdown.

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

- `format_analyse_header(operation, label_paths, format_name, class_file, recursive)` — header block (shows "Source:" for single path, "Sources:" for multi-path)
- `format_stats_result(total_files, total_annotations, per_class, categories)` — stats table with per-class breakdown
- `format_stats_path_breakdown(path_stats)` — per-path file/annotation counts (multi-path only)
- `format_split_result(train_count, val_count, train_dir, val_dir, ratio, seed, mode, move)` — split summary
- `format_filter_result(total_files, total_files_with_annotations, annotations_before, annotations_after, kept_categories, removed_categories, missing_categories, output_dir)` — filter comparison + summary
- `format_partition_result(num_partitions, partition_sizes, partition_dirs, total_files, seed, shuffle, mode, move)` — partition summary with per-partition breakdown
- `format_analyse_result(status, log_path)` — final result block

### 6.5 Partition Output Format

```
══════════════════════════════════════
Analyse: Dataset Partition
  Mode:        both
  Format:      yolo (auto-detected)
  Partitions:  3
  Shuffle:     Yes (seed=42)
  Move:        No

── Partition ──
  Part 1:  20000 files → output/part_1/
  Part 2:  20000 files → output/part_2/
  Part 3:  20002 files → output/part_3/

── Result ──
  Status: ✓ Success

  Log saved to: logs/analyse_partition_20260717_100000.log
══════════════════════════════════════
```

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
| Mixed formats in multi-path | `AnalysisResult(success=False, errors=["All paths must be the same format. Got yolo for ... but ... is labelme"])` |
| Empty dataset (no annotations) | `AnalysisResult(success=True, data=StatsResult(total_files=0, ...))` — not an error |
| Class file not found | `AnalysisResult(success=False, errors=["Class file not found: ..."])` |
| Data class not in class_file (strict) | `AnalysisResult(success=False, errors=["Categories in data not found in class file: {...}. Path: ..."])` |
| `--recursive` finds no matching files | `AnalysisResult(success=False, errors=["No TXT files found in ..."]`) — error raised by handler |
| Split ratio out of range | `AnalysisResult(success=False, errors=["Ratio must be between 0 and 1, got: ..."])` |
| Output directory not writable | `AnalysisResult(success=False, errors=["Cannot create output directory: ..."])` |
| Skipped invalid files (non-strict) | Warning in `warnings` list; analysis continues |
| New class file has no valid entries | `AnalysisResult(success=False, errors=["No valid class names in new class file: ..."])` |
| No matching categories found | `AnalysisResult(success=False, errors=["No matching categories between source and new class file"])` |
| New class file has entries not in source | WARNING per missing class; class recorded in `FilterResult.missing_categories` |
| All annotations filtered out | Not an error — result reports `total_annotations_after=0` with warning |
| Partition num < 2 | `AnalysisResult(success=False, errors=["Number of partitions must be at least 2, got: ..."])` |
| Partition num > total files | Not an error — some partitions will be empty (0 files). Files are distributed across as many partitions as possible. |
| No input specified (both dirs None) | `AnalysisResult(success=False, errors=["At least one of label_dir or image_dir must be provided"])` |
| COCO format for partition | `AnalysisResult(success=False, errors=["partition does not support COCO format. COCO is a single JSON file — use 'analyse split' for train/val split."])` |
| Label-image stem mismatch (both mode) | WARNING per unmatched file; partition continues |
| Move confirmation declined (CLI) | `click.Abort()` — no files touched |
| Move single file fails | WARNING, continues with remaining files |

### 9.3 Non-Strict Mode

All analyser operations run in **non-strict mode by default** (handlers created with `strict_mode=False`):

- Corrupted/invalid annotation files are **skipped** with a warning
- Missing images are **ignored** (files counted from label files, not images)
- The operation continues with valid data
- Skipped counts are reported in the warnings list

## 10. CLI Integration

### 10.1 Command Signatures

```
dataflow-cv analyse stats [OPTIONS] LABEL_PATH [LABEL_PATH ...]
dataflow-cv analyse split [OPTIONS] LABEL_PATH OUTPUT_DIR
dataflow-cv analyse filter [OPTIONS] LABEL_PATH ORIGINAL_CLASS_FILE NEW_CLASS_FILE OUTPUT_DIR
dataflow-cv analyse partition [OPTIONS] OUTPUT_DIR --num N [--label-dir L] [--image-dir I]
```

### 10.2 Options

| Option | Applies To | Type | Default | Description |
|--------|-----------|------|---------|-------------|
| `--verbose` | all three | Flag | False | Enable verbose log output |
| `--log-dir` | all three | Path | `./logs` | Log file output directory |
| `--class-file`, `-c` | stats, split | Path | None | Classes.txt for name mapping. For stats: enables **strict validation** — data classes not in this file cause an error. |
| `--image-dir` | all three | Path | None | Image directory for YOLO (auto-detected if omitted: tries ``labels/images/``, ``dataset/images/``, ``dataset_parent/images/``) |
| `--sort-by` | stats | Choice | `"id"` | `"id"` or `"count"` |
| `--descending/--ascending` | stats | Flag | `--ascending` | Sort direction |
| `--recursive`, `-R` | stats | Flag | False | Recursively traverse subdirectories for label files (YOLO/LabelMe only) |
| `--ratio`, `-r` | split | Float | 0.8 | Train proportion |
| `--seed`, `-s` | split, partition | Int | 42 | Random seed |
| `--num`, `-n` | partition | Int | *(required)* | Number of partitions (>= 2) |
| `--shuffle` | partition | Flag | False | Randomly shuffle before partitioning |
| `--move` | partition | Flag | False | Move source files instead of copying (CLI requires confirmation) |
| `--label-dir`, `-l` | partition | Path | None | Label directory (YOLO or LabelMe). At least one of `--label-dir` / `--image-dir` required. |
| `--image-dir`, `-i` | partition | Path | None | Image directory. At least one of `--label-dir` / `--image-dir` required. |

### 10.3 Stats Command (Multi-Path)

```
dataflow-cv analyse stats [OPTIONS] LABEL_PATH [LABEL_PATH ...]
```

| Argument / Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `LABEL_PATH` (argument, repeatable) | Path | *(at least 1 required)* | Directory (YOLO/LabelMe) or COCO JSON file. All paths must be the same format. |
| `--recursive`, `-R` | Flag | False | Recursively find label files in subdirectories. For YOLO: `rglob("*.txt")`. For LabelMe: `rglob("*.json")` filtered to those with `"shapes"` key. Ignored for COCO. |
| `--class-file`, `-c` | Path | None | Shared classes.txt across all paths. Enables strict validation. |

Plus shared options: `--verbose`, `--log-dir`, `--image-dir`, `--sort-by`, `--descending/--ascending`.

**Multi-path merge behavior**:
- Each path is analyzed independently, then results are merged
- All paths must be the same format (first path sets the expected format)
- `total_files` and `total_annotations` are summed
- `per_class` is merged by class_name (counts summed)
- When `--class-file` is provided, strict validation runs on the merged data

### 10.4 Filter Command

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

### 10.5 Partition Command

```
dataflow-cv analyse partition [OPTIONS] OUTPUT_DIR --num N [--label-dir L] [--image-dir I]
```

| Argument / Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `OUTPUT_DIR` (argument) | Path | *(required)* | Output root directory |
| `--num`, `-n` | Int | *(required)* | Number of partitions (>= 2) |
| `--label-dir`, `-l` | Path | None | Label directory (YOLO or LabelMe). At least one of `--label-dir` / `--image-dir` required. |
| `--image-dir`, `-i` | Path | None | Image directory. At least one of `--label-dir` / `--image-dir` required. |
| `--shuffle` | Flag | False | Randomly shuffle before partitioning |
| `--seed`, `-s` | Int | 42 | Random seed (only meaningful with `--shuffle`) |
| `--class-file`, `-c` | Path | None | Classes.txt (label mode only) |
| `--move` | Flag | False | Move source files instead of copying. Requires CLI confirmation via `click.confirm()`. |
| `--verbose` | Flag | False | Enable verbose log output |
| `--log-dir` | Path | `./logs` | Log file output directory |

**Three modes**:

| Mode | CLI invocation | Behavior |
|------|---------------|----------|
| Labels-only | `partition OUT --num 3 --label-dir labels/` | Partition label files only |
| Images-only | `partition OUT --num 3 --image-dir images/` | Partition image files only |
| Both | `partition OUT --num 3 --label-dir labels/ --image-dir images/` | Labels drive partition; images matched by stem |

**Move mode confirmation flow** (CLI layer only):

```python
if move:
    click.echo(
        f"\nWARNING: --move will permanently relocate source files.\n"
        f"  Source label dir:  {label_dir}\n"
        f"  Source image dir:  {image_dir if image_dir else 'N/A'}\n"
        f"  Target:            {output_dir}/\n"
    )
    if not click.confirm("Continue?", default=False):
        raise click.Abort()
```

Python API callers can pass ``move=True`` directly without interactive confirmation.

**Behavior by format** (label mode):

| Format | Read | Write | Output |
|--------|------|-------|--------|
| YOLO | `iter_images()` streaming | `write_one()` per `.txt` | Individual `.txt` files in each `part_N/` (or `part_N/labels/` in both mode) |
| LabelMe | `iter_images()` streaming | `write_one()` per `.json` | Individual `.json` files in each `part_N/` (or `part_N/labels/` in both mode) |
| COCO | — | — | **Rejected** — error directing to `split` |

## 11. Summary of Public API

| API | Location | Purpose |
|-----|----------|---------|
| `StatsAnalyser(log_config)` | `stats.py` | Dataset statistics computation (single or multi-path, optional recursive) |
| `analyser.analyse(label_paths, class_file, image_dir, sort_by, descending, recursive) → AnalysisResult` | `stats.py` | Run statistics |
| `SplitAnalyser(log_config)` | `split.py` | Train/val dataset splitting (labels, images, or both). YOLO/LabelMe only. |
| `analyser.analyse(output_dir, ratio, seed, label_dir, image_dir, class_file, move) → AnalysisResult` | `split.py` | Run split |
| `FilterAnalyser(log_config)` | `filter.py` | Category-based annotation filtering |
| `analyser.analyse(label_path, original_class_file, new_class_file, output_dir, image_dir) → AnalysisResult` | `filter.py` | Run filter |
| `PartitionAnalyser(log_config)` | `partition.py` | N-way dataset partitioning (labels, images, or both). YOLO/LabelMe only. |
| `analyser.analyse(output_dir, num, label_dir, image_dir, shuffle, seed, class_file, move) → AnalysisResult` | `partition.py` | Run partition |
| `detect_format(label_path) → str` | `utils.py` | Auto-detect annotation format |
| `create_handler(label_path, format, class_file, image_dir, logger, skip_image_loading=False, recursive=False) → BaseAnnotationHandler` | `utils.py` | Handler factory. `recursive=True` enables handler-side rglob file discovery. |
| `load_class_names(class_file) → Dict[int, str]` | `utils.py` | Parse classes.txt |
| `parse_yolo_class_id(token) → Optional[int]` | `label/utils.py` | Float-tolerant YOLO class ID parser — accepts ``"5"`` and ``"5.000000"`` |
| `_auto_generate_class_file(label_dir, recursive=False) → Path` | `utils.py` | Generate temp classes.txt from observed class IDs |
