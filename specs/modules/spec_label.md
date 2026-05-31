# Label Module Specification

> **Version:** 1.0
> **Layer:** Modules
> **Dependencies:** None (foundation module)

## 1. Module Overview

The Label module (`dataflow/label/`) is the **foundation layer** of DataFlow-CV. It provides:

- **Data models** — the universal internal representation for all annotation formats
- **Handlers** — format-specific read/write/validate implementations
- **Utilities** — lossless round-trip verification and category management

### 1.1 Module Contract

The Label module is the **only module** that Convert and Visualize are allowed to depend on. It exposes a stable public API through:

- `DatasetAnnotations` and its component data classes
- `BaseAnnotationHandler` abstract interface
- `AnnotationResult` return type

### 1.2 File Map

```
dataflow/label/
├── models.py              # Core data structures
├── base.py                # Abstract handler + AnnotationResult
├── yolo_handler.py        # YOLO ↔ internal model
├── coco_handler.py        # COCO ↔ internal model
├── labelme_handler.py     # LabelMe ↔ internal model
└── utils.py               # Round-trip verification, category utilities
```

## 2. Data Model (`models.py`)

### 2.1 Universal Coordinate Convention

**All coordinates in the internal model are normalized to [0, 1].**

- `BoundingBox.(x, y)` is the **center** of the box (YOLO convention)
- `BoundingBox.(width, height)` are normalized dimensions
- `Segmentation.points` are normalized (x, y) vertex pairs

This is the **single most important design decision** in the entire codebase. Every handler normalizes on read and denormalizes on write.

### 2.2 `DatasetAnnotations`

The top-level container returned by all handler `read()` calls and consumed by all handler `write()` calls.

```
DatasetAnnotations
├── images: List[ImageAnnotation]     # All images in the dataset
├── categories: Dict[int, str]        # category_id → category_name
└── dataset_info: Dict[str, Any]      # Format-specific metadata
```

**Invariants:**
- `categories` keys are always `int`, values are always `str`
- Attempting to add a duplicate `category_id` with a different name raises `ValueError`
- `num_images` = `len(images)`
- `num_objects` = `sum(len(img.objects) for img in images)`

### 2.3 `ImageAnnotation`

Per-image container:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_id` | str | **Yes** | Unique identifier (typically file stem) |
| `image_path` | str | **Yes** | Path to image file (relative to source directory) |
| `width` | int | **Yes** | Image width in pixels (> 0) |
| `height` | int | **Yes** | Image height in pixels (> 0) |
| `objects` | List[ObjectAnnotation] | No | List of annotated objects |
| `original_data` | Optional[OriginalData] | No | Raw source data for lossless round-trip |

**Invariants:**
- `width > 0` and `height > 0` (validated in `__post_init__`)
- `image_path` is stored as a relative path when the image is under the source directory

### 2.4 `ObjectAnnotation`

Single annotation instance:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `class_id` | int | **Yes** | Category ID |
| `class_name` | str | **Yes** | Category name |
| `bbox` | Optional[BoundingBox] | No | Detection bounding box |
| `segmentation` | Optional[Segmentation] | No | Instance segmentation polygon |
| `confidence` | float | No | Detection confidence (default 1.0) |
| `is_crowd` | bool | No | COCO crowd flag (default False) |
| `original_data` | Optional[OriginalData] | No | Raw source data for lossless round-trip |

**Invariants:**
- At least one of `bbox` or `segmentation` must be non-None (validated in `__post_init__`)

### 2.5 `BoundingBox`

Center-based normalized bounding box:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `x` | float | [0, 1] | Center x (normalized) |
| `y` | float | [0, 1] | Center y (normalized) |
| `width` | float | [0, 1] | Width (normalized) |
| `height` | float | [0, 1] | Height (normalized) |
| `original_data` | Optional[OriginalData] | — | Raw source data |

**Key methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `xyxy(img_w, img_h)` | `(x1, y1, x2, y2)` | Top-left to bottom-right in absolute pixels. **Use this for COCO bbox output.** |
| `xywh_abs(img_w, img_h)` | `(cx, cy, w, h)` | Center-x, center-y in absolute pixels. **Do NOT use for COCO output.** |

**Critical rule**: For COCO bbox output, always use `xyxy()` then convert to `[x1, y1, x2-x1, y2-y1]`. Using `xywh_abs()` directly produces center-based coordinates, which will cause systematically offset bboxes in COCO output.

### 2.6 `Segmentation`

Normalized polygon segmentation:

| Field | Type | Description |
|-------|------|-------------|
| `points` | List[Tuple[float, float]] | Normalized polygon vertices |
| `rle` | Optional[Dict] | Preserved RLE data (COCO only) |
| `original_data` | Optional[OriginalData] | Raw source data |

**Key methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `points_abs(img_w, img_h)` | List[Tuple[int, int]] | Absolute pixel coordinates |
| `has_rle()` | bool | Whether RLE data is present |

### 2.7 `OriginalData`

Lossless round-trip preservation container:

```
OriginalData
├── format: str                  # "yolo" | "coco" | "labelme"
├── raw_data: Dict[str, Any]     # Exact raw bytes/strings/objects from source
└── metadata: Dict[str, Any]     # Additional context (line numbers, dimensions, etc.)
```

**Contract:**
- `format` determines how `raw_data` is interpreted during write-back
- Write handlers check `OriginalData.format == target_format` before using original data
- When original data is used, the write path reproduces the source exactly (lossless)
- When original data is unavailable, the write path converts from the internal model (float precision)

### 2.8 `OriginalDataManager`

Static utility for coordinating original data across conversion chains:

| Method | Purpose |
|--------|---------|
| `should_use_original(obj, target_format)` | Determines if original data path should be used for writing |
| `merge_original_data(existing, new)` | Merges original data from different sources (keeps first match) |
| `extract_original_coordinates(obj, w, h)` | Extracts bbox/segmentation coords from original data for comparison |

### 2.9 `AnnotationFormat` Enum

```python
class AnnotationFormat(Enum):
    LABELME = "labelme"
    YOLO = "yolo"
    COCO = "coco"
    UNKNOWN = "unknown"
```

## 3. Handler Interface (`base.py`)

### 3.1 `AnnotationResult`

Standardized return type for all handler operations:

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Whether the operation succeeded |
| `data` | Optional[Any] | Operation result (DatasetAnnotations for read) |
| `message` | str | Human-readable status message |
| `errors` | List[str] | Accumulated error messages |

**Behavior:**
- `add_error(msg)` sets `success = False` and appends to both `message` and `errors`
- Multiple errors are concatenated with `; ` in `message`

### 3.2 `BaseAnnotationHandler`

Abstract base class — all handlers must implement these three methods:

```python
class BaseAnnotationHandler(ABC):
    def __init__(self, strict_mode: bool = True, logger=None): ...

    @abstractmethod
    def read(self, *args, **kwargs) -> AnnotationResult: ...

    @abstractmethod
    def write(self, annotations: DatasetAnnotations, *args, **kwargs) -> AnnotationResult: ...

    @abstractmethod
    def validate(self, *args, **kwargs) -> bool: ...
```

**Constructor parameters:**
- `strict_mode` (default `True`): Validation errors immediately abort processing
- `logger`: Optional `logging.Logger` instance

**Provided validation utilities:**
- `_validate_normalized_coordinate(value, name)` — checks value ∈ [0, 1]
- `_validate_bbox(bbox)` — validates all 4 bbox coordinates
- `_validate_segmentation_points(points)` — validates polygon has ≥ 3 normalized points
- `_validate_image_dimensions(w, h)` — checks > 0
- `_set_annotation_flags(dataset)` — sets `is_det`/`is_seg` flags based on data

**Logging utilities:**
- `_log_info(msg)` / `_log_error(msg)` / `_log_warning(msg)` / `_log_debug(msg)`
- `_log_error()` raises `ValueError` in strict mode

### 3.3 Handler State Flags

After reading, the handler sets introspection flags:

| Flag | Meaning |
|------|---------|
| `is_det` | Dataset contains detection (bbox) annotations |
| `is_seg` | Dataset contains segmentation (polygon/RLE) annotations |
| `is_rle` | Dataset contains RLE-encoded segmentation (COCO specific) |

Both `is_det` and `is_seg` can be `True` simultaneously (mixed dataset).

## 4. Concrete Handlers

### 4.1 `YoloAnnotationHandler`

**Constructor:** `YoloAnnotationHandler(label_dir, class_file, image_dir, **kwargs)`

| Parameter | Required | Description |
|-----------|----------|-------------|
| `label_dir` | Yes | Directory containing `.txt` label files |
| `class_file` | Yes | Path to `classes.txt` |
| `image_dir` | Yes | Directory containing image files |

**`read()`**: Scans `label_dir` for `.txt` files, matches to images by stem, normalizes all coordinates to [0, 1] center-based internal model.

**`write(annotations, output_dir)`**: Writes one `.txt` per image to `output_dir`.
- Priority 1: Use `OriginalData` if format matches YOLO (lossless)
- Priority 2: Convert from internal model (6 decimal places)
- Empty images produce empty `.txt` files

**`validate(annotation_file)`**: Validates a single `.txt` file structure.

**Format detection**: Per-line, based on token count (5 = detection, odd > 5 = segmentation).

### 4.2 `CocoAnnotationHandler`

**Constructor:** `CocoAnnotationHandler(annotation_file, do_rle=False, **kwargs)`

| Parameter | Required | Description |
|-----------|----------|-------------|
| `annotation_file` | Yes | Path to COCO `.json` file |
| `do_rle` | No | Whether to output RLE format (default False) |

**`read()`**: Reads the single JSON file, normalizes bbox to center-based [0,1], preserves RLE data.

**`write(annotations, output_file, output_rle=None)`**: Writes COCO JSON with:
- Priority 1: Use `OriginalData` if format matches COCO (lossless, preserves all optional fields)
- Priority 2: Convert from internal model
- Crowd annotations (`iscrowd=1`) always use RLE format
- Non-crowd annotations follow `output_rle` flag

**RLE handling:**
- Write: `counts_bytes.decode("latin1")` — mandatory; UTF-8 cannot represent all 256 byte values
- Read: `counts_str.encode("latin1")` → `mask.decode()`
- `HAS_COCO_MASK` flag indicates pycocotools availability

### 4.3 `LabelMeAnnotationHandler`

**Constructor:** `LabelMeAnnotationHandler(label_dir, class_file=None, **kwargs)`

| Parameter | Required | Description |
|-----------|----------|-------------|
| `label_dir` | Yes | Directory containing `.json` label files |
| `class_file` | No | Optional path to `classes.txt` |

**`read()`**: Scans `label_dir` for `.json` files, parses shapes:
- `rectangle` (2 points) → `BoundingBox` (center, normalized)
- `polygon` (≥ 3 points) → `Segmentation` (normalized)
- Other shape types (circle, line, point) → **parse error**

**`write(annotations, output_dir)`**: Writes one `.json` per image.
- Priority 1: Use `OriginalData` if format matches LabelMe (preserves `flags`, `group_id`, etc.)
- Priority 2: Convert from internal model (default structure)
- `imageData` is always set to `None`

**`validate(annotation_file)`**: Validates a single `.json` file structure.

**Category discovery**: If `class_file` is not provided, categories are auto-extracted from `shape.label` values during reading.

## 5. Strict Mode vs Non-Strict Mode Contract

| Error Type | Strict Mode | Non-Strict Mode |
|------------|-------------|-----------------|
| Invalid annotation format (wrong token count, bad JSON) | Abort immediately | Skip annotation, log warning, continue |
| Invalid class_id (not in categories) | Abort immediately | Skip annotation, log warning, continue |
| Invalid coordinate (out of [0,1]) | Abort immediately | Skip annotation, log warning, continue |
| Invalid bbox (zero area, overflow) | Abort immediately | Skip annotation, log warning, continue |
| Image file not found | **Always skip with warning** | **Always skip with warning** |
| Image unreadable / corrupt | **Always skip with warning** | **Always skip with warning** |
| Invalid image dimensions | **Always skip with warning** | **Always skip with warning** |

**Key rule**: Image-related errors never cause an abort — they are always downgraded to warnings regardless of `strict_mode`. This is because image availability is orthogonal to annotation validity.

## 6. Lossless Round-Trip Contract

### 6.1 Priority Chain

During `write()`, each handler follows this priority for every annotation:

```
1. OriginalData (format matches target) → Byte-for-byte reproduction
2. Internal model conversion           → Float precision (.6f or JSON float)
```

### 6.2 What Is Preserved Per Format

| Format | Preserved in OriginalData |
|--------|--------------------------|
| YOLO | Exact line text, tokenized items with numeric types, line number |
| COCO | Full annotation dict (all fields including custom ones), full image dict, full category dict |
| LabelMe | Full shape dict (including `flags`, `group_id`), full per-image JSON dict |

### 6.3 When OriginalData Is NOT Used

Original data is bypassed when:
1. `OriginalData.format != target_format` (not a round-trip to the same format)
2. `obj.has_original_data()` returns `False` (data was created from internal model)
3. The original data path is intentionally bypassed (e.g., `output_rle != source_rle`)
4. Class mapping has changed and class_id needs updating (partial update: original items with new class_id)

## 7. Utility Functions (`utils.py`)

| Function | Purpose |
|----------|---------|
| `verify_lossless_roundtrip(handler, annotations)` | Validates that read→write→read produces identical data |
| `extract_categories_from_coco_data(coco_data)` | Extracts `Dict[id, name]` from COCO categories |
| `generate_classes_file(categories, output_path)` | Generates `classes.txt` from category dict |
| `load_classes_file(file_path)` | Loads `classes.txt` into `Dict[int, str]` |

## 8. Validation Constraints (Complete Checklist)

A valid read operation must satisfy ALL of:

1. Source directory/file exists and is readable
2. Source files match the expected structure (`.txt` for YOLO, `.json` for LabelMe/COCO)
3. At least one annotation file is found
4. Categories are loaded (from class_file or auto-extracted)
5. Each annotation line/shape has a valid class_id
6. All coordinates are finite floats in [0, 1] (after normalization for absolute-pixel formats)
7. Detection annotations have valid bbox (w > 0, h > 0, no boundary overflow)
8. Segmentation annotations have at least 3 polygon vertices
9. COCO RLE annotations have valid `size` and `counts` fields
10. Image dimensions are positive integers
