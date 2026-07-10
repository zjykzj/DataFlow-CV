# Label Module Specification

> **Version:** v4.4 | **Last Updated:** 2026-07-02
> **Status:** Draft — logger receives from caller's LogManager via `logger=` parameter
> **Layer:** Modules
> **Dependencies:** None (foundation module; logging via stdlib `logging.Logger` only)

## 1. Module Overview

The Label module (`dataflow/label/`) is the **foundation layer** of DataFlow-CV. It provides:

- **Data models** — format-aware containers that store coordinates in their **native representation**
- **Handlers** — format-specific read/write/validate implementations
- **Utilities** — category management and format comparison

### 1.1 Design Principle: Native Coordinate Storage

**Each format handler stores coordinates in its format's native coordinate system.
There is no universal normalized internal model.**

This eliminates unnecessary coordinate round-trips (normalize → denormalize) and the
associated ±1 pixel precision loss that plagued the v1 architecture. Coordinate
transformation is the exclusive responsibility of the **Convert module**, not handlers.

| Format  | Bounding Box Semantics                              | Coordinate Space       |
|---------|-----------------------------------------------------|------------------------|
| YOLO    | `(cx, cy, w, h)` center-based                       | Normalized [0, 1]      |
| LabelMe | `(x1, y1, x2, y2)` two opposite corners             | Absolute pixels        |
| COCO    | `(x, y, w, h)` top-left origin                      | Absolute pixels        |

### 1.2 Module Contract

The Label module is the **only module** that Convert and Visualize are allowed to depend on.
It exposes a stable public API through:

- `DatasetAnnotations` with `format` field identifying coordinate semantics
- `BaseAnnotationHandler` abstract interface (read/write/validate/iter_images)
- `AnnotationResult` return type
- `ImageError` exception — raised for image-related errors (missing files, unreadable images, invalid dimensions). In strict mode causes abort; in non-strict mode causes skip. YOLO non-strict mode: uses placeholder dimensions (1,1) instead of raising
- `iter_images()` streaming iterator — incremental per-image yield for memory-efficient processing

## 2. Data Model (`models.py`)

### 2.1 Core Container: `DatasetAnnotations`

The top-level container returned by all handler `read()` calls. **The `format` field
defines how all coordinate fields should be interpreted.**

```
DatasetAnnotations
├── format: AnnotationFormat       # SOURCE format — defines coordinate semantics
├── images: List[ImageAnnotation]  # All images in the dataset
├── categories: Dict[int, str]     # category_id → category_name
└── dataset_info: Dict[str, Any]   # Format-specific metadata (raw JSON preserved)
```

**Invariants:**
- `categories` keys are always `int`, values are always `str`
- `num_images` = `len(images)`
- `num_objects` = `sum(len(img.objects) for img in images)`
- `format` must be set to the actual source format (never `UNKNOWN` after a successful read)

### 2.2 `ImageAnnotation`

Per-image container:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_id` | str | **Yes** | Unique identifier (typically file stem) |
| `image_path` | str | **Yes** | Path to image file (relative to source directory) |
| `width` | int | **Yes** | Image width in pixels (> 0) |
| `height` | int | **Yes** | Image height in pixels (> 0) |
| `objects` | List[ObjectAnnotation] | No | List of annotated objects |

**Invariants:**
- `width > 0` and `height > 0` (validated in `__post_init__`)
- `image_path` is stored as a relative path when the image is under the source directory

### 2.3 `ObjectAnnotation`

Single annotation instance:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `class_id` | int | **Yes** | Category ID |
| `class_name` | str | **Yes** | Category name |
| `bbox` | Optional[BoundingBox] | No | Detection bounding box (native coords) |
| `segmentation` | Optional[Segmentation] | No | Instance segmentation polygon (native coords) |
| `confidence` | float | No | Detection confidence (default 1.0) |
| `is_crowd` | bool | No | COCO crowd flag (default False) |

**Invariants:**
- At least one of `bbox` or `segmentation` must be non-None (validated in `__post_init__`)

### 2.4 `BoundingBox`

A 4-value bounding box **in native format coordinates**. Interpretation depends on
`DatasetAnnotations.format`:

```
format=YOLO:    (x, y, w, h) = center (normalized [0,1]), dimensions (normalized [0,1])
format=COCO:    (x, y, w, h) = top-left (absolute pixels), width/height (absolute pixels)
format=LABELME: (x, y, w, h) = top-left (absolute pixels), width/height (absolute pixels)
                (derived from two corner points x1,y1,x2,y2 → x=min(x1,x2), y=min(y1,y2),
                 w=|x2-x1|, h=|y2-y1|)
```

| Field | Type | Description |
|-------|------|-------------|
| `x` | float | X value in native coordinate space |
| `y` | float | Y value in native coordinate space |
| `width` | float | Width in native coordinate space |
| `height` | float | Height in native coordinate space |

**No conversion methods.** Use converters from the Convert module for coordinate
transformations.

**Validation:**
- YOLO format: `x`, `y`, `width`, `height` ∈ [0, 1]; `width > 0`, `height > 0`
- COCO/LabelMe format: `x`, `y` > 0; `width` > 0, `height` > 0; values are absolute pixels

### 2.5 `Segmentation`

Polygon segmentation points **in native format coordinates**:

| Field | Type | Description |
|-------|------|-------------|
| `points` | List[Tuple[float, float]] | Polygon vertices in native coords |
| `rle` | Optional[Dict] | Preserved RLE data (COCO only) |

```
format=YOLO:       points are (x, y) normalized [0,1]
format=COCO:       points are (x, y) absolute pixels (from [[x1,y1,x2,y2,...], ...] flattened)
format=LABELME:    points are (x, y) absolute pixels
```

**Key methods:**
- `has_rle()` — whether RLE data is present (COCO specific)

**No conversion methods.** Use converters from the Convert module for coordinate
transformations.

### 2.6 `AnnotationFormat` Enum

```python
class AnnotationFormat(Enum):
    YOLO = "yolo"       # Normalized [0,1] center-based coordinates
    COCO = "coco"       # Absolute pixel top-left coordinates
    LABELME = "labelme" # Absolute pixel corner coordinates
    UNKNOWN = "unknown"
```

## 3. Handler Interface (`base.py`)

### 3.1 `AnnotationResult`

Standardized return type for all handler operations:

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Whether the operation succeeded |
| `data` | Optional[Any] | Operation result (`DatasetAnnotations` for read) |
| `message` | str | Human-readable status message |
| `errors` | List[str] | Accumulated error messages |

### 3.2 `BaseAnnotationHandler`

Abstract base class — all handlers must implement these five methods:

```python
class BaseAnnotationHandler(ABC):
    def __init__(self, strict_mode: bool = True, logger=None): ...

    @abstractmethod
    def read(self, *args, **kwargs) -> AnnotationResult: ...

    @abstractmethod
    def iter_images(self) -> Iterator[ImageAnnotation]: ...

    @abstractmethod
    def write(self, annotations: DatasetAnnotations, *args, **kwargs) -> AnnotationResult: ...

    @abstractmethod
    def write_one(self, image_ann: ImageAnnotation, output_dir: Path) -> AnnotationResult: ...

    @abstractmethod
    def validate(self, *args, **kwargs) -> bool: ...
```

#### `read()` — Batch Load

Returns all annotations as a complete `DatasetAnnotations`. Suitable for workflows that need
the entire dataset in memory (conversion, evaluation).

#### `iter_images()` — Streaming Iterator

Yields `ImageAnnotation` objects **one at a time** as they are parsed. This is the streaming
alternative to `read()` — callers process images incrementally without waiting for the entire
dataset to load.

**Signature:**

```python
@abstractmethod
def iter_images(self) -> Iterator[ImageAnnotation]:
    """Yield ImageAnnotation objects one at a time.

    Validates directories and categories upfront (raises immediately if invalid).
    Then scans annotation files and yields each successfully parsed image.

    Strict mode (default):
        Raises ValueError on the first invalid file or annotation line.
        The iterator stops — partial results before the error are available.

    Non-strict mode:
        Skips invalid files/lines, logs warnings, continues yielding valid images.

    Yields:
        ImageAnnotation with format-native coordinates, one per image file.

    Raises:
        ValueError: In strict mode, when parsing fails for any file or line.
    """
```

**Key contract:**
- Each yielded `ImageAnnotation` contains **format-native coordinates** (same as `read()`).
- Callers must check `DatasetAnnotations.format` (from handler metadata) to interpret
  coordinate semantics correctly.
- Image errors (missing file, unreadable) always skip regardless of strict_mode,
  consistent with the image-error downgrade rule (Section 5).
- Categories are validated upfront — if no categories are loaded, iteration raises
  immediately (no images are yielded).

**Constructor parameters:**
- `strict_mode` (default `True`): Validation errors immediately abort processing
- `logger`: Optional `logging.Logger` instance — received from the calling module's `LogManager` (e.g., `log_manager.logger` or `log_manager.child("handler")`). See [`spec_logging.md`](spec_logging.md).

**Provided validation utilities:**

| Method | Description |
|--------|-------------|
| `_validate_image_dimensions(w, h)` | Checks > 0 (format-independent) |
| `_validate_bbox(bbox, format)` | Format-aware bbox validation |
| `_validate_segmentation_points(points, format)` | Format-aware polygon validation |
| `_set_annotation_flags(dataset)` | Sets `is_det`/`is_seg` flags based on data |

**Format-specific validation in `_validate_bbox`:**
- `YOLO`: All values ∈ [0, 1]; width > 0, height > 0; center (x,y) within [0,1]
- `COCO/LabelMe`: All values > 0; width > 0, height > 0; finite real numbers

**Format-specific validation in `_validate_segmentation_points`:**
- `YOLO`: ≥ 3 points; all coordinates ∈ [0, 1]
- `COCO/LabelMe`: ≥ 3 points; all coordinates > 0; finite real numbers

#### `write_one()` — Single-Image Write (Streaming)

Writes annotation data for a **single image** to the target format. This is the per-image
counterpart to `write()` — used by the Convert module's streaming pipeline
(`stream_convert()`) to write each image immediately after conversion.

**Signature:**

```python
@abstractmethod
def write_one(
    self, image_ann: ImageAnnotation, output_dir: Path
) -> AnnotationResult:
    """Write annotations for a single image.

    Args:
        image_ann: Single ImageAnnotation with target-native coordinates.
        output_dir: Directory to write the output file into.

    Returns:
        AnnotationResult with success status.
    """
```

**Key contract:**
- Receives a single `ImageAnnotation` — coordinates are already in the **target format's
  native space** (conversion happened upstream in the converter)
- Writes exactly one output file (e.g., one `.txt` for YOLO, one `.json` for LabelMe)
- Output filename is derived from `image_ann.image_id`
- Does NOT accumulate data or maintain cross-image state — each call is independent
- Empty images (no objects) produce valid empty output files
- **Not applicable to single-file targets**: COCO handler MUST raise `NotImplementedError`
  because COCO is always written as a single JSON via `write()`

**Concrete implementations:**

| Handler | Output per call | File naming |
|---------|----------------|-------------|
| `YoloAnnotationHandler` | One `.txt` per image | `{image_id}.txt` |
| `LabelMeAnnotationHandler` | One `.json` per image | `{image_id}.json` |
| `CocoAnnotationHandler` | N/A — raises `NotImplementedError` | — |

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

**Constructor:** `YoloAnnotationHandler(label_dir, class_file, image_dir, prediction=False, **kwargs)`

| Parameter | Required | Description |
|-----------|----------|-------------|
| `label_dir` | Yes | Directory containing `.txt` label or prediction files |
| `class_file` | Yes | Path to `classes.txt` |
| `image_dir` | Conditional | Directory containing image files. Required in strict mode. In non-strict mode, missing directory or missing images are tolerated — labels are still parsed with placeholder dimensions (1, 1). |
| `prediction` | No | If True, parse prediction format (with confidence). Default False (label format). |

**`read()`**: Returns `DatasetAnnotations(format=YOLO)` with:
- `BoundingBox`: `(cx, cy, w, h)` all normalized [0,1], center-based (native YOLO format).
  Bbox edges are **clamped to [0, 1]** before validation — a WARNING is emitted if clamping
  changes any coordinate value.
- `Segmentation.points`: `(x, y)` normalized [0,1]. Each point is similarly clamped
  to [0, 1] with a WARNING if modified.
- `ObjectAnnotation.confidence`: Set from the last token when `prediction=True`; defaults to `1.0` in label mode
- No coordinate transformation — coordinates are stored as-is from YOLO text files
  (after clamping)

**`iter_images()`**: Yields `ImageAnnotation` objects one at a time with format-native coordinates (same per-line parsing logic as `read()`, including [0, 1] clamping). In strict mode, raises `ValueError` on first parse error. In non-strict mode, logs warnings and skips invalid files/lines. Image errors always skip regardless of strict_mode.

**Key difference from `read()`**: Does not accumulate all images into a `DatasetAnnotations`.
Callers that need `DatasetAnnotations` (with format flags, full category dict) should use
`read()` instead. Callers that only need per-image data (visualization) should prefer
`iter_images()` for lower memory and faster first-image latency.

**`write(annotations, output_dir)`**: Writes one `.txt` per image.
- Expects `DatasetAnnotations.format == YOLO`
- Output: `class_id cx cy w h` (5 tokens for detection) or `class_id x1 y1 x2 y2 ...` (segmentation)
- **Prediction write**: When `ObjectAnnotation.confidence < 1.0`, writes 6 tokens (detection) or appends confidence (segmentation)
- Coordinates written with 6 decimal places (`.6f`); confidence written with 6 decimal places
- Empty images produce empty `.txt` files

**`validate(annotation_file)`**: Validates a single `.txt` file structure.

**Format detection** (per-line, based on token count and `prediction` flag):

| Mode | Detection | Segmentation | Invalid |
|------|-----------|-------------|---------|
| Label (`prediction=False`) | `len == 5` | `len > 5 AND len % 2 == 1` | 6, 8, 10, ... |
| Prediction (`prediction=True`) | `len == 6` | `len > 6 AND len % 2 == 0` | 5, 7, 9, ... |

**Strict mode**: In prediction mode, confidence values outside [0, 1] raise errors. In label mode, any line not matching the label format raises errors.

### 4.2 `CocoAnnotationHandler`

**Constructor:** `CocoAnnotationHandler(annotation_file, do_rle=False, prediction=False, **kwargs)`

| Parameter | Required | Description |
|-----------|----------|-------------|
| `annotation_file` | Yes | Path to COCO `.json` file |
| `do_rle` | No | Whether to output RLE format (default False) |
| `prediction` | No | When True, `write()` outputs plain JSON list of annotation dicts (Variant B, prediction format). Default False (full COCO dict, Variant A). |

**`read()`**: Returns `DatasetAnnotations(format=COCO)` with:
- `BoundingBox`: `(x_tl, y_tl, w_abs, h_abs)` in absolute pixels (native COCO format).
  Coordinates are **clamped to image boundaries** `[0, width] × [0, height]` before
  validation — a WARNING is emitted if clamping changes any coordinate value.
- `Segmentation.points`: `(x, y)` in absolute pixels (from COCO segmentation polygons).
  Each point is similarly clamped to image boundaries.
- `Segmentation.rle`: Preserved RLE data when segmentation is RLE-encoded
- No coordinate normalization — COCO native coordinates are absolute pixels

**`iter_images()`**: Yields `ImageAnnotation` objects one at a time with format-native coordinates. Groups annotations by `image_id` from the COCO JSON, yields each image with its annotations. In strict mode, raises `ValueError` on first invalid annotation. In non-strict mode, logs warnings and skips invalid annotations. Image errors always skip regardless of strict_mode.

**`write(annotations, output_file, output_rle=None)`**: Writes COCO JSON.
- Expects `DatasetAnnotations.format == COCO`
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

**`read()`**: Returns `DatasetAnnotations(format=LABELME)` with:
- `BoundingBox`: `(x_tl, y_tl, w_abs, h_abs)` derived from rectangle corner points
  (`x = min(x1,x2)`, `y = min(y1,y2)`, `w = |x2-x1|`, `h = |y2-y1|`), in absolute pixels.
  Coordinates are **clamped to image boundaries** `[0, width] × [0, height]` before
  validation — a WARNING is emitted if clamping changes any coordinate value.
- `Segmentation.points`: `(x, y)` in absolute pixels from polygon shapes. Each point is
  similarly clamped to image boundaries with a WARNING if modified.
- Shape types: `rectangle` → BoundingBox; `polygon` (≥ 3 points) → Segmentation
- Other shape types (circle, line, point) → parse error

**`iter_images()`**: Yields `ImageAnnotation` objects one at a time with format-native coordinates. Parses each `.json` file's shapes (rectangle → BoundingBox, polygon → Segmentation). In strict mode, raises `ValueError` on first parse error. In non-strict mode, logs warnings and skips invalid files/shapes. If `imagePath` is inaccessible but JSON contains valid `imageWidth`/`imageHeight`, the image is not needed — read succeeds silently. Categories are auto-extracted from `shape.label` if `class_file` not provided.

**`write(annotations, output_dir)`**: Writes one `.json` per image.
- Expects `DatasetAnnotations.format == LABELME`
- `imageData` is always set to `None`
- For rectangle annotations: writes as `shape_type="rectangle"` with two corner points `[[x1,y1],[x2,y2]]`

**`validate(annotation_file)`**: Validates a single `.json` file structure.

**Category discovery**: If `class_file` is not provided, categories are auto-extracted
from `shape.label` values during reading.

## 5. Strict Mode vs Non-Strict Mode Contract

### 5.1 Batch Read (`read()`)

| Error Type | Strict Mode | Non-Strict Mode |
|------------|-------------|-----------------|
| Invalid annotation format (wrong token count, bad JSON) | Abort immediately | Skip annotation, log warning, continue |
| Invalid class_id (not in categories) | Abort immediately | Skip annotation, log warning, continue |
| Invalid coordinate — YOLO (not finite) | Abort immediately | Skip annotation, log warning, continue |
| Invalid coordinate — YOLO (outside [0,1]) | **Clamp to [0, 1] + WARNING**, then validate. Abort only if clamp produces zero-area bbox | Same as strict (clamping is independent of strict_mode) |
| Invalid coordinate — COCO/LabelMe (outside image boundary) | **Clamp to `[0, width] × [0, height]` + WARNING**, then validate. Abort only if clamp produces zero-area bbox | Same as strict (clamping is independent of strict_mode) |
| Invalid bbox (zero area, NaN, overflow) | Abort immediately | Skip annotation, log warning, continue |
| Image file not found (YOLO) | Raise `ImageError` → skip with warning | Use placeholder dims (1,1), log debug, continue |
| Image file not found (COCO) | **Always skip with warning** | **Always skip with warning** |
| Image file not found (LabelMe) | JSON has `imageWidth`/`imageHeight` → read succeeds silently. JSON lacks dimensions → abort immediately | JSON has `imageWidth`/`imageHeight` → read succeeds silently. JSON lacks dimensions → skip with warning |
| Image unreadable / corrupt | **Always skip with warning** | **Always skip with warning** |
| Invalid image dimensions (YOLO non-strict) | Raise `ImageError` → skip with warning | Use placeholder dims (1,1), log debug, continue |
| Invalid image dimensions (other) | **Always skip with warning** | **Always skip with warning** |

### 5.2 Streaming Read (`iter_images()`)

| Error Type | Strict Mode | Non-Strict Mode |
|------------|-------------|-----------------|
| Label/image directory not found (YOLO) | Raise `ValueError` immediately | Log debug, continue (labels parsed with placeholder dims) |
| Label/image directory not found (COCO/LabelMe) | Raise `ValueError` immediately | Raise `ValueError` immediately |
| No categories loaded | Raise `ValueError` immediately (no images yielded) | Raise `ValueError` immediately (no images yielded) |
| No annotation files found | Raise `ValueError` immediately (no images yielded) | Raise `ValueError` immediately (no images yielded) |
| Invalid annotation format (per-file) | Raise `ValueError` immediately, stop iteration | Skip file, log warning, continue to next |
| Invalid coordinate — YOLO (not finite) | Raise `ValueError` immediately, stop iteration | Skip line, log warning, continue |
| Invalid coordinate — YOLO (outside [0,1]) | **Clamp to [0, 1] + WARNING**, then validate. Stop iteration only if clamp produces zero-area bbox | Same as strict (clamping is independent of strict_mode) |
| Invalid coordinate — COCO/LabelMe (outside image boundary) | **Clamp to `[0, width] × [0, height]` + WARNING**, then validate. Stop iteration only if clamp produces zero-area bbox | Same as strict (clamping is independent of strict_mode) |
| Invalid class_id / bbox (other) | Raise `ValueError` immediately, stop iteration | Skip line, log warning, continue |
| Image file not found (YOLO) | Raise `ImageError` → skip with warning, stop iteration | Use placeholder dims (1,1), log debug, continue yielding |
| Image file not found (COCO) | **Always skip with warning** | **Always skip with warning** |
| Image file not found (LabelMe) | JSON has `imageWidth`/`imageHeight` → yield succeeds silently. JSON lacks dimensions → raise `ValueError`, stop iteration | JSON has `imageWidth`/`imageHeight` → yield succeeds silently. JSON lacks dimensions → skip with warning |
| Image unreadable / corrupt | **Always skip with warning** | **Always skip with warning** |
| Invalid image dimensions | **Always skip with warning** | **Always skip with warning** |

**Key rule**: Image-related errors never cause an abort — they are always downgraded to
warnings regardless of `strict_mode`. **Exception — LabelMe**: When the JSON file contains
valid `imageWidth`/`imageHeight`, the image file itself is not required. Absence of the image
in this case is not an error at all and produces no warning.

**Upfront validation**: Both `read()` and `iter_images()` perform upfront validation of
directory existence and category availability. These structural errors always raise
immediately (cannot yield any valid data without them).

**Streaming strict-mode behavior**: When a parse error occurs in strict mode, the iterator
raises `ValueError`. Images that were already yielded before the error are valid — the
caller has already processed them. This is different from `read()` which returns nothing
on error (all-or-nothing). Callers should handle this by wrapping iteration in try/except
and discarding partial results if needed.

**Coordinate validity depends on format:**
- **YOLO**: coordinates are in normalized [0, 1]. Before validation, bbox edges and
  polygon points are **clamped to [0, 1]**. This tolerates minor floating-point
  imprecision at image edges (e.g., `cx + w/2 = 1.00000015` → `1.0`). A WARNING is
  emitted only when clamping modifies a value by more than `1e-6` (one unit in
  YOLO's `.6f` output precision), suppressing noise from harmless string↔float
  round-trips and FP comparison edge cases at the 5e-7 level.
  Clamping is applied regardless of `strict_mode` — it is data normalization, not
  error handling. After clamping, if the bbox has zero area or non-finite values,
  it is rejected as an invalid bbox.
- **COCO / LabelMe**: coordinates are in absolute pixels. Before validation, they are
  **clamped to image boundaries** `[0, width] × [0, height]`. This tolerates minor
  floating-point imprecision at image edges (e.g., `x = -0.39` → `0`). A WARNING is
  emitted when clamping modifies a value by more than `1e-9` pixels. Clamping is
  applied regardless of `strict_mode` — it is data normalization, not error handling.
  After clamping, if the bbox has zero area or non-finite values, it is rejected as
  an invalid bbox.

## 6. Format-Aware Validation Constraints

A valid read operation must satisfy ALL of:

1. Source directory/file exists and is readable
2. Source files match the expected structure (`.txt` for YOLO, `.json` for LabelMe/COCO)
3. At least one annotation file is found
4. Categories are loaded (from class_file or auto-extracted)
5. Each annotation line/shape has a valid class_id
6. All coordinates are finite floats in the **format-native range**:
   - YOLO: [0, 1] (coordinates outside [0, 1] are **clamped** before validation — see §5)
   - COCO/LabelMe: absolute pixels (coordinates outside image boundaries are
     **clamped** before validation — see §5 coordinate validity note)
7. Detection annotations have valid bbox (w > 0, h > 0, finite)
8. Segmentation annotations have at least 3 polygon vertices
9. COCO RLE annotations have valid `size` and `counts` fields
10. Image dimensions are positive integers

## 7. Utility Functions (`utils.py`)

| Function | Purpose |
|----------|---------|
| `extract_categories_from_coco_data(coco_data)` | Extracts `Dict[id, name]` from COCO categories |
| `generate_classes_file(categories, output_path)` | Generates `classes.txt` from category dict |
| `load_classes_file(file_path)` | Loads `classes.txt` into `Dict[int, str]` |
| `calculate_file_hash(file_path, algorithm="md5")` | Computes hash of a file for integrity comparison |
| `compare_annotation_dirs(dir_a, dir_b, format)` | Format-aware comparison (text diff for YOLO, JSON diff for LabelMe) |

