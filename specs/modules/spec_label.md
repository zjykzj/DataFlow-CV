# Label Module Specification

> **Version:** 2.0
> **Status:** Draft — major redesign removing unified normalized coordinate model
> **Layer:** Modules
> **Dependencies:** None (foundation module)

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
| COCO    | `(x, y, w, h)` top-left origin                      | Absolute pixels        |
| LabelMe | `(x1, y1, x2, y2)` two opposite corners             | Absolute pixels        |

### 1.2 Module Contract

The Label module is the **only module** that Convert and Visualize are allowed to depend on.
It exposes a stable public API through:

- `DatasetAnnotations` with `format` field identifying coordinate semantics
- `BaseAnnotationHandler` abstract interface (read/write/validate)
- `AnnotationResult` return type

### 1.3 File Map

```
dataflow/label/
├── models.py              # Format-aware data structures (no unified normalization)
├── base.py                # Abstract handler + AnnotationResult + format-specific validation
├── yolo_handler.py        # YOLO ↔ native representation
├── coco_handler.py        # COCO ↔ native representation
├── labelme_handler.py     # LabelMe ↔ native representation
└── utils.py               # Category utilities, format comparison
```

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

**`read()`**: Returns `DatasetAnnotations(format=YOLO)` with:
- `BoundingBox`: `(cx, cy, w, h)` all normalized [0,1], center-based (native YOLO format)
- `Segmentation.points`: `(x, y)` normalized [0,1]
- No coordinate transformation — coordinates are stored as-is from YOLO text files

**`write(annotations, output_dir)`**: Writes one `.txt` per image.
- Expects `DatasetAnnotations.format == YOLO`
- Output: `class_id cx cy w h` (5 tokens for detection) or `class_id x1 y1 x2 y2 ...` (segmentation)
- Coordinates written with 6 decimal places (`.6f`)
- Empty images produce empty `.txt` files

**`validate(annotation_file)`**: Validates a single `.txt` file structure.

**Format detection**: Per-line, based on token count (5 = detection, odd > 5 = segmentation).

### 4.2 `CocoAnnotationHandler`

**Constructor:** `CocoAnnotationHandler(annotation_file, do_rle=False, **kwargs)`

| Parameter | Required | Description |
|-----------|----------|-------------|
| `annotation_file` | Yes | Path to COCO `.json` file |
| `do_rle` | No | Whether to output RLE format (default False) |

**`read()`**: Returns `DatasetAnnotations(format=COCO)` with:
- `BoundingBox`: `(x_tl, y_tl, w_abs, h_abs)` in absolute pixels (native COCO format)
- `Segmentation.points`: `(x, y)` in absolute pixels (from COCO segmentation polygons)
- `Segmentation.rle`: Preserved RLE data when segmentation is RLE-encoded
- No coordinate normalization — COCO native coordinates are absolute pixels

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
  (`x = min(x1,x2)`, `y = min(y1,y2)`, `w = |x2-x1|`, `h = |y2-y1|`), in absolute pixels
- `Segmentation.points`: `(x, y)` in absolute pixels from polygon shapes
- Shape types: `rectangle` → BoundingBox; `polygon` (≥ 3 points) → Segmentation
- Other shape types (circle, line, point) → parse error

**`write(annotations, output_dir)`**: Writes one `.json` per image.
- Expects `DatasetAnnotations.format == LABELME`
- `imageData` is always set to `None`
- For rectangle annotations: writes as `shape_type="rectangle"` with two corner points `[[x1,y1],[x2,y2]]`

**`validate(annotation_file)`**: Validates a single `.json` file structure.

**Category discovery**: If `class_file` is not provided, categories are auto-extracted
from `shape.label` values during reading.

## 5. Strict Mode vs Non-Strict Mode Contract

| Error Type | Strict Mode | Non-Strict Mode |
|------------|-------------|-----------------|
| Invalid annotation format (wrong token count, bad JSON) | Abort immediately | Skip annotation, log warning, continue |
| Invalid class_id (not in categories) | Abort immediately | Skip annotation, log warning, continue |
| Invalid coordinate (out of valid range for format) | Abort immediately | Skip annotation, log warning, continue |
| Invalid bbox (zero area, overflow) | Abort immediately | Skip annotation, log warning, continue |
| Image file not found | **Always skip with warning** | **Always skip with warning** |
| Image unreadable / corrupt | **Always skip with warning** | **Always skip with warning** |
| Invalid image dimensions | **Always skip with warning** | **Always skip with warning** |

**Key rule**: Image-related errors never cause an abort — they are always downgraded to
warnings regardless of `strict_mode`.

**Coordinate validity depends on format:**
- **YOLO**: coordinates must be finite floats in [0, 1]
- **COCO**: coordinates must be finite positive numbers (absolute pixels)
- **LabelMe**: coordinates must be finite positive numbers (absolute pixels)

## 6. Format-Aware Validation Constraints

A valid read operation must satisfy ALL of:

1. Source directory/file exists and is readable
2. Source files match the expected structure (`.txt` for YOLO, `.json` for LabelMe/COCO)
3. At least one annotation file is found
4. Categories are loaded (from class_file or auto-extracted)
5. Each annotation line/shape has a valid class_id
6. All coordinates are finite floats in the **format-native range**:
   - YOLO: [0, 1]
   - COCO: positive absolute pixels
   - LabelMe: positive absolute pixels
7. Detection annotations have valid bbox (w > 0, h > 0, no boundary overflow)
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

## 8. Changes from v1 Architecture

| v1 (Deprecated) | v2 (Current) |
|-----------------|--------------|
| Unified normalized internal model (all coords → [0,1] center) | Format-native coordinate storage |
| `OriginalData` + `OriginalDataManager` for lossless A→A round-trip | **Removed entirely** — not needed since coordinates stay native |
| `BoundingBox.xyxy()` / `.xywh_abs()` conversion methods | **Removed** — converters handle coordinate math |
| `Segmentation.points_abs()` conversion method | **Removed** — converters handle coordinate math |
| Validation assumed [0,1] for all formats | Format-aware validation per coordinate system |
| `DatasetAnnotations` had no `format` field | `DatasetAnnotations.format` is required and governs semantics |
| Lossless round-trip contract (Section 6) | **Removed** — see `spec_conversion.md` for cross-format precision docs |
