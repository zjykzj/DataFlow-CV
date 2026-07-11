# Convert Module Specification

> **Version:** v5.2 | **Last Updated:** 2026-07-02
> **Status:** Draft — unified logging via `LogManager`
> **Layer:** Modules
> **Dependencies:** Label module (handlers + models) + Logging module (LogManager)

## 1. Module Overview

The Convert module (`dataflow/convert/`) transforms annotation data between YOLO, COCO, and
LabelMe formats through a standardized pipeline. It depends **only** on the Label module —
it does not import from Visualize or CLI.

### 1.1 Key Design: Explicit Coordinate Transforms + Dual Pipeline

**Converters own all coordinate transformations.** The Label module stores coordinates in
format-native representation. When converting between formats, the converter:

1. Reads source data in its native format (via Label handler)
2. Applies explicit coordinate transformation (this module)
3. Sets `DatasetAnnotations.format` to the target format
4. Writes target data in its native format (via Label handler)

**Two pipelines are available**, selected by the converter based on the target format:

| Pipeline | Trigger | When Used |
|----------|---------|-----------|
| **Batch** (`convert()`) | Default — `handler.read()` → `convert_annotations()` | Single-file target (COCO JSON) or small datasets |
| **Streaming** (`stream_convert()`) | New — `handler.iter_images()` → per-image convert+write | Per-file target (YOLO .txt, LabelMe .json) or large datasets |

The streaming pipeline processes images one at a time:
```
handler.iter_images() → ImageAnnotation → _convert_single_image() → ImageAnnotation → handler.write_one()
```
This avoids holding both source and target datasets in memory simultaneously.

### 1.2 Module Contract

- **Input**: Source format annotations (via `handler.read()` for batch, `handler.iter_images()` for streaming)
- **Processing**: Explicit coordinate transformation — either all-at-once (`convert_annotations()`) or per-image (`_convert_single_image()`)
- **Output**: Target format annotations (via `handler.write()` for batch, `handler.write_one()` for streaming)
- **State**: `_source_annotations_for_target` — MUST be cleaned up in `try/finally`
- **Memory**: Batch path holds source + target in memory; streaming path holds one image at a time

dataflow/convert/
├── base.py                # BaseConverter + ConversionResult + shared streaming/batch pipelines
├── yolo_and_coco.py       # YOLO ↔ COCO converter (uses shared coordinate transforms from utils)
├── labelme_and_yolo.py    # LabelMe ↔ YOLO converter (uses shared coordinate transforms from utils)
├── coco_and_labelme.py    # COCO ↔ LabelMe converter (absolute ↔ absolute, no normalize)
├── rle_converter.py       # Polygon ↔ RLE utility
├── log_templates.py       # Log message template helpers
└── utils.py               # Shared coordinate transforms, category extraction, path resolution
```

## 2. Conversion Pipeline (`BaseConverter`)

### 2.1 Pipeline Contract

#### 2.1.1 Batch Pipeline (`convert()`)

Used when the target format is a single file (e.g., COCO JSON) or the dataset is small:

```
validate_inputs()
    → create_source_handler()
    → handler.read()              → DatasetAnnotations (source-native coords)
    → convert_annotations()       → DatasetAnnotations (target-native coords)
    → create_target_handler()
    → handler.write()             → Target format files
```

**The critical step is `convert_annotations()`, which performs coordinate transformation
on the complete dataset.**

#### 2.1.2 Streaming Pipeline (`stream_convert()`)

Used when the target format is per-file (YOLO .txt, LabelMe .json) and the dataset
is large enough to warrant memory-conscious processing:

```
validate_inputs()
    → create_source_handler()
    → create_target_handler()
    → for each image_ann in handler.iter_images():
          target_ann = _convert_single_image(image_ann)   → ImageAnnotation (target-native coords)
          target_handler.write_one(target_ann)            → Single target file
```

**The critical step is `_convert_single_image()`, which transforms one image at a time.**

**Streaming preconditions:**
- Target format MUST support per-file output (YOLO .txt, LabelMe .json)
- For COCO target (single JSON), use the batch pipeline; streaming is not applicable
- Categories must be extractable upfront (via `handler.categories` or source metadata)
  before iteration begins, so the target handler can be configured correctly

### 2.2 `BaseConverter` Abstract Class

```python
class BaseConverter(ABC):
    def __init__(self, source_format: str, target_format: str,
                 strict_mode=True, log_config=None): ...

    # Template methods — orchestrate the pipeline
    def convert(self, source_path, target_path, **kwargs) -> ConversionResult: ...
    def stream_convert(self, source_path, target_path, **kwargs) -> ConversionResult: ...

    # Hook 1: Validate inputs (default: check source exists, create target parent dir)
    def validate_inputs(self, source_path, target_path, kwargs) -> bool: ...

    # Hook 2: Create source handler (abstract — subclass must implement)
    @abstractmethod
    def create_source_handler(self, source_path, kwargs) -> BaseAnnotationHandler: ...

    # Hook 3: Transform ALL annotations (abstract — batch path)
    @abstractmethod
    def convert_annotations(self, source_annotations, kwargs) -> DatasetAnnotations: ...

    # Hook 4: Transform SINGLE image (abstract — streaming path)
    @abstractmethod
    def _convert_single_image(self, image_ann: ImageAnnotation, kwargs) -> ImageAnnotation: ...

    # Hook 5: Create target handler (abstract — subclass must implement)
    @abstractmethod
    def create_target_handler(self, target_path, kwargs) -> BaseAnnotationHandler: ...

    # Hook 6: Post-stream-image processing (optional — streaming path)
    def _post_stream_image(self, source_ann: ImageAnnotation,
                           target_ann: ImageAnnotation,
                           target_path: str, kwargs: Dict) -> None: ...

    # Hook 7: Post-batch-convert processing (optional — batch path)
    def _post_batch_convert(self, result: ConversionResult,
                            source_handler: BaseAnnotationHandler,
                            kwargs: Dict) -> None: ...

    # Log helpers — shared logging methods for subclasses
    def _log_info(self, message: str) -> None: ...
    def _log_progress(self, current: int, total_objects: int, message: str = "") -> None: ...
    def _log_warning(self, message: str) -> None: ...
    def _log_error(self, message: str) -> None: ...
```

**`_convert_single_image()` contract:**
- Receives a single `ImageAnnotation` with coordinates in the **source format's native space**
- Transforms coordinates to the **target format's native space**
- Returns a new `ImageAnnotation` with the same metadata (image_path, width, height) but
  converted objects
- Does NOT accumulate data — each call is independent
- Same precision characteristics as the batch `convert_annotations()` (§9)

**`_post_stream_image()` contract (optional hook):**
- Called after `_convert_single_image()` and `handler.write_one()` for each image
  in the streaming pipeline
- Default implementation in `BaseConverter` is a no-op
- Receives the source `ImageAnnotation` (pre-transform), target `ImageAnnotation`
  (post-transform), target path, and kwargs
- Override point for per-image side effects during streaming. Example:
  `LabelMeAndYoloConverter` overrides this hook to copy source image files to the
  target images directory per-image during LabelMe→YOLO conversion

**`_post_batch_convert()` contract (optional hook):**
- Called after `handler.write()` completes in the batch pipeline
- Default implementation in `BaseConverter` is a no-op
- Receives the `ConversionResult`, source handler, and kwargs
- Override point for post-conversion side effects. Example: `YoloAndCocoConverter`
  overrides this hook to add RLE accuracy warnings to the result when `do_rle=True`
  and segmentation data exists

### 2.3 Conversion Method Contracts

**`convert_annotations()` — Batch Path:**

**Base class contract**: The default implementation in `BaseConverter` MUST raise
`NotImplementedError`. This forces every concrete converter to explicitly implement
the coordinate transformation for its direction — there is no safe "pass-through"
default because coordinate semantics differ between formats.

Every implementation MUST:

1. Receive a `DatasetAnnotations` with coordinates in the **source format's native space**
2. Transform ALL coordinates to the **target format's native space**
3. Set `result.format = target_format` on the returned DatasetAnnotations
4. Preserve all non-coordinate data (categories, image metadata, is_crowd flags)
5. Document the precision characteristics of the transform (lossless or lossy, see §9)

**Implementation pattern**: The canonical batch `convert_annotations()` delegates to
`_convert_single_image()` per image. This ensures the batch and streaming paths
produce identical output for the same input:

```python
def convert_annotations(self, source_annotations, kwargs):
    target = DatasetAnnotations(format=target_format)
    target.categories = source_annotations.categories.copy()
    for img in source_annotations.images:
        target.add_image(self._convert_single_image(img, **kwargs))
    return target
```

**`_convert_single_image()` — Streaming Path:**

Every implementation MUST:

1. Receive a single `ImageAnnotation` with coordinates in the **source format's native space**
2. Transform all objects' coordinates to the **target format's native space**
3. Return a new `ImageAnnotation` with the same metadata (image_path, width, height,
   image_id) but transformed objects
4. Preserve non-coordinate data (class_name, class_id, confidence, is_crowd)
5. Be stateless — each call is independent
6. Match the precision characteristics of `convert_annotations()` for the same direction

### 2.4 `ConversionResult`

Return type for all converter operations:

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Conversion succeeded |
| `source_format` | str | Source format name |
| `target_format` | str | Target format name |
| `source_path` | str | Input path |
| `target_path` | str | Output path |
| `num_images_converted` | int | Images successfully processed |
| `num_objects_converted` | int | Objects converted |
| `warnings` | List[str] | Non-fatal warnings |
| `errors` | List[str] | Error messages |
| `metadata` | Dict[str, Any] | Additional conversion metadata (includes precision note) |
| `verbose_log` | List[str] | Detailed processing log (verbose mode only) |
| `log_path` | Optional[str] | Log file path (verbose mode only) |

### 2.5 State Management Contract

**Batch path**: Converters store `self._source_annotations_for_target` between pipeline
stages. This is used by `create_target_handler()` for:

- **COCO → YOLO / COCO → LabelMe**: Generating `classes.txt` from COCO categories
- **LabelMe → YOLO**: Copying image files to the target images directory

**Critical rule**: `_source_annotations_for_target` MUST be cleaned up in a `try/finally` block:

```python
try:
    write_result = target_handler.write(converted_annotations, target_path)
finally:
    self._source_annotations_for_target = None
```

Failure to do this causes **stale state leakage** — the next conversion would see the previous
conversion's data.

**Streaming path**: The `_source_annotations_for_target` field is reset to `None`
at the start of `stream_convert()` and again in a `finally` block, ensuring
cleanup even if the streaming loop is interrupted. Category extraction from the
source handler happens **before** iteration begins. For example:

- **COCO → YOLO**: Extract `handler.categories` after `handler.read()` (or from
  `handler.iter_images()` metadata) to generate `classes.txt` before writing images
- **LabelMe → YOLO**: Copy image files per-image during the streaming loop via
  `_post_stream_image()` (no need to hold all images in memory)

The streaming path explicitly cleans up `_source_annotations_for_target` with
the same `try/finally` pattern used in the batch path — no stale state leak.

### 2.6 Coordinate Transform Responsibility

| Direction | Transform Type | Precision | Implementation Location |
|-----------|---------------|-----------|------------------------|
| YOLO → COCO | Normalized → Absolute pixels | **Lossy** (±1 px) | `_convert_single_image()` → shared `utils.yolo_to_absolute_pixel()` |
| COCO → YOLO | Absolute pixels → Normalized | **Lossy** (±1 px) | `_convert_single_image()` → shared `utils.absolute_pixel_to_yolo()` |
| LabelMe → YOLO | Absolute pixels → Normalized | **Lossy** (±1 px) | `_convert_single_image()` → shared `utils.absolute_pixel_to_yolo()` |
| YOLO → LabelMe | Normalized → Absolute pixels | **Lossy** (±1 px) | `_convert_single_image()` → shared `utils.yolo_to_absolute_pixel()` |
| COCO → LabelMe | Absolute → Absolute (no normalize) | **Near-lossless** | `_convert_single_image()` — direct passthrough (no transform needed) |
| LabelMe → COCO | Absolute → Absolute (no normalize) | **Near-lossless** | `_convert_single_image()` — direct passthrough (no transform needed) |

**Shared coordinate transform utilities** (`convert/utils.py`):

All YOLO↔absolute pixel transforms share the same underlying math (center↔top-left
origin shift, normalization/denormalization). These are factored into two canonical
utility functions to eliminate the duplication previously spread across
`YoloAndCocoConverter` and `LabelMeAndYoloConverter`:

```python
def yolo_to_absolute_pixel(
    bbox: Optional[BoundingBox],
    seg: Optional[Segmentation],
    img_width: int, img_height: int,
) -> Tuple[Optional[BoundingBox], Optional[Segmentation]]:
    """Convert YOLO normalized center → absolute pixel top-left.

    Bbox: (cx_norm, cy_norm, w_norm, h_norm) → (x_tl, y_tl, w_abs, h_abs)
    Segmentation: (x_norm, y_norm) per point → (x_abs, y_abs) per point

    Validates that ``img_width`` and ``img_height`` are positive integers.
    Raises ``ValueError`` if either dimension is <= 0.

    This is a pure function with no side effects. Used by:
    - YoloAndCocoConverter (YOLO → COCO)
    - LabelMeAndYoloConverter (YOLO → LabelMe)
    """

def absolute_pixel_to_yolo(
    bbox: Optional[BoundingBox],
    seg: Optional[Segmentation],
    img_width: int, img_height: int,
) -> Tuple[Optional[BoundingBox], Optional[Segmentation]]:
    """Convert absolute pixel top-left → YOLO normalized center.

    Bbox: (x_tl, y_tl, w_abs, h_abs) → (cx_norm, cy_norm, w_norm, h_norm)
    Segmentation: (x_abs, y_abs) per point → (x_norm, y_norm) per point

    Validates that ``img_width`` and ``img_height`` are positive integers.
    Raises ``ValueError`` if either dimension is <= 0.

    This is a pure function with no side effects. Used by:
    - YoloAndCocoConverter (COCO → YOLO)
    - LabelMeAndYoloConverter (LabelMe → YOLO)
    """
```

**Key design rules:**
1. These utilities operate on individual `BoundingBox`/`Segmentation` objects — they
   do NOT process `ImageAnnotation` or `DatasetAnnotations` containers
2. They are pure functions — stateless, no side effects, no handler interaction
3. Callers remain responsible for class_id/class_name/confidence/is_crowd preservation
4. The COCO↔LabelMe direction (both absolute pixel) does NOT use these — coordinates
   pass through unchanged

### 2.7 Streaming Category Pre-Loading

When the streaming pipeline targets YOLO or LabelMe (per-file formats), the target
handler must be configured before iteration begins. This requires category information
to be available upfront. For source formats that expose categories lazily (e.g., COCO
JSON), a dedicated pre-loading step is needed.

**`_ensure_categories_for_streaming()` contract:**

```python
def _ensure_categories_for_streaming(
    self,
    source_handler: BaseAnnotationHandler,
    source_path: str,
    kwargs: Dict,
) -> None:
    """Ensure categories are available before streaming iteration.

    Called by ``stream_convert()`` before ``create_target_handler()``.
    Subclasses may override to pre-load categories from source files
    that don't expose them until ``read()`` / ``iter_images()`` runs.

    Default implementation:
        Resets ``self._source_annotations_for_target = None`` at entry,
        then checks ``source_handler.categories`` — if it's a non-empty dict,
        stores a minimal ``DatasetAnnotations`` with those categories in
        ``self._source_annotations_for_target``.

    Subclass overrides (COCO sources):
        Resets ``self._source_annotations_for_target = None`` at entry,
        then reads categories directly from the COCO JSON file's ``"categories"``
        array. This avoids loading the full dataset into memory just to
        extract category mappings.
    """
```

Implementation is in `BaseConverter` (default) with COCO-specific overrides. The
COCO JSON reading logic (`read_coco_categories()`) is a shared helper in
`convert/utils.py` to avoid duplication between `YoloAndCocoConverter` and
`CocoAndLabelMeConverter`.

### 2.8 Logging Contract

See [`spec_logging.md`](spec_logging.md) for the full `LogManager` contract. Convert-specific:

**Constructor**: `BaseConverter.__init__(source_format, target_format, strict_mode=True, log_config=None)`

- If `log_config` is None, a default `LogConfig(name=f"convert.{source_format}_to_{target_format}")` is created
- The converter creates a `LogManager` from the config and propagates `self.logger` to handlers via `logger=self.logger`

**Streaming path progress** (yolo2labelme, labelme2yolo, coco2yolo, coco2labelme):

Every 50 images, the converter emits an INFO-level progress line:

```
14:23:17  INFO     Converted 50 images, 320 objects - test.jpg
14:23:19  INFO     Converted 100 images, 645 objects - test2.jpg
```

At completion, a summary line is emitted:

```
14:23:22  INFO     Converted 500 images, 3240 objects in 16.2s
```

**Batch path progress** (yolo2coco, labelme2coco):

After reading and after writing, the converter emits INFO-level statistics:

```
14:23:17  INFO     Read 500 images, 3 categories, 3240 objects
14:23:22  INFO     Wrote output.json (3240 annotations) in 4.8s
```

**Verbose mode** (`--verbose`):

| Aspect | `verbose=False` (default) | `verbose=True` |
|--------|--------------------------|----------------|
| Console | Progress every 50 images (streaming) or read/write stats (batch); final result | Same as non-verbose, plus a header line with paths/mode/options |
| File output | None | Full DEBUG-level log (per-image details, handler internals) |
| `log_path` in result | `None` | Set to log file path |

## 3. Concrete Converters

### 3.1 `YoloAndCocoConverter`

**Constructor:** `YoloAndCocoConverter(source_to_target: bool, prediction: bool = False, log_config=None, **kwargs)`

- `source_to_target=True` → YOLO → COCO
- `source_to_target=False` → COCO → YOLO
- `prediction=True` → Read YOLO files in prediction format (with confidence). Only meaningful when `source_to_target=True`.

**Required parameters per direction:**

| Direction | class_file | image_dir | do_rle | prediction |
|-----------|-----------|-----------|--------|------------|
| YOLO → COCO | **Required** | **Required** | Optional (default False) | Optional (default False) |
| COCO → YOLO | Optional (auto-generated) | Optional (auto-created) | N/A | N/A |

**YOLO → COCO behavior (batch — COCO output is single JSON):**
1. Validates `class_file` and `image_dir` exist
2. Creates `YoloAnnotationHandler(prediction=prediction)` as source, reads labels → `DatasetAnnotations(format=YOLO)`
   - Label mode (`prediction=False`): parses 5-token detection, odd-token segmentation; confidence defaults to 1.0
   - Prediction mode (`prediction=True`): parses 6-token detection prediction, even-token segmentation prediction; confidence extracted from last token
3. `convert_annotations()`:
   - Reads YOLO-native coordinates (normalized, center-based)
   - Transforms to COCO-native coordinates (absolute pixels, top-left)
   - Preserves `ObjectAnnotation.confidence` (used as COCO `score` in output)
   - Sets `result.format = AnnotationFormat.COCO`
4. Creates `CocoAnnotationHandler(prediction=prediction, do_rle=...)` as target
5. COCO output depends on prediction mode:
   - **Annotation mode** (`prediction=False`): Full COCO dict with `info`, `images`, `annotations`, `categories`. ``score`` is omitted unless ``confidence < 1.0``.
   - **Prediction mode** (`prediction=True`): Plain JSON list of annotation dicts (Variant B per `spec_coco_format.md` §10.1). Each dict contains `image_id`, `category_id`, `bbox`, `area`, `score`, and optionally `segmentation`. No `images`/`categories`/`info` wrapper — these are sourced from GT at evaluation time via `loadRes()`.
6. If `do_rle=True`, adds RLE accuracy warning to result

**COCO → YOLO behavior (streaming — YOLO output is per-file):**
1. Creates directory structure: `target_path/labels/` and `target_path/images/`
2. Generates `classes.txt` from COCO categories if not provided
3. Creates `CocoAnnotationHandler` as source. Uses `_ensure_categories_for_streaming()` (which calls shared helper `read_coco_categories()` from utils) to pre-load categories from the COCO JSON file without loading the full dataset into memory, since the streaming pipeline needs categories to configure the target handler before iteration begins.
4. `stream_convert()`:
   - Iterates `handler.iter_images()` (yields one `ImageAnnotation` per image)
   - `_convert_single_image()`:
     - Reads COCO-native coordinates (absolute pixels, top-left)
     - Transforms to YOLO-native coordinates (normalized, center-based)
   - Writes one `.txt` file per image immediately
5. Creates `YoloAnnotationHandler` as target upfront (categories known from COCO metadata)

**Streaming applicability:**
- YOLO → COCO: **Batch only** — COCO output is a single JSON; cannot stream
- COCO → YOLO: **Streaming** — YOLO output is per-file .txt; streaming reduces memory

### 3.2 `LabelMeAndYoloConverter`

**Constructor:** `LabelMeAndYoloConverter(source_to_target: bool, log_config=None, **kwargs)`

- `source_to_target=True` → LabelMe → YOLO
- `source_to_target=False` → YOLO → LabelMe

**Required parameters per direction:**

| Direction | class_file | image_dir |
|-----------|-----------|-----------|
| LabelMe → YOLO | **Required** | Optional |
| YOLO → LabelMe | **Required** | **Required** |

**LabelMe → YOLO behavior (streaming — both formats are per-file):**
1. Creates directory structure: `target_path/labels/`, `target_path/images/`
2. Copies `classes.txt` to target directory
3. `stream_convert()`:
   - Iterates `LabelMeAnnotationHandler.iter_images()` (yields one `ImageAnnotation` per JSON)
   - `_convert_single_image()`: LabelMe absolute pixels → YOLO normalized
   - `_post_stream_image()` (overridden): Copies the source image file to
     `target_path/images/` per-image during the streaming loop, avoiding the
     need to hold all image paths in memory
   - Writes one `.txt` file per image immediately
4. Creates `LabelMeAnnotationHandler` as source, `YoloAnnotationHandler` as target

**YOLO → LabelMe behavior (streaming — both formats are per-file):**
1. Creates output directory
2. `stream_convert()`:
   - Iterates `YoloAnnotationHandler.iter_images()` (yields one `ImageAnnotation` per .txt)
   - `_convert_single_image()`: YOLO normalized → LabelMe absolute pixels
   - Writes one `.json` file per image immediately
3. Creates `YoloAnnotationHandler` as source, `LabelMeAnnotationHandler` as target

### 3.3 `CocoAndLabelMeConverter`

**Constructor:** `CocoAndLabelMeConverter(source_to_target: bool, log_config=None, **kwargs)`

- `source_to_target=True` → COCO → LabelMe
- `source_to_target=False` → LabelMe → COCO

**Required parameters per direction:**

| Direction | class_file | do_rle |
|-----------|-----------|--------|
| COCO → LabelMe | Optional (auto-generated) | N/A |
| LabelMe → COCO | **Required** | Optional (default False) |

**Conversion characteristics:**
- Both formats use absolute pixels — **no normalization step**
- Transform is limited to reformatting between COCO bbox `[x,y,w,h]` and LabelMe rectangle `[[x1,y1],[x2,y2]]`
- Polygon points are reformatted (flatten/unflatten) but values are preserved
- Precision: **near-lossless** (bound by floating-point arithmetic, << 1 pixel)

**COCO → LabelMe behavior (streaming — LabelMe output is per-file):**
1. `stream_convert()`:
   - Iterates `CocoAnnotationHandler.iter_images()` (yields one `ImageAnnotation` per image)
   - `_convert_single_image()`: COCO absolute → LabelMe absolute (structural reformat only)
   - Writes one `.json` file per image immediately
2. Generates `classes.txt` from COCO categories if not provided

**LabelMe → COCO behavior (batch — COCO output is single JSON):**
1. Validates `class_file` exists
2. Creates `LabelMeAnnotationHandler` as source → reads → `DatasetAnnotations(format=LABELME)`
3. `convert_annotations()`:
   - Reformat LabelMe-native coordinates to COCO-native coordinates
   - Both are absolute pixels; only structural reformatting needed
   - Sets `result.format = AnnotationFormat.COCO`
4. Creates `CocoAnnotationHandler` as target with `do_rle` setting
5. If `do_rle=True`, adds RLE accuracy warning

**Streaming applicability:**
- COCO → LabelMe: **Streaming** — LabelMe output is per-file .json
- LabelMe → COCO: **Batch only** — COCO output is a single JSON; cannot stream

## 4. RLE Converter (`RLEConverter`)

Standalone utility class for polygon ↔ RLE conversion. Not a subclass of `BaseConverter`.

### 4.1 Constructor

```python
RLEConverter(logger: Optional[logging.Logger] = None)  # Receives logger from converter's LogManager
```

### 4.2 Public API

| Method | Input → Output | Requires pycocotools |
|--------|---------------|---------------------|
| `polygon_to_rle(points, img_w, img_h)` | Absolute pixel polygon → RLE dict | Yes |
| `rle_to_polygon(rle, img_w, img_h)` | RLE dict → Absolute pixel polygon | Yes |
| `get_rle_accuracy_warning()` | — → Warning string | No |
| `check_coco_mask_available()` | — → bool | No |
| `validate_rle_dict(rle)` | RLE dict → bool | No |

### 4.3 RLE Encoding Contract

1. Polygon points → absolute pixels → binary mask (cv2.fillPoly) → RLE (coco_mask.encode)
2. RLE `counts` bytes → `counts_bytes.decode("latin1")` for JSON serialization
3. **Accuracy loss warning**: Polygon → RLE involves rasterization to a binary mask. The RLE accurately represents the mask, but the mask is an approximation of the original polygon.

### 4.4 Graceful Degradation

When pycocotools is not installed:
- `HAS_COCO_MASK = False`
- `polygon_to_rle()` raises `ImportError` if `require_coco_mask=True`
- `rle_to_polygon()` raises `ImportError` if `require_coco_mask=True`
- RLE data in COCO files is preserved as-is but cannot be decoded

### 4.5 Coordinate Contract

The RLE converter works in **absolute pixel coordinates** only. It is the caller's
responsibility to:
- Convert YOLO normalized coordinates → absolute pixels before calling `polygon_to_rle()`
- Convert absolute pixel coordinates → normalized after calling `rle_to_polygon()` (if needed)

## 5. Converter Utilities (`utils.py`)

| Function | Purpose |
|----------|---------|
| `extract_categories_from_annotations(dataset)` | Extracts `Dict[id, name]` from DatasetAnnotations |
| `yolo_to_absolute_pixel(bbox, seg, img_w, img_h)` | **Shared** — YOLO normalized center → absolute pixel top-left (bbox + polygon). The parameter ``seg`` is a ``Segmentation`` object (alias for the historical ``segmentation`` name in earlier docs). |
| `absolute_pixel_to_yolo(bbox, seg, img_w, img_h)` | **Shared** — absolute pixel top-left → YOLO normalized center (bbox + polygon). The parameter ``seg`` is a ``Segmentation`` object (alias for the historical ``segmentation`` name in earlier docs). |
| `read_coco_categories(json_path)` | **Shared** — reads categories from a COCO JSON file without full dataset load |
| `generate_classes_file(categories, path)` | Writes `classes.txt` from category dict |
| `load_classes_file(path)` | Reads `classes.txt` into `Dict[int, str]` |
| `extract_categories_from_coco(coco_data)` | Extracts categories from raw COCO dict |
| `ensure_categories_in_annotations(dataset)` | Ensures category consistency |
| `get_image_dimensions_from_handler(handler)` | Extracts image dimensions |
| `normalize_path(path, base_dir)` | Normalizes file paths |
| `validate_conversion_chain(source_format, target_format, allowed_chains)` | Validates a (source, target) pair against a list of allowed conversion chains |
| `create_conversion_chain(chain)` | Creates a multi-step conversion pipeline from a list of format names |
| `resolve_image_paths(annotations, source_dir, target_dir)` | Resolves relative image paths |

### 5.1 RLE Accuracy Warning

The RLE accuracy warning is **conditionally emitted** only when both conditions are met:

- `do_rle=True` is set on the converter
- The source dataset contains segmentation data (`source_handler.is_seg` is `True`)

Detection-only datasets with `--do-rle` do not produce a misleading warning. The check is performed in `_post_batch_convert()` (e.g., `YoloAndCocoConverter` and `CocoAndLabelMeConverter`).

## 6. Dependency Contract

```
Convert module imports FROM:
├── dataflow.label.base          (AnnotationResult)
├── dataflow.label.models        (DatasetAnnotations, AnnotationFormat, ...)
├── dataflow.label.yolo_handler  (YoloAnnotationHandler)
├── dataflow.label.coco_handler  (CocoAnnotationHandler)
├── dataflow.label.labelme_handler (LabelMeAnnotationHandler)
├── dataflow.util.logging        (LogConfig, LogManager)

Convert module does NOT import FROM:
├── dataflow.visualize.*         (FORBIDDEN — zero cross-dependency)
└── dataflow.cli.*               (FORBIDDEN — CLI depends on Convert, not vice versa)
```

## 7. Error Handling Contract

### 7.1 Error Propagation

**Batch path:**
```
validate_inputs() fails → ConversionResult(success=False, errors=[...])
handler.read() fails     → ConversionResult(success=False, errors=read_result.errors)
handler.write() fails    → ConversionResult(success=False, errors=write_result.errors)
```

**Streaming path:**
```
validate_inputs() fails           → ConversionResult(success=False, errors=[...])
handler.iter_images() raises      → ConversionResult(success=False, errors=[error_msg])
                                   (partial results: images written before the error are valid)
handler.write_one() fails         → ConversionResult(success=False, errors=[file_error])
                                   (remaining images not processed)
```

### 7.2 Strict vs Non-Strict

Converters pass `strict_mode` to handlers. The handler controls whether errors abort or
skip — the converter does not add its own strict/non-strict logic beyond what handlers
already enforce.

**Streaming strict mode**: When `iter_images()` raises `ValueError` mid-stream,
the converter catches it and returns a partial result — images already written before
the error are on disk. This is different from the batch path which is all-or-nothing.

### 7.3 State Cleanup Guarantee

**Batch path**: `_source_annotations_for_target` is always cleared — even if
`handler.write()` raises an exception — because it's wrapped in `try/finally`.

**Streaming path**: No accumulated state to clean up — images are written as they
are processed. If the stream is interrupted, already-written files remain on disk.


