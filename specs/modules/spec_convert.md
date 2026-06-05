# Convert Module Specification

> **Version:** 2.0
> **Status:** Draft — updated for explicit coordinate transforms in converters
> **Layer:** Modules
> **Dependencies:** Label module (handlers + models)

## 1. Module Overview

The Convert module (`dataflow/convert/`) transforms annotation data between YOLO, COCO, and
LabelMe formats through a standardized pipeline. It depends **only** on the Label module —
it does not import from Visualize or CLI.

### 1.1 Key Design: Explicit Coordinate Transforms

**Converters own all coordinate transformations.** The Label module stores coordinates in
format-native representation. When converting between formats, the converter:

1. Reads source data in its native format (via Label handler)
2. Applies explicit coordinate transformation (this module)
3. Sets `DatasetAnnotations.format` to the target format
4. Writes target data in its native format (via Label handler)

### 1.2 Module Contract

- **Input**: Source format annotations (via Label handlers — coordinates in source-native form)
- **Processing**: Explicit coordinate transformation from source to target native representation
- **Output**: Target format annotations (via Label handlers — coordinates in target-native form)
- **State**: `_source_annotations_for_target` — MUST be cleaned up in `try/finally`

### 1.3 File Map

```
dataflow/convert/
├── base.py                # BaseConverter + ConversionResult
├── yolo_and_coco.py       # YOLO ↔ COCO converter (with normalize/denormalize)
├── labelme_and_yolo.py    # LabelMe ↔ YOLO converter (with normalize/denormalize)
├── coco_and_labelme.py    # COSO ↔ LabelMe converter (absolute ↔ absolute, no normalize)
├── rle_converter.py       # Polygon ↔ RLE utility
└── utils.py               # Category extraction, path resolution
```

## 2. Conversion Pipeline (`BaseConverter`)

### 2.1 Pipeline Contract

Every converter must follow this exact sequence:

```
validate_inputs()
    → create_source_handler()
    → handler.read()              → DatasetAnnotations (source-native coords)
    → convert_annotations()       → DatasetAnnotations (target-native coords)
    → create_target_handler()
    → handler.write()             → Target format files
```

**The critical step is `convert_annotations()`, which performs coordinate transformation.**

### 2.2 `BaseConverter` Abstract Class

```python
class BaseConverter(ABC):
    def __init__(self, source_format: str, target_format: str,
                 strict_mode=True, verbose=False, logger=None): ...

    # Template method — orchestrates the pipeline
    def convert(self, source_path, target_path, **kwargs) -> ConversionResult: ...

    # Hook 1: Validate inputs (default: check source exists, create target parent dir)
    def validate_inputs(self, source_path, target_path, kwargs) -> bool: ...

    # Hook 2: Create source handler (abstract — subclass must implement)
    @abstractmethod
    def create_source_handler(self, source_path, kwargs) -> BaseAnnotationHandler: ...

    # Hook 3: Transform annotations (abstract — subclass implements coordinate transform)
    @abstractmethod
    def convert_annotations(self, source_annotations, kwargs) -> DatasetAnnotations: ...

    # Hook 4: Create target handler (abstract — subclass must implement)
    @abstractmethod
    def create_target_handler(self, target_path, kwargs) -> BaseAnnotationHandler: ...
```

### 2.3 `convert_annotations()` Contract

Every implementation of `convert_annotations()` MUST:

1. Receive a `DatasetAnnotations` with coordinates in the **source format's native space**
2. Transform ALL coordinates to the **target format's native space**
3. Set `result.format = target_format` on the returned DatasetAnnotations
4. Preserve all non-coordinate data (categories, image metadata, is_crowd flags)
5. Document the precision characteristics of the transform (lossless or lossy, see §9)

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
| `log_file_path` | Optional[str] | Log file path (verbose mode only) |

### 2.5 State Management Contract

Converters store `self._source_annotations_for_target` between pipeline stages. This is used
by `create_target_handler()` for:

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

### 2.6 Coordinate Transform Responsibility

| Direction | Transform Type | Precision | Implementation Location |
|-----------|---------------|-----------|------------------------|
| YOLO → COCO | Normalized → Absolute pixels | **Lossy** (±1 px) | `convert_annotations()` in `YoloAndCocoConverter` |
| COCO → YOLO | Absolute pixels → Normalized | **Lossy** (±1 px) | `convert_annotations()` in `YoloAndCocoConverter` |
| LabelMe → YOLO | Absolute pixels → Normalized | **Lossy** (±1 px) | `convert_annotations()` in `LabelMeAndYoloConverter` |
| YOLO → LabelMe | Normalized → Absolute pixels | **Lossy** (±1 px) | `convert_annotations()` in `LabelMeAndYoloConverter` |
| COCO → LabelMe | Absolute → Absolute (no normalize) | **Near-lossless** | `convert_annotations()` in `CocoAndLabelMeConverter` |
| LabelMe → COCO | Absolute → Absolute (no normalize) | **Near-lossless** | `convert_annotations()` in `CocoAndLabelMeConverter` |

### 2.7 Verbose Logging Contract

When `verbose=True`:
- Logger is configured with file output via `VerboseLoggingOperations`
- `log_file_path` is recorded in `ConversionResult`
- Detailed processing steps are logged at DEBUG level
- Conversion duration is recorded

When `verbose=False`:
- Console-only logging at INFO level
- `log_file_path` is `None`

## 3. Concrete Converters

### 3.1 `YoloAndCocoConverter`

**Constructor:** `YoloAndCocoConverter(source_to_target: bool, prediction: bool = False, verbose=False, **kwargs)`

- `source_to_target=True` → YOLO → COCO
- `source_to_target=False` → COCO → YOLO
- `prediction=True` → Read YOLO files in prediction format (with confidence). Only meaningful when `source_to_target=True`.

**Required parameters per direction:**

| Direction | class_file | image_dir | do_rle | prediction |
|-----------|-----------|-----------|--------|------------|
| YOLO → COCO | **Required** | **Required** | Optional (default False) | Optional (default False) |
| COCO → YOLO | Optional (auto-generated) | Optional (auto-created) | N/A | N/A |

**YOLO → COCO behavior:**
1. Validates `class_file` and `image_dir` exist
2. Creates `YoloAnnotationHandler(prediction=prediction)` as source, reads labels → `DatasetAnnotations(format=YOLO)`
   - Label mode (`prediction=False`): parses 5-token detection, odd-token segmentation; confidence defaults to 1.0
   - Prediction mode (`prediction=True`): parses 6-token detection prediction, even-token segmentation prediction; confidence extracted from last token
3. `convert_annotations()`:
   - Reads YOLO-native coordinates (normalized, center-based)
   - Transforms to COCO-native coordinates (absolute pixels, top-left)
   - Preserves `ObjectAnnotation.confidence` (used as COCO `score` in output)
   - Sets `result.format = AnnotationFormat.COCO`
4. Creates `CocoAnnotationHandler` as target with `do_rle` setting
5. COCO output: includes `"score"` field when `confidence < 1.0` (always true in prediction mode)
6. If `do_rle=True`, adds RLE accuracy warning to result

**COCO → YOLO behavior:**
1. Creates directory structure: `target_path/labels/` and `target_path/images/`
2. Generates `classes.txt` from COCO categories if not provided
3. Creates `CocoAnnotationHandler` as source, reads JSON → `DatasetAnnotations(format=COCO)`
4. `convert_annotations()`:
   - Reads COCO-native coordinates (absolute pixels, top-left)
   - Transforms to YOLO-native coordinates (normalized, center-based)
   - Sets `result.format = AnnotationFormat.YOLO`
5. Creates `YoloAnnotationHandler` as target, writes `.txt` files

### 3.2 `LabelMeAndYoloConverter`

**Constructor:** `LabelMeAndYoloConverter(source_to_target: bool, verbose=False, **kwargs)`

- `source_to_target=True` → LabelMe → YOLO
- `source_to_target=False` → YOLO → LabelMe

**Required parameters per direction:**

| Direction | class_file | image_dir |
|-----------|-----------|-----------|
| LabelMe → YOLO | **Required** | Optional |
| YOLO → LabelMe | **Required** | **Required** |

**LabelMe → YOLO behavior:**
1. Creates directory structure: `target_path/labels/`, `target_path/images/`
2. Copies `classes.txt` to target directory
3. Copies image files from source to `target_path/images/`
4. Creates `LabelMeAnnotationHandler` → reads → `DatasetAnnotations(format=LABELME)`
5. `convert_annotations()`:
   - Reads LabelMe-native coordinates (absolute pixels)
   - Transforms to YOLO-native coordinates (normalized)
   - Sets `result.format = AnnotationFormat.YOLO`
6. Creates `YoloAnnotationHandler` → writes

**YOLO → LabelMe behavior:**
1. Creates output directory
2. Creates `YoloAnnotationHandler` → reads → `DatasetAnnotations(format=YOLO)`
3. `convert_annotations()`:
   - Reads YOLO-native coordinates (normalized)
   - Transforms to LabelMe-native coordinates (absolute pixels)
   - Sets `result.format = AnnotationFormat.LABELME`
4. Creates `LabelMeAnnotationHandler` → writes

### 3.3 `CocoAndLabelMeConverter`

**Constructor:** `CocoAndLabelMeConverter(source_to_target: bool, verbose=False, **kwargs)`

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

**LabelMe → COCO behavior:**
1. Validates `class_file` exists
2. Creates `LabelMeAnnotationHandler` as source → reads → `DatasetAnnotations(format=LABELME)`
3. `convert_annotations()`:
   - Reformat LabelMe-native coordinates to COCO-native coordinates
   - Both are absolute pixels; only structural reformatting needed
   - Sets `result.format = AnnotationFormat.COCO`
4. Creates `CocoAnnotationHandler` as target with `do_rle` setting
5. If `do_rle=True`, adds RLE accuracy warning

## 4. RLE Converter (`RLEConverter`)

Standalone utility class for polygon ↔ RLE conversion. Not a subclass of `BaseConverter`.

### 4.1 Constructor

```python
RLEConverter(logger: Optional[logging.Logger] = None)
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
| `generate_classes_file(categories, path)` | Writes `classes.txt` from category dict |
| `load_classes_file(path)` | Reads `classes.txt` into `Dict[int, str]` |
| `extract_categories_from_coco(coco_data)` | Extracts categories from raw COCO dict |
| `ensure_categories_in_annotations(dataset)` | Ensures category consistency |
| `get_image_dimensions_from_handler(handler)` | Extracts image dimensions |
| `normalize_path(path)` | Normalizes file paths |
| `validate_conversion_chain(source_format, target_format, allowed_chains)` | Validates a (source, target) pair against a list of allowed conversion chains |
| `create_conversion_chain(chain)` | Creates a multi-step conversion pipeline from a list of format names |
| `resolve_image_paths(dataset, base_dir)` | Resolves relative image paths |

## 6. Dependency Contract

```
Convert module imports FROM:
├── dataflow.label.base          (AnnotationResult)
├── dataflow.label.models        (DatasetAnnotations, AnnotationFormat, ...)
├── dataflow.label.yolo_handler  (YoloAnnotationHandler)
├── dataflow.label.coco_handler  (CocoAnnotationHandler)
├── dataflow.label.labelme_handler (LabelMeAnnotationHandler)
└── dataflow.util                (FileOperations, logging)

Convert module does NOT import FROM:
├── dataflow.visualize.*         (FORBIDDEN — zero cross-dependency)
└── dataflow.cli.*               (FORBIDDEN — CLI depends on Convert, not vice versa)
```

## 7. Error Handling Contract

### 7.1 Error Propagation

```
validate_inputs() fails → ConversionResult(success=False, errors=[...])
handler.read() fails     → ConversionResult(success=False, errors=read_result.errors)
handler.write() fails    → ConversionResult(success=False, errors=write_result.errors)
```

### 7.2 Strict vs Non-Strict

Converters pass `strict_mode` to handlers. The handler controls whether errors abort or
skip — the converter does not add its own strict/non-strict logic beyond what handlers
already enforce.

### 7.3 State Cleanup Guarantee

`_source_annotations_for_target` is always cleared — even if `handler.write()` raises an
exception — because it's wrapped in `try/finally`.
