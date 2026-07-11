# Annotation Format Conversion Specification

> **Version:** v2.0 | **Last Updated:** 2026-07-02
> **Status:** Draft — updated to reflect native-coordinate architecture

## 1. Conversion Architecture

### 1.1 Pipeline

All conversions follow this pipeline:

```
Source Format → Handler.read() → DatasetAnnotations (native coords)
                                      ↓
                              Converter.convert()
                              (explicit coordinate transform)
                                      ↓
                              DatasetAnnotations (target native coords)
                                      ↓
                              Target Handler.write() → Target Format
```

### 1.2 Coordinate Semantics by Format

`DatasetAnnotations.format` field defines how to interpret all coordinates:

| Format | Bounding Box | Segmentation | Type |
|--------|-------------|-------------|------|
| YOLO | `(cx, cy, w, h)` normalized [0,1], center-based | `(x, y)` normalized [0,1] | Normalized |
| COCO | `(x_tl, y_tl, w_abs, h_abs)` absolute pixels | `(x, y)` absolute pixels | Absolute |
| LabelMe | `(x_tl, y_tl, w_abs, h_abs)` absolute pixels | `(x, y)` absolute pixels | Absolute |

### 1.3 Converter State Management

Converters store `self._source_annotations_for_target` during the conversion pipeline.
This state is used by `create_target_handler()` for:
- Generating `classes.txt` from source categories (COCO→YOLO, COCO→LabelMe)
- Copying image files (LabelMe→YOLO)

**Critical**: This state must be cleared in a `try/finally` block to prevent stale state leakage on exceptions.

## 2. Supported Conversion Directions

| Direction | Converter Class | CLI Command | Notes |
|-----------|----------------|-------------|-------|
| YOLO → COCO | `YoloAndCocoConverter(source_to_target=True)` | `yolo2coco` | Supports `--prediction` flag for model output conversion |
| COCO → YOLO | `YoloAndCocoConverter(source_to_target=False)` | `coco2yolo` | |
| LabelMe → YOLO | `LabelMeAndYoloConverter(source_to_target=True)` | `labelme2yolo` | |
| YOLO → LabelMe | `LabelMeAndYoloConverter(source_to_target=False)` | `yolo2labelme` | |
| LabelMe → COCO | `CocoAndLabelMeConverter(source_to_target=False)` | `labelme2coco` | |
| COCO → LabelMe | `CocoAndLabelMeConverter(source_to_target=True)` | `coco2labelme` | |

## 3. Coordinate Transformation Rules

All coordinate transforms happen in converter code, not in handlers.

### 3.1 YOLO ↔ COCO (bbox)

#### YOLO → COCO

YOLO bbox `(cx_norm, cy_norm, w_norm, h_norm)` → COCO bbox `[x_tl, y_tl, w_abs, h_abs]`:

```
cx = cx_norm * img_width
cy = cy_norm * img_height
w  = w_norm * img_width
h  = h_norm * img_height

x_tl = cx - w / 2
y_tl = cy - h / 2

COCO bbox = [x_tl, y_tl, w, h]
```

**This conversion is lossy.** See §9 for details.

#### COCO → YOLO

COCO bbox `[x_tl, y_tl, w_abs, h_abs]` → YOLO bbox `(cx_norm, cy_norm, w_norm, h_norm)`:

```
cx_abs = x_tl + w_abs / 2
cy_abs = y_tl + h_abs / 2

cx_norm = cx_abs / img_width
cy_norm = cy_abs / img_height
w_norm  = w_abs / img_width
h_norm  = h_abs / img_height
```

**This conversion is lossy.** See §9 for details.

#### 3.1.1 YOLO → COCO Prediction Mode

When converting YOLO **prediction** files to COCO **prediction** JSON (`--prediction` flag):

1. Coordinate transform is **identical** to the label conversion path (§3.1 YOLO → COCO bbox, §3.2 YOLO → COCO segmentation)
2. The converter passes `prediction=True` to **both** `YoloHandler` (source) and `CocoAnnotationHandler` (target)
3. **Output format**: Plain JSON list of annotation dicts (Variant B per `spec_coco_format.md` §10.1) — NOT a full COCO dict with `images`/`categories`. This is the standard prediction format used by Detectron2, MMDetection, and pycocotools' `loadRes()`. Images and categories are sourced from GT at evaluation time.
4. YOLO confidence values are preserved as the COCO `score` field in every annotation

**Data flow:**

```
YOLO pred (.txt):   class_id cx cy w h confidence  (6 tokens, detection)
                 or class_id x1 y1 ... xn yn confidence  (even tokens, segmentation)

  → YoloHandler(prediction=True): parse last token as confidence
  → DatasetAnnotations(format=YOLO, ObjectAnnotation.confidence = parsed_value)
  → Converter.convert_annotations(): denormalize coords (same as label mode)
  → DatasetAnnotations(format=COCO, ObjectAnnotation.confidence preserved)
  → CocoHandler(prediction=True).write(): list of annotation dicts with "score" field
  → pred.json: [{"image_id": 0, "category_id": 7, "bbox": [...], "score": 0.95}, ...]
```

**Annotation mode** (without `--prediction`) produces a full COCO dict:
```
  → pred.json: {"info": {...}, "images": [...], "annotations": [...], "categories": [...]}
```

**Segmentation prediction output**: Polygon format is recommended (default `do_rle=False`). pycocotools `COCOeval` handles polygon→RLE conversion internally during mask IoU computation. See `spec_coco_format.md` §10.4 for rationale.

### 3.2 YOLO ↔ COCO (segmentation)

```
YOLO → COCO:  abs_x = x_norm * img_width;   abs_y = y_norm * img_height

COCO → YOLO:  norm_x = x_abs / img_width;    norm_y = y_abs / img_height
```

All conversions involving YOLO are lossy (normalized ↔ absolute pixel round-trip).

### 3.3 YOLO ↔ LabelMe (bbox)

#### YOLO → LabelMe (bbox → rectangle)

```
cx = cx_norm * img_width
cy = cy_norm * img_height
w  = w_norm * img_width
h  = h_norm * img_height

x1 = cx - w / 2
y1 = cy - h / 2
x2 = cx + w / 2
y2 = cy + h / 2

LabelMe rectangle points = [[x1, y1], [x2, y2]]
```

**Lossy** (normalized → absolute pixel rounding).

#### LabelMe → YOLO (rectangle → bbox)

```
x_min = min(x1, x2);  x_max = max(x1, x2)
y_min = min(y1, y2);  y_max = max(y1, y2)

cx_norm = ((x_min + x_max) / 2) / img_width
cy_norm = ((y_min + y_max) / 2) / img_height
w_norm  = (x_max - x_min) / img_width
h_norm  = (y_max - y_min) / img_height
```

**Lossy** (absolute pixel → normalized rounding).

### 3.4 YOLO ↔ LabelMe (segmentation)

```
YOLO → LabelMe:  abs_pts = [(x * W, y * H) for (x, y) in points]
LabelMe → YOLO:  norm_pts = [(x / W, y / H) for (x, y) in points]
```

**Lossy** (normalized ↔ absolute pixel round-trip).

### 3.5 COCO ↔ LabelMe

Both formats use absolute pixels, so no normalization is involved:

#### COCO → LabelMe (bbox → rectangle)

```
COCO bbox [x_tl, y_tl, w, h] → rectangle points [[x_tl, y_tl], [x_tl+w, y_tl+h]]
```

#### LabelMe → COCO (rectangle → bbox)

```
LabelMe [[x1, y1], [x2, y2]] → COCO [min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)]
```

#### COCO ↔ LabelMe (segmentation)

```
COCO polygon [[x1,y1,x2,y2,...], ...] ↔ LabelMe [[x1,y1], [x2,y2], ...]
Flatten/unflatten coordinate lists.
```

**COCO ↔ LabelMe conversions involve no normalization step.**
Fidelity is limited by integer pixel rounding in the internal representation; see §9.

### 3.6 Post-Conversion `format` Assignment

After conversion, the converter **must** set `DatasetAnnotations.format` to the target format
before passing data to `TargetHandler.write()`. This ensures the target handler interprets
coordinates correctly.

## 4. Category Mapping Rules

### 4.1 YOLO Category Model

- Categories are defined by `classes.txt` (one name per line)
- `class_id` = line number (0-indexed, contiguous)

### 4.2 COCO Category Model

- Categories are defined in the `categories` array
- `category_id` is an **arbitrary integer** (not necessarily 0-based or contiguous)

### 4.3 LabelMe Category Model

- Categories are defined by `shape.label` string values
- Labels map to class_ids through `classes.txt` or auto-assignment

### 4.4 Conversion Rules

| Conversion | Category Handling |
|------------|-------------------|
| YOLO → COCO | Map class_id → COCO category_id (preserve original IDs if available, otherwise use class_id as category_id) |
| COCO → YOLO | Map COCO category_id → 0-based index. Generate `classes.txt` from COCO `categories[].name` sorted by ID |
| LabelMe → YOLO | Map label string → class_id via `classes.txt`. New labels get `len(categories)` as ID |
| YOLO → LabelMe | Map class_id → label string via `classes.txt` |
| LabelMe → COCO | Map label string → COCO category_id via `classes.txt` ordering |
| COCO → LabelMe | Map COCO category_id → label string. Generate `classes.txt` from COCO categories |

## 5. File Structure Changes

### 5.1 YOLO ↔ COCO

- **YOLO → COCO**: Many `.txt` files → Single `.json` file
- **COCO → YOLO**: Single `.json` file → Many `.txt` files in `labels/` directory + `classes.txt`

### 5.2 YOLO ↔ LabelMe

- **YOLO → LabelMe**: Many `.txt` files → Many `.json` files. Each `.txt` produces a `.json` with the same stem.
- **LabelMe → YOLO**: Many `.json` files → Many `.txt` files in `labels/` directory + `classes.txt`. Images are copied to `images/` directory.

### 5.3 COCO ↔ LabelMe

- **COCO → LabelMe**: Single `.json` file → Many `.json` files (one per image). `classes.txt` generated from COCO categories.
- **LabelMe → COCO**: Many `.json` files → Single `.json` file.

## 6. RLE Handling

### 6.1 RLE Encoding (non-COCO → COCO)

When `do_rle=True`:
- Polygon points are encoded to RLE format using pycocotools
- RLE involves **accuracy loss**: polygon vertices are rasterized to a binary mask, then RLE-encoded
- For lossless conversion, use `do_rle=False` (default) to preserve polygon format
- Crowd annotations (`iscrowd=1`) always use RLE format regardless of `do_rle` setting

### 6.2 RLE Decoding (COCO → non-COCO)

- If pycocotools is available: RLE is decoded to polygon contours
- If pycocotools is not available: RLE data is preserved as-is but no polygon is generated
- **Latin1 encoding is mandatory** for RLE `counts` byte↔string conversion (UTF-8 cannot represent all 256 byte values)

## 7. Crowd Annotations (COCO-Specific)

### 7.1 Behavior During Conversion

| Direction | `iscrowd=1` Handling |
|-----------|---------------------|
| COCO → YOLO | `is_crowd` flag preserved in `ObjectAnnotation.is_crowd` |
| COCO → LabelMe | `is_crowd` flag preserved but has no LabelMe equivalent |
| YOLO → COCO | Crowd not natively represented, default `iscrowd=0` |
| LabelMe → COCO | Crowd not natively represented, default `iscrowd=0` |

### 7.2 RLE for Crowd

- COCO crowd annotations must use RLE segmentation
- When writing COCO with `iscrowd=1`, the handler forces RLE output regardless of `do_rle` setting
- Non-crowd annotations follow the `do_rle` flag

## 8. Strict Mode vs Non-Strict Mode

| Mode | Invalid Annotation | Invalid Category | Invalid Image | Invalid Coordinate |
|------|-------------------|-----------------|---------------|-------------------|
| **Strict** (`strict_mode=True`) | Abort with error | Abort with error | Skip with warning (always) | Abort with error |
| **Non-Strict** (`strict_mode=False`) | Skip annotation, continue | Skip annotation, continue | Skip with warning (always) | Skip annotation, continue |

**Image errors** (missing file, unreadable, invalid dimensions) are **always treated as warnings** regardless of strict mode.

## 9. Precision and Round-Trip Fidelity

### 9.1 Precision Classification

Each conversion falls into one of three precision categories:

| Category | Coordinate Space | Conversions |
|----------|----------------|-------------|
| **Lossless** | No coordinate transform | COCO ↔ LabelMe (same space, no normalization) |
| **Near-lossless** — within integer rounding | Absolute → absolute with arithmetic | COCO → LabelMe → COCO (same absolute space) |
| **Lossy** — ±1 pixel or more | Normalized ↔ Absolute or RLE involved | Any conversion involving YOLO; any conversion involving RLE |

**Important**: Even COCO ↔ LabelMe is not strictly byte-for-byte identical after a round-trip,
because intermediate floating-point arithmetic may shift a coordinate by < 1 pixel.
The difference is bounded by the precision of the internal data type (Python float, ~15 decimal digits).

### 9.2 Why Cross-Format Conversions Are Lossy

**Normalized → Absolute → Normalized (e.g., YOLO ↔ anything):**

```
YOLO:  cx = 0.523456 (normalized)
  ↓ denormalize
       cx_abs = 0.523456 × 1920 = 1005.03552
  ↓ store as pixel (int or float)
       cx_abs → approximately 1005.03552 (stored as float)
  ↓ re-normalize
       cx_norm = 1005.03552 / 1920 ≈ 0.523456  (may differ due to float rounding)
```

The round-trip `normalize → denormalize → normalize` introduces ±~1e-6 relative error,
which translates to ±1 pixel for typical image sizes. Two passes through the conversion
doubles the error.

**RLE → Polygon → RLE:**

```
Polygon → fillPoly (rasterize) → mask (binary) → encode → RLE
RLE → decode → mask → findContours → Polygon (approximated)
```

Rasterization loses polygon vertex precision. The decoded polygon is an approximation
of the binary mask, which is already an approximation of the original polygon.

### 9.3 Round-Trip Fidelity Matrix

**Same-format round-trips** (A→A) are fully lossless:

| Round-Trip | Fidelity | Reason |
|------------|----------|--------|
| YOLO → YOLO | **Lossless** | Coordinates stay native; no transform |
| COCO → COCO | **Lossless** | Coordinates stay native; no transform |
| LabelMe → LabelMe | **Lossless** | Coordinates stay native; no transform |

**Cross-format round-trips** (A→B→A) are lossy — the matrix shows expected fidelity:

| Round-Trip | Detection | Segmentation (Polygon) | Segmentation (RLE) |
|------------|-----------|----------------------|-------------------|
| YOLO → COCO → YOLO | **Lossy** (±1 px) | **Lossy** (±1 px per vertex) | N/A |
| COCO → YOLO → COCO | **Lossy** (±1 px) | **Lossy** (±1 px per vertex) | **Lossy** (rasterization loss) |
| YOLO → LabelMe → YOLO | **Lossy** (±1 px) | **Lossy** (±1 px per vertex) | N/A |
| LabelMe → YOLO → LabelMe | **Lossy** (±1 px) | **Lossy** (±1 px per vertex) | N/A |
| COCO → LabelMe → COCO | **Near-lossless** (< 1 px) | **Near-lossless** (< 1 px) | **Lossy** (rasterization loss) |
| LabelMe → COCO → LabelMe | **Near-lossless** (< 1 px) | **Near-lossless** (< 1 px) | N/A |

**Notes:**
- "±1 px" means individual coordinate values may differ by up to 1 pixel from the original
- "Near-lossless" means differences are bounded by floating-point precision of Python float
  (~1e-15 relative), which for typical image sizes is well under 1 pixel
- RLE-based round-trips always have additional rasterization loss (polygon → mask → polygon)

## 10. Error Handling

### 10.1 Error Categories

| Category | Description | Example |
|----------|-------------|---------|
| **Input validation** | Source path doesn't exist, missing required params | `class_file` not provided for YOLO→COCO |
| **Read error** | Source format is invalid or corrupted | Invalid JSON, wrong number of tokens in YOLO line |
| **Conversion error** | Data can't be mapped between formats | Unknown category, unsupported shape type |
| **Write error** | Target can't be written | Permission denied, disk full |
| **State leakage** | Converter state not cleaned up after error | `_source_annotations_for_target` not cleared |

### 10.2 Error Recovery

- Converters use `try/finally` to guarantee state cleanup
- `write_result.errors` are collected and reported in `ConversionResult.errors`
- Non-strict mode allows processing to continue past recoverable errors

## 11. Parameter Reference

| Parameter | Required For | Description |
|-----------|-------------|-------------|
| `class_file` | YOLO→COCO, LabelMe→YOLO, YOLO→LabelMe, LabelMe→COCO | Path to `classes.txt` |
| `image_dir` | YOLO→COCO, YOLO→LabelMe | Directory containing image files |
| `do_rle` | YOLO→COCO, LabelMe→COCO (optional) | Enable RLE encoding for segmentation |
| `strict_mode` | All (optional, default=True) | Enable strict validation |
