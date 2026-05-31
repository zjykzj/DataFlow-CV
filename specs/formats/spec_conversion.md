# Annotation Format Conversion Specification

> **Version:** 1.0
> **Status:** Canonical — this document defines the authoritative conversion rules between YOLO, COCO, and LabelMe formats for DataFlow-CV.

## 1. Conversion Architecture

### 1.1 Universal Pipeline

All conversions follow the same pipeline:

```
Source Format → Handler.read() → DatasetAnnotations → Converter.convert_annotations() → Target Handler.write() → Target Format
```

### 1.2 Internal Data Model as Canonical Representation

The `DatasetAnnotations` data model is the **universal intermediate representation**. All source formats are normalized into this model on read, and all target formats are denormalized from it on write.

**Key properties of the internal model**:
- All coordinates are **normalized [0, 1]**
- `BoundingBox.(x, y)` is the **center** of the box (YOLO convention)
- `Segmentation.points` are normalized polygon vertices
- Categories are stored as `Dict[int, str]` (ID → name)

### 1.3 Lossless Round-Trip Priority

When writing to a target format, the handler checks for original data in this order:

1. **Original data match** (highest priority): If the object has `OriginalData` matching the target format, use it directly (with updated IDs/labels as needed). This preserves exact coordinate precision and format-specific fields.
2. **Convert from internal model**: If no original data exists, convert from the normalized internal model to the target format. This may introduce minor floating-point precision changes.

### 1.4 Converter State Management

Converters store `self._source_annotations_for_target` during the conversion pipeline. This state is used by `create_target_handler()` for:
- Generating `classes.txt` from source categories (COCO→YOLO, COCO→LabelMe)
- Copying image files (LabelMe→YOLO)

**Critical**: This state must be cleared in a `try/finally` block to prevent stale state leakage on exceptions.

## 2. Supported Conversion Directions

| Direction | Converter Class | CLI Command |
|-----------|----------------|-------------|
| YOLO → COCO | `YoloAndCocoConverter(source_to_target=True)` | `yolo2coco` |
| COCO → YOLO | `YoloAndCocoConverter(source_to_target=False)` | `coco2yolo` |
| LabelMe → YOLO | `LabelMeAndYoloConverter(source_to_target=True)` | `labelme2yolo` |
| YOLO → LabelMe | `LabelMeAndYoloConverter(source_to_target=False)` | `yolo2labelme` |
| LabelMe → COCO | `CocoAndLabelMeConverter(source_to_target=False)` | `labelme2coco` |
| COCO → LabelMe | `CocoAndLabelMeConverter(source_to_target=True)` | `coco2labelme` |

## 3. Coordinate Transformation Rules

### 3.1 YOLO ↔ COCO

#### YOLO → COCO (bbox)

```
Input (YOLO):  x_center_norm, y_center_norm, width_norm, height_norm  [0..1, center-based]
Output (COCO): [x_tl, y_tl, w_abs, h_abs]                            [absolute px, top-left]

Step 1: Denormalize to absolute pixels (center-based)
  cx = x_center_norm * img_width
  cy = y_center_norm * img_height
  w  = width_norm * img_width
  h  = height_norm * img_height

Step 2: Convert center → top-left
  x_tl = cx - w / 2
  y_tl = cy - h / 2

Step 3: Output COCO bbox
  bbox = [x_tl, y_tl, w, h]
```

**Implementation**: Use `BoundingBox.xyxy(img_w, img_h)` → `(x1, y1, x2, y2)`, then COCO bbox = `[x1, y1, x2-x1, y2-y1]`.

**Do NOT use** `BoundingBox.xywh_abs()` for COCO output — it returns center-based coordinates, causing systematically offset bboxes.

#### COCO → YOLO (bbox)

```
Input (COCO):  [x_tl, y_tl, w_abs, h_abs]                                  [absolute px, top-left]
Output (YOLO): x_center_norm, y_center_norm, width_norm, height_norm        [0..1, center-based]

Step 1: Compute center in absolute pixels
  cx = x_tl + w_abs / 2
  cy = y_tl + h_abs / 2

Step 2: Normalize
  x_center_norm = cx / img_width
  y_center_norm = cy / img_height
  width_norm     = w_abs / img_width
  height_norm    = h_abs / img_height
```

#### YOLO ↔ COCO (segmentation/polygon)

```
YOLO → COCO:   abs_x = x_norm * img_width;  abs_y = y_norm * img_height
               Output as [[abs_x1, abs_y1, abs_x2, abs_y2, ...], ...]

COCO → YOLO:   norm_x = x_abs / img_width;  norm_y = y_abs / img_height
               Output as class_id norm_x1 norm_y1 norm_x2 norm_y2 ...
```

### 3.2 YOLO ↔ LabelMe

#### YOLO → LabelMe (bbox → rectangle)

```
Input (YOLO):  x_center_norm, y_center_norm, width_norm, height_norm
Output (LabelMe): [[x1, y1], [x2, y2]]  (2 corner points, absolute px)

Use BoundingBox.xyxy(img_w, img_h) → (x1, y1, x2, y2)
points = [[float(x1), float(y1)], [float(x2), float(y2)]]
shape_type = "rectangle"
```

#### LabelMe → YOLO (rectangle → bbox)

```
Input (LabelMe): [[x1, y1], [x2, y2]]  (absolute px, 2 corners)

Step 1: Handle corner-order agnosticism
  x_min = min(x1, x2);  x_max = max(x1, x2)
  y_min = min(y1, y2);  y_max = max(y1, y2)

Step 2: Compute center
  x_center = ((x_min + x_max) / 2) / img_width
  y_center = ((y_min + y_max) / 2) / img_height

Step 3: Compute normalized size
  width  = (x_max - x_min) / img_width
  height = (y_max - y_min) / img_height
```

#### YOLO ↔ LabelMe (segmentation/polygon)

```
YOLO → LabelMe:  abs_pts = [(x*W, y*H) for (x,y) in points]
                 shape_type = "polygon"

LabelMe → YOLO:  norm_pts = [(x/W, y/H) for (x,y) in points]
```

### 3.3 COCO ↔ LabelMe

Both use absolute pixels, but with different representations:

#### COCO → LabelMe (bbox → rectangle)

```
Input (COCO): [x_tl, y_tl, w, h]
Output (LabelMe): [[x_tl, y_tl], [x_tl+w, y_tl+h]]

Convert internally: COCO bbox → normalized center → denormalize to LabelMe rectangle corners
```

#### LabelMe → COCO (rectangle → bbox)

```
Input (LabelMe): [[x1, y1], [x2, y2]]
Output (COCO): [min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)]
```

#### COCO ↔ LabelMe (segmentation/polygon)

```
COCO polygon [[x1,y1,x2,y2,...], ...] ↔ LabelMe [[x1,y1], [x2,y2], ...]

Flatten/unflatten coordinate lists between the two formats.
```

## 4. Category Mapping Rules

### 4.1 YOLO Category Model

- Categories are defined by `classes.txt` (one name per line)
- `class_id` = line number (0-indexed, contiguous)
- Example: `classes.txt` line 0 → class_id=0, line 1 → class_id=1

### 4.2 COCO Category Model

- Categories are defined in the `categories` array
- `category_id` is an **arbitrary integer** (not necessarily 0-based or contiguous)
- Example: COCO uses IDs like 1, 2, 3, ...

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

### 6.1 RLE Conversion (YOLO → COCO / LabelMe → COCO)

When `do_rle=True`:
- Polygon points are encoded to RLE format using pycocotools
- RLE involves **accuracy loss**: polygon vertices are rasterized to a binary mask, then RLE-encoded
- For lossless conversion, use `do_rle=False` (default) to preserve polygon format
- Crowd annotations (`iscrowd=1`) always use RLE format regardless of `do_rle` setting

### 6.2 RLE Decoding (COCO → YOLO / COCO → LabelMe)

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

### 9.1 Precision Levels

| Path | Precision |
|------|-----------|
| Original data write path | Exact (byte-for-byte reproduction) |
| Internal model conversion | Float precision (`.6f` for YOLO, `float` for JSON) |

### 9.2 Known Precision Issues

1. **YOLO ↔ COCO bbox**: Converting normalized center coordinates to absolute top-left and back may introduce ±1 pixel offset due to integer rounding.
2. **RLE conversion**: Polygon→RLE→Polygon is NOT lossless — the RLE mask rasterization introduces accuracy loss.
3. **Polygon coordinate order**: The internal model stores all polygon coordinates from a single flattened list. Multi-polygon COCO annotations are merged into one polygon during reading and separated again on write only if original data is preserved.

### 9.3 Round-Trip Fidelity Matrix

| Round-Trip | Detection | Segmentation (Polygon) | Segmentation (RLE) |
|------------|-----------|----------------------|-------------------|
| YOLO → YOLO | **Lossless** | **Lossless** | N/A |
| COCO → COCO | **Lossless** | **Lossless** | **Lossless** |
| LabelMe → LabelMe | **Lossless** | **Lossless** | N/A |
| YOLO → COCO → YOLO | **Lossless** (original data preserved) | **Lossless** | N/A |
| COCO → YOLO → COCO | **Lossless** (original data preserved) | **Lossless** | Float precision |
| LabelMe → YOLO → LabelMe | **Lossless** (original data preserved) | **Lossless** | N/A |

Lossless round-trips are achieved via `OriginalData` preservation. When original data is not available, conversion goes through the internal model with float precision.

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
| `verbose` | All (optional, default=False) | Enable verbose file logging |
