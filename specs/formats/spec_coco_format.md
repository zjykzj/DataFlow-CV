# COCO Annotation Format Specification

> **Version:** v1.1 | **Last Updated:** 2026-07-30
> **Status:** Canonical — this document defines the authoritative COCO format contract for DataFlow-CV.

## 1. File Organization

### 1.1 File Structure

- **Single `.json` file** aggregating all annotations for a dataset.
- Encoding: **UTF-8**.
- Top-level keys: `info`, `images`, `annotations`, `categories`. Additional keys (`licenses`) are preserved but not required.

### 1.2 JSON Schema (Top-Level)

```json
{
  "info": { ... },
  "licenses": [ ... ],
  "images": [ ... ],
  "annotations": [ ... ],
  "categories": [ ... ]
}
```

## 2. `info` Section (Optional)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `description` | string | Yes | Dataset description |
| `url` | string | No | Dataset URL |
| `version` | string | No | Dataset version |
| `year` | int | No | Creation year |
| `contributor` | string | No | Dataset contributor |
| `date_created` | string | No | Creation date (e.g., "2026-03-22") |

Additional fields are preserved as-is during conversion.

## 3. `images` Array (Required)

Each image object:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | **Yes** | Unique image identifier |
| `file_name` | string | **Yes** | Image file name (may include relative path) |
| `width` | int | **Yes** | Image width in pixels |
| `height` | int | **Yes** | Image height in pixels |
| `license` | int | No | License ID |
| `coco_url` | string | No | COCO image URL |
| `flickr_url` | string | No | Flickr image URL |
| `date_captured` | string | No | Capture date |

**Constraints**:
- `width > 0` and `height > 0`
- `id` must be unique across all images

## 4. `annotations` Array (Required)

Each annotation object:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | **Yes** | Unique annotation identifier |
| `image_id` | int | **Yes** | References `images[].id` |
| `category_id` | int | **Yes** | References `categories[].id` |
| `bbox` | [float×4] | **Recommended** | `[x, y, width, height]` in absolute pixels. If absent but ``segmentation`` is present, the code derives the bbox from the polygon extent automatically. |
| `segmentation` | list or dict | No | Polygon `[[x1,y1,...], ...]` or RLE `{"size":[h,w],"counts":"..."}` |
| `area` | float | **Yes** | Area in square pixels |
| `iscrowd` | int | **Yes** | 0 = single instance, 1 = crowd region |

### 4.1 `bbox` Field

**Critical**: COCO bbox is `[x, y, width, height]` where `(x, y)` is the **top-left corner** in **absolute pixel coordinates**.

```
bbox = [x_top_left, y_top_left, width_pixels, height_pixels]
```

This is **different from YOLO** (center-based, normalized) and **different from the internal data model** (center-based, normalized). The conversion path is:

```
Internal (center, norm) → xyxy() → (x1, y1, x2, y2) → COCO [x1, y1, x2-x1, y2-y1]
```

**Common pitfall**: Using `BoundingBox.xywh_abs()` (which returns center-based coordinates) for COCO output produces systematically offset bboxes. Always use `BoundingBox.xyxy()` and then convert to `[x1, y1, w, h]`.

### 4.2 `segmentation` Field

Two mutually exclusive formats:

#### Polygon Format

```json
"segmentation": [[x1, y1, x2, y2, ..., xn, yn], ...]
```

- A list of polygon coordinate lists (supports multiple polygons per annotation).
- Each polygon is a flattened list of `[x, y]` coordinate pairs in **absolute pixel coordinates**.
- Each polygon must have at least 3 points (6 values).
- For non-crowd annotations (`iscrowd=0`), segmentation is typically polygon format.

#### RLE Format

```json
"segmentation": {
  "size": [height, width],
  "counts": "<RLE-encoded string>"
}
```

- Used for crowd annotations (`iscrowd=1`).
- `size`: `[height, width]` of the binary mask.
- `counts`: RLE-encoded string (latin1 encoding of bytes).

### 4.3 `iscrowd` Field

| Value | Meaning | Segmentation Format |
|-------|---------|-------------------|
| 0 | Single object instance | Polygon (typically) |
| 1 | Crowd / group of objects | RLE |

Crowd annotations (`iscrowd=1`) are always written with RLE segmentation.

### 4.4 `area` Field

- For bbox: `area = width * height`
- For segmentation without bbox: estimated from polygon bounding box

## 5. `categories` Array (Required)

Each category object:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | int | **Yes** | Category identifier (arbitrary integer, not necessarily zero-based or contiguous) |
| `name` | string | **Yes** | Category name |
| `supercategory` | string | No | Parent category name |

**Critical distinction from YOLO**: COCO `category_id` is an **arbitrary integer** — it does NOT need to be zero-based or contiguous. For example, the COCO dataset uses IDs like 1, 2, 3, ... rather than 0, 1, 2.

## 6. Coordinate System

### 6.1 Absolute Pixel Coordinates

All COCO coordinates are in **absolute pixels** with the origin at the **top-left corner**:
- X-axis: increases to the right
- Y-axis: increases downward
- Origin: `(0, 0)` = top-left pixel

### 6.2 Conversion to/from YOLO Normalized Coordinates

COCO absolute pixel coordinates (top-left origin) and YOLO normalized coordinates (center origin) are related by:

```
x_center  = (x_top_left + width / 2) / image_width
y_center  = (y_top_left + height / 2) / image_height
width_n   = width / image_width
height_n  = height / image_height
```

Reverse (YOLO normalized → COCO absolute):

```
x_top_left = x_center * image_width  - (width_n  * image_width)  / 2
y_top_left = y_center * image_height - (height_n * image_height) / 2
width_abs  = width_n  * image_width
height_abs = height_n * image_height
```

These formulas are used by the Convert module. DataFlow-CV stores COCO coordinates in their native absolute-pixel representation — no internal normalization is performed.

### 6.3 Coordinate System Comparison

| Property | COCO | YOLO | LabelMe |
|----------|------|------|---------|
| Bbox origin | Top-left | Center | Top-left + Bottom-right (2 points) |
| Coordinate space | Absolute pixels | Normalized [0,1] | Absolute pixels |
| Bbox representation | `[x, y, w, h]` | `class x y w h` (5 values) | `[[x1,y1], [x2,y2]]` |

## 7. RLE Encoding

### 7.1 RLE Data Format

RLE (Run-Length Encoding) is used for crowd annotations and optionally for instance segmentation:

```json
{
  "size": [height, width],
  "counts": "<binary data encoded as latin1 string>"
}
```

### 7.2 Encoding Rules

- **Write path**: `counts_bytes.decode("latin1")` — converts binary RLE to JSON-safe string
- **Read path**: `counts_str.encode("latin1")` — restores binary RLE for mask decoding
- **Never use UTF-8**: RLE `counts` contains arbitrary byte values (0-255); UTF-8 cannot represent all 256 values and will raise `UnicodeDecodeError`
- Latin1 provides a **lossless 1:1 byte-to-character mapping** that is essential for RLE round-trips

### 7.3 Dependencies

- RLE encoding/decoding requires **pycocotools** (`pip install pycocotools`)
- Without pycocotools: RLE data is preserved as-is but cannot be converted to/from polygon format
- Polygon format (non-RLE) does not require pycocotools

## 8. Validation Rules

A valid COCO JSON file must satisfy:

1. Top-level `images`, `annotations`, `categories` keys are present
2. Each image has `id` and `file_name`
3. Each category has `id` and `name`
4. Each annotation has `id`, `image_id`, `category_id`
5. `image_id` in each annotation references a valid image
6. `category_id` in each annotation references a valid category
7. Bbox `[x, y, w, h]` has `w > 0` and `h > 0`
8. Polygon segmentation has at least 3 vertices (6 values)
9. RLE segmentation has `size` (2-element list) and `counts` (string)

## 9. Prediction JSON (vs Annotation JSON)

COCO JSON files serve two distinct purposes in the DataFlow-CV pipeline:

| Purpose | File | Content |
|---------|------|---------|
| **Annotation** (Ground Truth) | `anno.json` | Human-annotated ground truth labels |
| **Prediction** (Detection Result) | `pred.json` | Model inference output |

### 9.1 Structural Variants

Prediction JSON can appear in **two valid formats**:

**Variant A — Full COCO dict (same structure as annotation JSON):**
- `images` array — same schema as annotation JSON
- `categories` array — same schema as annotation JSON
- `annotations` array — each entry includes `score` (see §9.2)

This format is typical when predictions are generated by Convert module tools (e.g., `yolo2coco --prediction`).

**Variant B — Plain annotation list (JSON array):**

```json
[
  {
    "image_id": 1,
    "category_id": 1,
    "bbox": [320.0, 135.0, 160.0, 180.0],
    "score": 0.95
  },
  ...
]
```

This format is common in model inference output from frameworks like Detectron2, MMDetection, and custom training scripts. It contains only the annotation array — no `images` or `categories` arrays. The `images` and `categories` information is obtained from the GT file at evaluation time via `COCO.loadRes()`.

**Loading strategy**: Variant B must be loaded via `pycocotools.COCO.loadRes()`, which copies `images` and `categories` from the GT and indexes the annotations list. The `COCO()` constructor rejects this format (it requires a dict). See `spec_evaluate.md` §9.1 for the loading pipeline.

### 9.2 The `score` Field

For **Variant A** (full COCO dict), the only structural difference from
annotation JSON is in the ``annotations`` array:

| Field | Annotation JSON | Prediction JSON |
|-------|----------------|----------------|
| `score` | **Not present** | **Required** — float in [0, 1] |

For **Variant B** (plain annotation list), each annotation dict in the
array also requires ``score`` — there is no ``images`` or ``categories``
wrapper at all (see §9.1).

Prediction annotations must include a ``score`` field representing the
model's confidence:

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [320.0, 135.0, 160.0, 180.0],
  "segmentation": [[320, 135, 480, 135, 480, 315, 320, 315]],
  "area": 28800.0,
  "score": 0.95
}
```

### 9.3 Other Differences

| Aspect | Annotation JSON | Prediction JSON |
|--------|----------------|----------------|
| `annotations[].score` | Always absent (must never be included) | **Required** (float [0,1]) |
| `annotations[].iscrowd` | 0 or 1 | Always 0 (crowd is a GT-only concept) |
| Annotation count | Typically matches actual object count | Typically higher (model outputs multiple candidates per image) |
| Evaluation role | Used as GT in `COCOeval` | Used as DT in `COCOeval` |
| Loading method | `COCO(annotation_file)` | `COCO(annotation_file)` if full dict; `COCO.loadRes(annotation_file)` if plain list |

The `loadRes()` method is pycocotools' designated API for loading prediction results. It accepts both a file path (to either a full COCO dict or a plain JSON list) and an in-memory list of annotation dicts. Unlike `COCO()`, it copies `images` and `categories` from the GT dataset and indexes the provided annotation list. This is the reason prediction files do not need to include `images`/`categories` arrays — `loadRes()` sources them from GT. |

### 9.4 Segmentation Predictions

For segmentation predictions, **polygon format is recommended** over RLE:

- pycocotools `COCOeval` internally converts polygon → RLE during evaluation via `annToRLE()`. Pre-converting to RLE in the prediction file provides no evaluation benefit.
- Polygon is **lossless** (preserves original vertex coordinates). RLE involves rasterization loss during polygon→mask conversion.
- The `--do-rle` flag remains available for users who prioritize smaller file size over precision.

## 10. Reference Example

```json
{
  "info": {
    "description": "Example dataset",
    "version": "1.0",
    "year": 2026,
    "contributor": "",
    "date_created": "2026-03-22"
  },
  "images": [
    {
      "id": 1,
      "file_name": "train_001.jpg",
      "width": 800,
      "height": 600,
      "license": 1,
      "coco_url": "",
      "flickr_url": "",
      "date_captured": ""
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [320.0, 135.0, 160.0, 180.0],
      "segmentation": [[320, 135, 480, 135, 480, 315, 320, 315]],
      "area": 28800.0,
      "iscrowd": 0
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 2,
      "bbox": [540.0, 240.0, 120.0, 120.0],
      "area": 14400.0,
      "iscrowd": 0,
      "segmentation": []
    }
  ],
  "categories": [
    { "id": 1, "name": "person", "supercategory": "none" },
    { "id": 2, "name": "car", "supercategory": "none" }
  ]
}
```

**Verification for annotation 1** (person):
- Bbox: top-left=(320, 135), size=160×180 → bottom-right=(480, 315)
- For an 800×600 image, the internal representation is: center=(400/800=0.5, 225/600=0.375), size=(160/800=0.2, 180/600=0.3)
- This matches the YOLO equivalent: `0 0.500000 0.375000 0.200000 0.300000`
