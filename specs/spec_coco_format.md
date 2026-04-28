# COCO JSON Annotation Format Specification

> **Role:** Single source of truth for COCO `.json` annotation format in DataFlow-CV. All handlers, converters, and visualizers MUST conform to this spec.

## 1. File Overview

COCO annotations are stored as a **single JSON file** that aggregates all annotations for an entire dataset.

| Property | Specification |
|----------|---------------|
| File extension | `.json` |
| Encoding | UTF-8 |
| Structure | Single JSON object containing `info`, `images`, `annotations`, `categories`, and `licenses` arrays |
| File naming | Arbitrary; typically `instances_<subset>.json` (e.g., `instances_train2017.json`, `coco_annotations.json`) |

**Aggregation model:**

```
coco.json = {
    "info": {...},
    "licenses": [...],
    "images": [img_1, img_2, ..., img_N],
    "annotations": [ann_1, ann_2, ..., ann_M],
    "categories": [cat_1, cat_2, ..., cat_K]
}
```

Unlike YOLO and LabelMe (one file per image), COCO uses one file for the entire dataset. Each image is referenced by a unique integer `id`, and annotations link to images via `image_id`.

## 2. Top-Level Fields

Every COCO JSON file MUST contain the following top-level keys:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `info` | `object` | Optional | Dataset-level metadata (description, version, year, contributor, date_created). |
| `licenses` | `array` | Optional | License metadata for images, referenced via `images[].license`. |
| `images` | `array` | **Required** | List of all images in the dataset. Each element is an image object (see §3). |
| `annotations` | `array` | **Required** | List of all annotation instances across all images. Each element is an annotation object (see §4). An empty array `[]` indicates a dataset with no annotations. |
| `categories` | `array` | **Required** | List of all category/class definitions. Each element is a category object (see §5). |

### 2.1 The `info` Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `description` | `string` | Optional | Human-readable description of the dataset. |
| `url` | `string` | Optional | URL to the dataset homepage. |
| `version` | `string` | Optional | Dataset version string. |
| `year` | `integer` | Optional | Year of dataset creation. |
| `contributor` | `string` | Optional | Name of the dataset contributor. |
| `date_created` | `string` | Optional | Creation date in ISO 8601 or `YYYY-MM-DD` format. |

### 2.2 The `licenses` Array

Each element is a license object:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `integer` | **Required** | Unique license identifier. |
| `name` | `string` | **Required** | License name (e.g., `"CC BY 4.0"`). |
| `url` | `string` | Optional | URL to the license text. |

## 3. `images` Array — Image Metadata

Each object describes exactly one image in the dataset.

### 3.1 Image Object Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `integer` | **Required** | Unique image identifier across the dataset. Used as the foreign key from `annotations[].image_id`. |
| `file_name` | `string` | **Required** | Image filename, typically including the extension (e.g., `"scene.jpg"`, `"frame_001.png"`). |
| `width` | `integer` | **Required** | Image width in pixels. Serves as the authoritative reference for coordinate interpretation (see §7). |
| `height` | `integer` | **Required** | Image height in pixels. Same semantics as `width`. |
| `license` | `integer` | Optional | License ID referencing `licenses[].id`. |
| `coco_url` | `string` | Optional | URL to the image on the COCO server. |
| `flickr_url` | `string` | Optional | URL to the image on Flickr. |
| `date_captured` | `string` | Optional | Capture date in ISO 8601 format. |

### 3.2 ID Namespace

```
images[].id ←── annotations[].image_id
```

- Each `image_id` in `annotations` MUST have a corresponding entry in `images`.
- Image IDs are in an independent namespace from annotation IDs and category IDs — `images[].id = 1` and `annotations[].id = 1` are independent (they do not collide).

## 4. `annotations` Array — Annotation Instances

Each object describes exactly one annotated instance. One image may have zero or more annotations.

### 4.1 Annotation Object Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `integer` | **Required** | Unique annotation identifier across the dataset. |
| `image_id` | `integer` | **Required** | Foreign key — MUST match an `images[].id`. |
| `category_id` | `integer` | **Required** | Foreign key — MUST match a `categories[].id`. |
| `bbox` | `[float, float, float, float]` | **Required** | Bounding box in `[x, y, width, height]` format. Origin `(x, y)` is the **top-left corner** of the bounding box. All four values are in **absolute pixel** units. |
| `segmentation` | `array` or `object` | **Required** | Polygon list or RLE dict (see §6). Use an empty list `[]` for detection-only annotations. |
| `area` | `float` | **Required** | Area measured in squared pixels. For detection: `width × height` of the bbox. For segmentation: actual mask area. |
| `iscrowd` | `integer` (`0` or `1`) | **Required** | `0` = single instance; `1` = crowded / dense region containing multiple objects (typically paired with RLE segmentation). |

### 4.2 The `bbox` Field — Detailed Definition

This is the single most important definition in the spec:

> The COCO bounding box is defined by `[x, y, width, height]` where `(x, y)` is the **top-left corner** of the box, measured in **absolute pixel coordinates**.

| Index | Meaning | Unit |
|-------|---------|------|
| 0 — `x` | X-coordinate of the **top-left corner** | Pixels |
| 1 — `y` | Y-coordinate of the **top-left corner** | Pixels |
| 2 — `width` | Width of the bounding box | Pixels |
| 3 — `height` | Height of the bounding box | Pixels |

**Pixel-to-normalized conversion** (used internally by DataFlow-CV):

Given an image of dimensions `W` × `H` pixels, and a COCO bbox `[x, y, w, h]`:

```
x_center  = (x + w / 2) / W
y_center  = (y + h / 2) / H
width_n   = w / W
height_n  = h / H
```

The internal data model stores coordinates in normalized center-origin format `(x_center, y_center, width, height)`.

**Normalized-to-pixel conversion** (for writing COCO):

Given internal normalized values and image dimensions `W` × `H`:

```
x = (x_center - width_n / 2) × W
y = (y_center - height_n / 2) × H
w = width_n × W
h = height_n × H
```

### 4.3 The `iscrowd` Field

The `iscrowd` flag (COCO-specific) distinguishes between:

| `iscrowd` | Meaning | Typical `segmentation` |
|-----------|---------|----------------------|
| `0` | Single, well-defined instance | Polygon (`array`) |
| `1` | Crowded region with overlapping or indistinguishable objects | RLE (`object`) |

- `iscrowd = 0` is the default for most annotations.
- `iscrowd = 1` annotations typically use RLE segmentation, as polygon boundaries are impractical for dense crowds.
- When writing, crowd annotations (`iscrowd = 1`) ALWAYS use RLE format for segmentation, regardless of the `output_rle` setting.

### 4.4 Validation Constraints

A compliant parser MUST reject (or warn, depending on strictness mode) any annotation meeting these conditions:

- `id` is not a valid integer.
- `image_id` does not reference an existing entry in `images`.
- `category_id` does not reference an existing entry in `categories`.
- `bbox` has fewer than 4 elements, or any bbox value is negative.
- `bbox` has zero area (`width <= 0` or `height <= 0`).
- `segmentation` is in polygon format with an odd number of coordinates in any ring.
- `iscrowd` is not `0` or `1`.

## 5. `categories` Array — Class Definitions

Each object defines one class/category.

### 5.1 Category Object Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `integer` | **Required** | Unique category identifier. Unlike YOLO, this is NOT required to be zero-based or contiguous. It serves as the foreign key from `annotations[].category_id`. |
| `name` | `string` | **Required** | Human-readable category name (e.g., `"person"`, `"bicycle"`). Must be non-empty after trimming whitespace. |
| `supercategory` | `string` | Optional | Parent category name for hierarchy grouping (e.g., `"animal"`, `"vehicle"`, `"person"`). Defaults to `"none"` if absent. |

### 5.2 Category ID Semantics

This is a critical distinction from YOLO:

> COCO `category_id` is a **named integer identifier** that does NOT have to follow a zero-based contiguous sequence. The integer is arbitrary — the mapping to class name is explicit in the `categories` array.

| Format | Class Identifier | Mapping |
|--------|-----------------|---------|
| YOLO | `class_id` = line index in `classes.txt` (0-based, contiguous) | Implicit by position |
| COCO | `category_id` = arbitrary integer | Explicit via `categories[].id` → `categories[].name` |
| LabelMe | `label` = human-readable string | Direct (no mapping needed) |

### 5.3 Referential Integrity

```
categories[].id ←── annotations[].category_id
```

Every `category_id` in `annotations` MUST have a corresponding entry in `categories`.

## 6. The `segmentation` Field

The `segmentation` field accepts two mutually exclusive encodings:

### 6.1 Polygon Format (`array`)

A list of flattened coordinate arrays. Each inner array represents one polygon ring.

- **Shape:** `[[x1, y1, x2, y2, ..., xn, yn], ...]`
- **Single polygon:** Outer array has exactly one element (a single ring).
- **Multi-polygon** (holes / disconnected parts): Outer array has multiple elements (multiple rings).
- **Coordinate system:** Absolute pixel coordinates; origin at top-left.
- **Closure:** Polygons are implicitly closed — the first vertex connects to the last vertex. The array SHOULD NOT duplicate the first point at the end.
- **Minimum vertices:** Each inner array MUST have at least 6 values (3 coordinate pairs). An array with fewer than 6 values describes a degenerate polygon and MUST be rejected.

**Example — single polygon (rectangle with 4 vertices):**

```json
"segmentation": [[100, 80, 300, 80, 300, 320, 100, 320]]
```

This describes a rectangular region with top-left at `(100, 80)` and bottom-right at `(300, 320)`.

**Example — multi-polygon with a hole:**

```json
"segmentation": [
  [100, 80, 300, 80, 300, 320, 100, 320],
  [150, 150, 250, 150, 250, 250, 150, 250]
]
```

### 6.2 RLE Format (`object`)

Run-Length Encoding for densely packed mask data. **Requires optional dependency `pycocotools` for encoding/decoding.**

- **Shape:** `{"size": [height, width], "counts": "<compressed_string>"}`
- `size`: Array of two integers `[height, width]` specifying the mask dimensions in pixels.
- `counts`: RLE-encoded byte string (COCO-compressed format), representing a row-major scan of the binary mask. When stored in JSON, this is a UTF-8 string decoded from the raw bytes.

**Example:**

```json
"segmentation": {
  "size": [480, 640],
  "counts": "oZ\\\\^Tmi1j6;g1mN4b3MPZ..."
}
```

**RLE requirements and behavior:**

| Constraint | Rule |
|------------|------|
| Crowd annotations (`iscrowd = 1`) | RLE is the preferred and recommended encoding. When writing, crowd annotations MUST use RLE format, regardless of the `output_rle` setting. |
| Single instance (`iscrowd = 0`) | Polygon format is preferred. RLE is used only when `output_rle = True`. |
| `pycocotools` unavailable | RLE encoding/decoding is not possible. Implementers MUST fall back to polygon format and surface a warning. RLE data that is already present is preserved in the `Segmentation.rle` field without decoding. |

**RLE ↔ polygon conversion:**

- RLE → polygon: Uses `pycocotools.mask.decode()` to produce a binary mask, then extracts contours with OpenCV. The result is an approximate polygon — boundaries may appear pixelated. Polygon format is the lossless source when available.
- Polygon → RLE: Creates a binary mask from polygon points, then encodes with `pycocotools.mask.encode()`. The resulting `counts` bytes are decoded to a UTF-8 string for JSON serialization.

## 7. Coordinate System

### 7.1 Absolute Pixel Coordinates (External / File Format)

**All coordinates in the COCO JSON file are in absolute pixel units.** The coordinate system is defined as follows:

| Property | Definition |
|----------|------------|
| Origin | **Top-left corner** of the image: `(0, 0)`. |
| X-axis | Horizontal, **increasing to the right**. Range: `[0, imageWidth]`. |
| Y-axis | Vertical, **increasing downward**. Range: `[0, imageHeight]`. |
| Unit | **Pixels** (non-negative real numbers). |

```
(0,0) ──────────── X+
  │
  │    image
  │
  Y+
```

### 7.2 Bounding Box Semantics (Critical)

COCO's bounding box definition differs from both YOLO and LabelMe:

| Aspect | COCO | YOLO | LabelMe |
|--------|------|------|---------|
| Bounding box definition | `[x, y, width, height]` — top-left corner + size | `(x_center, y_center, width, height)` — center + size | `[[x_min, y_min], [x_max, y_max]]` — two corners |
| Coordinate type | Absolute pixels | Normalized `[0, 1]` | Absolute pixels |
| Origin | Top-left | Top-left | Top-left |

### 7.3 Internal Data Model (Normalized)

DataFlow-CV internally normalizes all COCO coordinates to `[0, 1]` using a center-origin format identical to YOLO. The conversion is:

**COCO bbox → Internal model:**

```
x_center  = (bbox[0] + bbox[2] / 2) / imageWidth
y_center  = (bbox[1] + bbox[3] / 2) / imageHeight
width_n   = bbox[2] / imageWidth
height_n  = bbox[3] / imageHeight
```

**Internal model → COCO bbox:**

```
bbox[0] = (x_center - width_n / 2) × imageWidth
bbox[1] = (y_center - height_n / 2) × imageHeight
bbox[2] = width_n × imageWidth
bbox[3] = height_n × imageHeight
```

**COCO polygon → Internal model:**

Each polygon vertex `(x_abs, y_abs)` is normalized:

```
x_n = x_abs / imageWidth
y_n = y_abs / imageHeight
```

**Internal model → COCO polygon:**

```
x_abs = x_n × imageWidth
y_abs = y_n × imageHeight
```

### 7.4 Coordinate Bounds

A point `(x, y)` in pixel coordinates is considered within the image boundary if:

```
0 ≤ x ≤ imageWidth
0 ≤ y ≤ imageHeight
```

- Coordinates outside this range describe regions beyond the image boundary. Consumers MAY accept, clamp, or reject such points depending on strictness configuration.
- A bounding box is fully contained if:
  ```
  0 ≤ bbox[0]            (left edge within image)
  bbox[0] + bbox[2] ≤ W  (right edge within image)
  0 ≤ bbox[1]            (top edge within image)
  bbox[1] + bbox[3] ≤ H  (bottom edge within image)
  ```

### 7.5 Common Misinterpretations to Avoid

| Incorrect Interpretation | Correct Interpretation |
|--------------------------|----------------------|
| Bounding box `(x, y)` = center point (like YOLO) | `(x, y)` = **top-left corner** of the box |
| `(x, y)` = bottom-left corner (Cartesian convention) | `(x, y)` = **top-left** corner (screen coordinate convention) |
| Coordinates are normalized `[0, 1]` | Coordinates are **absolute pixels** in the JSON file |
| `category_id` must be zero-based contiguous | `category_id` is an **arbitrary integer** mapped explicitly via `categories` |
| `width`, `height` in bbox are fractions of image dimensions | `width`, `height` in bbox are **absolute pixels** |
| Polygon coordinates are normalized | Polygon coordinates in the JSON are **absolute pixels** |
| `segmentation` is always a polygon list | `segmentation` can be a **polygon list** or an **RLE object** |

## 8. Original Data Preservation

COCO is unique among the three formats in that it stores extra metadata (licenses, URLs, capture dates) beyond basic annotation geometry. DataFlow-CV preserves this data for lossless round-trip conversion.

### 8.1 What Is Preserved

| Data | Preservation Mechanism |
|------|----------------------|
| Full `images[]` entries (including `license`, `coco_url`, `flickr_url`, `date_captured`) | Stored in `ImageAnnotation.original_data.raw_data["image_info"]` |
| Full `annotations[]` entries (including `id`, `area`, `iscrowd`) | Stored in `ObjectAnnotation.original_data.raw_data` |
| Full `categories[]` entries (including `supercategory`) | Stored in `DatasetAnnotations.dataset_info["__coco_original_data__"]["categories"]` |
| Full `info` object | Stored in `DatasetAnnotations.dataset_info["info"]` |
| RLE `segmentation` data | Stored in `Segmentation.rle` |

### 8.2 Round-Trip Priority

When writing COCO output, the handler applies the following priority:

1. **Original COCO data** (if the object came from COCO input): Use the original annotation dict, updating only fields that may have changed (`id`, `image_id`, `category_id`, `iscrowd`). This preserves all original fields losslessly.
2. **Preserved RLE data**: If the segmentation has preserved RLE in `Segmentation.rle`, use it directly.
3. **Encoded RLE** (if `output_rle = True`): Encode polygon points to RLE via `pycocotools`.
4. **Polygon format** (default): Convert internal normalized polygon points to absolute pixel coordinates and write as COCO polygon format.

### 8.3 Round-Trip Limitations

- When `pycocotools` is unavailable, RLE segmentation is preserved but cannot be regenerated if lost.
- RLE → polygon → RLE conversion is lossy; the decoded polygon is an approximation of the original mask boundary.
- Annotation `id` values are re-assigned (sequential starting from 1) during write to avoid conflicts.

## 9. Standard Example

### 9.1 Scenario

| Parameter | Value |
|-----------|-------|
| Image file | `street.jpg` |
| Image dimensions | 640 × 480 pixels |
| Annotation file | `instances_val.json` |
| Dataset | Single image with two annotated objects |

The image contains two annotated objects:

| Object | Pixel Bounding Box `(x_min, y_min, w, h)` | Segmentation | Class | `iscrowd` |
|--------|-------------------------------------------|--------------|-------|-----------|
| Person | (100, 80, 200, 240) | Polygon: rectangle covering the same region | `person` (id=1) | 0 |
| Car | (400, 300, 120, 160) | None (detection only) | `car` (id=2) | 0 |

### 9.2 Coordinate Verification

**Person bbox:** COCO `[x, y, w, h] = [100, 80, 200, 240]`

```
Top-left corner: (100, 80)
Width:            200 px
Height:           240 px
Bottom-right:     (100+200, 80+240) = (300, 320) ✓
Area:             200 × 240 = 48,000 px²
```

**Car bbox:** COCO `[x, y, w, h] = [400, 300, 120, 160]`

```
Top-left corner: (400, 300)
Width:            120 px
Height:           160 px
Bottom-right:     (400+120, 300+160) = (520, 460) ✓
Area:             120 × 160 = 19,200 px²
```

**Internal normalized representation** (center-origin, as stored in `BoundingBox`):

Person:
```
x_center = (100 + 200 / 2) / 640 = 200 / 640 = 0.312500
y_center = (80 + 240 / 2) / 480  = 200 / 480 = 0.416667
width_n  = 200 / 640             = 0.312500
height_n = 240 / 480             = 0.500000
```

Car:
```
x_center = (400 + 120 / 2) / 640 = 460 / 640 = 0.718750
y_center = (300 + 160 / 2) / 480 = 380 / 480 = 0.791667
width_n  = 120 / 640             = 0.187500
height_n = 160 / 480             = 0.333333
```

### 9.3 Complete JSON File

```json
{
  "info": {
    "description": "Example COCO dataset",
    "url": "",
    "version": "1.0",
    "year": 2026,
    "contributor": "",
    "date_created": "2026-04-27"
  },
  "licenses": [
    {
      "id": 1,
      "name": "CC BY 4.0",
      "url": "https://creativecommons.org/licenses/by/4.0/"
    }
  ],
  "images": [
    {
      "id": 1,
      "file_name": "street.jpg",
      "width": 640,
      "height": 480,
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
      "bbox": [100, 80, 200, 240],
      "segmentation": [[100, 80, 300, 80, 300, 320, 100, 320]],
      "area": 48000,
      "iscrowd": 0
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 2,
      "bbox": [400, 300, 120, 160],
      "segmentation": [],
      "area": 19200,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "human"
    },
    {
      "id": 2,
      "name": "car",
      "supercategory": "vehicle"
    }
  ]
}
```

### 9.4 Round-Trip Verification

To verify, reconstruct the pixel bounding box for the first annotation from its COCO bbox:

```
x      = bbox[0] = 100 ✓
y      = bbox[1] = 80 ✓
width  = bbox[2] = 200 ✓
height = bbox[3] = 240 ✓
```

Reconstruct the polygon rectangle:

```
Top-left:     (100, 80)   = polygon[0], polygon[1]   ✓
Top-right:    (300, 80)   = polygon[2], polygon[3]   ✓
Bottom-right: (300, 320)  = polygon[4], polygon[5]   ✓
Bottom-left:  (100, 320)  = polygon[6], polygon[7]   ✓
```

Convert to internal normalized form and back:

```
Person bbox COCO → internal → COCO:

x_center = (100 + 100) / 640 = 0.3125
y_center = (80 + 120) / 480  = 0.416667
width_n  = 200 / 640         = 0.3125
height_n = 240 / 480         = 0.5

→ COCO: x = (0.3125 - 0.15625) × 640 = 100 ✓
        y = (0.416667 - 0.25) × 480  = 80 ✓
        w = 0.3125 × 640             = 200 ✓
        h = 0.5 × 480                = 240 ✓
```

## 10. RLE Example (Reference)

For completeness, the person annotation expressed with RLE segmentation:

```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [100, 80, 200, 240],
  "segmentation": {
    "size": [480, 640],
    "counts": "cTc^O0[2mO2N2N2N2N2N2N2N2N2N2N2N2N2N1O100O100O100O1O2N1O1O2N1O1O2N1O1O2N1O1O1N2O1N2O1N2O1N2O1N2O2N1O2N2O1N2N2N3M2N3M2N3M3L4L3L4K4K4K4K5J5J6I6I7H8G9GaU^cT\\"
  },
  "area": 48000,
  "iscrowd": 0
}
```

The `counts` string is the UTF-8 decoded form of the COCO-compressed RLE bytes. When reading RLE, the handler converts the string back to bytes for `pycocotools.mask.decode()`. When writing RLE, the handler decodes bytes to a UTF-8 string for JSON serialization.

## 11. Dependency: `pycocotools`

COCO RLE segmentation support requires the optional `pycocotools` package.

| Capability | `pycocotools` Available | `pycocotools` Unavailable |
|------------|------------------------|--------------------------|
| Read RLE segmentation | Decode RLE → binary mask → polygon contours | Preserve RLE in `Segmentation.rle` without decoding; points list is empty |
| Write RLE segmentation | Encode polygon → binary mask → RLE | Fall back to polygon format with a warning |
| Crowd annotations | RLE preserved and usable | RLE preserved but cannot be regenerated if lost |
| Polygon ↔ RLE conversion | Supported | Not supported |

- Install with `pip install dataflow-cv[coco]` or `pip install pycocotools`.
- Without `pycocotools`, the handler gracefully degrades: RLE data is preserved for round-trip but cannot be decoded or encoded.
- All other COCO operations (bbox, polygon segmentation, category management) work without `pycocotools`.
