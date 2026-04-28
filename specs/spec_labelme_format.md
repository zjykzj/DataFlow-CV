# LabelMe JSON Annotation Format Specification

> **Role:** Single source of truth for LabelMe `.json` annotation format in DataFlow-CV. All handlers, converters, and visualizers MUST conform to this spec.

## 1. File Overview

LabelMe annotations are stored as **JSON files**, with one file per image.

| Property | Specification |
|----------|---------------|
| File extension | `.json` |
| Encoding | UTF-8 |
| File naming | `<image_stem>.json` — typically matches the corresponding image file name, excluding the image extension |
| Location | Stored alongside or in a parallel directory to the source image |

**Naming convention:**

```
image_stem = basename_of_image_file_without_extension
label_file  = image_stem + ".json"
```

Example: image `frame_001.jpg` → label `frame_001.json`, image `scene.png` → label `scene.json`.

Each JSON file describes the annotations for **exactly one image**. The file is self-contained: image dimensions, the path to the source image, and all shape annotations are stored within a single document.

## 2. Root-Level Fields

Every LabelMe JSON file MUST contain the following top-level keys:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | `string` | **Required** | LabelMe format version (e.g., `"5.0.1"`). Used for forward/backward compatibility. |
| `flags` | `object` | **Required** | Global flags for the annotation file. Typically an empty object `{}`. Reserved for application-level metadata. |
| `shapes` | `array` | **Required** | List of annotated objects. Each element is a shape object (see §3). An empty array `[]` indicates an image with no annotations. |
| `imagePath` | `string` | **Required** | Relative or absolute path to the source image file. When relative, it is resolved against the directory containing the JSON file. |
| `imageData` | `string` or `null` | **Required** | Base64-encoded image data. Typically `null` in annotation-only workflows. When non-null, it contains the full image encoded as a Base64 string, enabling self-contained distribution without separate image files. |
| `imageHeight` | `integer` | Optional | Image height in pixels. If absent, consumers MUST infer it from the source image or default to a sentinel value with a warning. |
| `imageWidth` | `integer` | Optional | Image width in pixels. Same semantics as `imageHeight`. |

### 2.1 Notes on `imageData`

- **Annotation-only mode:** `imageData` is `null`. The consumer reads the image from `imagePath`.
- **Self-contained mode:** `imageData` contains a Base64-encoded JPEG or PNG image. The consumer can decode it directly without accessing the filesystem.
- Consumers MUST handle both cases gracefully.
- When writing, producers SHOULD set `imageData` to `null` unless explicitly instructed to embed image data.

### 2.2 Notes on `imageHeight` and `imageWidth`

- These fields are present in LabelMe exports starting from version 4.x and later.
- If absent, the consumer MUST attempt to determine dimensions from the source image.
- Parsers MUST treat the dimensions as the authoritative reference for coordinate interpretation (see §4). Any mismatch between these values and the actual image file dimensions SHOULD be surfaced as a warning.

## 3. Shapes Array — Detailed Definition

The `shapes` array is the core of a LabelMe annotation file. Each element is a JSON object representing one annotated region.

### 3.1 Shape Object Structure

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `label` | `string` | **Required** | Human-readable class name for this object (e.g., `"person"`, `"car"`). Must be non-empty after trimming whitespace. |
| `shape_type` | `string` | **Required** | Geometric type of the annotation. Supported values: `"rectangle"`, `"polygon"`. Other values (e.g., `"circle"`, `"line"`, `"point"`) exist in the LabelMe ecosystem but are **not supported** by this spec. |
| `points` | `array` of `[float, float]` | **Required** | List of coordinate pairs. The **semantics depend on `shape_type`** (see §3.2). |
| `group_id` | `integer` or `null` | Optional | Group identifier for associating related shapes. `null` indicates no grouping. |
| `flags` | `object` | Optional | Per-shape flags. Typically an empty object `{}`. Reserved for application-level metadata (e.g., occlusion, difficulty). |

### 3.2 The `points` Field — Shape-Type Semantics

This is the single most important definition in the spec. The meaning of the `points` array is determined by `shape_type`.

#### 3.2.1 `"rectangle"` — Bounding Box via Two Corner Points

When `shape_type` is `"rectangle"`, `points` MUST contain **exactly 2** coordinate pairs:

```
points: [[x_min, y_min], [x_max, y_max]]
```

| Point | Meaning |
|-------|---------|
| `points[0]` — `[x_min, y_min]` | **Top-left corner** of the rectangle. |
| `points[1]` — `[x_max, y_max]` | **Bottom-right corner** of the rectangle. |

**Critical constraints:**

- The two points define the rectangle by its diagonal corners.
- The rectangle edges are **axis-aligned**. Rotation is not represented.
- It is NOT required that `x_min < x_max` or `y_min < y_max`. Consumers MUST handle cases where points are in any order by computing:
  ```
  actual_x_min = min(x_min, x_max)
  actual_y_min = min(y_min, y_max)
  actual_x_max = max(x_min, x_max)
  actual_y_max = max(y_min, y_max)
  ```
  The resulting bounding box is defined by `(actual_x_min, actual_y_min, actual_x_max, actual_y_max)`.
- A rectangle with `x_min == x_max` or `y_min == y_max` has zero area and SHOULD be rejected or warned about.

**Example:** A rectangle from `(100, 80)` to `(300, 320)` (in pixel coordinates):

```json
{
  "label": "person",
  "shape_type": "rectangle",
  "points": [[100, 80], [300, 320]],
  "group_id": null,
  "flags": {}
}
```

#### 3.2.2 `"polygon"` — Instance Segmentation via Vertex List

When `shape_type` is `"polygon"`, `points` MUST contain **at least 3** coordinate pairs:

```
points: [[x1, y1], [x2, y2], ..., [xn, yn]]
```

Each element `[xi, yi]` is a **vertex** of the polygon. Adjacent vertices are connected by straight line segments in the listed order.

**Critical constraints:**

- Minimum vertex count: **3**. A polygon with fewer than 3 points is degenerate and MUST be rejected.
- **Implicit closure:** The polygon is implicitly closed — the last vertex connects back to the first vertex. The array SHOULD NOT duplicate the first point at the end.
- **Self-intersection:** The spec does not define polygon validity rules (convex, simple, non-self-intersecting). Validation is the responsibility of downstream consumers.
- **Holes:** LabelMe does not natively support polygon holes at the shape level. Complex shapes with holes require multiple shapes with shared `group_id`.

**Example:** A triangular polygon with vertices at `(100, 80)`, `(300, 80)`, `(200, 320)`:

```json
{
  "label": "object",
  "shape_type": "polygon",
  "points": [[100, 80], [300, 80], [200, 320]],
  "group_id": null,
  "flags": {}
}
```

### 3.3 Relationship Between Rectangle and Polygon

A `rectangle` shape is semantically equivalent to a `polygon` with 4 vertices at the four corners:

```json
// rectangle form
{"shape_type": "rectangle", "points": [[100, 80], [300, 320]]}

// equivalent polygon form
{"shape_type": "polygon", "points": [[100, 80], [300, 80], [300, 320], [100, 320]]}
```

However, consumers MUST NOT automatically convert between these types. The `shape_type` field carries intentional semantics: `"rectangle"` indicates a detection bounding box, while `"polygon"` indicates a segmentation mask. Converting between them discards this semantic distinction.

## 4. Coordinate System

### 4.1 Absolute Pixel Coordinates

**All `points` values are in absolute pixel coordinates.** The coordinate system is defined as follows:

| Property | Definition |
|----------|------------|
| Origin | **Top-left corner** of the image: `(0, 0)`. |
| X-axis | Horizontal, **increasing to the right**. |
| Y-axis | Horizontal, **increasing downward**. |
| Unit | **Pixels** (non-negative real numbers). |
| Valid X range | `[0, imageWidth]` |
| Valid Y range | `[0, imageHeight]` |

### 4.2 Coordinate Bounds

A point `(x, y)` is considered within the image boundary if:

```
0 ≤ x ≤ imageWidth
0 ≤ y ≤ imageHeight
```

- Coordinates outside this range describe regions beyond the image boundary. Consumers MAY accept, clamp, or reject such points depending on strictness configuration.
- Sub-pixel precision (fractional coordinates, e.g., `100.5`) is valid and MUST be preserved by parsers.

### 4.3 Comparison with Other Formats

| Aspect | LabelMe | YOLO | COCO |
|--------|---------|------|------|
| Coordinate type | Absolute pixels | Normalized [0, 1] | Absolute pixels |
| Bounding box definition | Two corners `(x_min, y_min), (x_max, y_max)` | Center + size `(x_c, y_c, w, h)` | Top-left + size `(x, y, w, h)` |
| Origin | Top-left | Top-left | Top-left |

### 4.4 Common Misinterpretations to Avoid

| Incorrect Interpretation | Correct Interpretation |
|--------------------------|----------------------|
| Origin at **bottom-left** (image coordinate convention) | Origin at **top-left** (screen coordinate convention) |
| Coordinates are normalized [0, 1] | Coordinates are **absolute pixels** |
| `y` increases **upward** | `y` increases **downward** |
| Rectangle points are `(x_center, y_center)` and `(width, height)` | Rectangle points are two **corner** coordinates |

## 5. Standard Example

### 5.1 Scenario

| Parameter | Value |
|-----------|-------|
| Image file | `street.jpg` |
| Image dimensions | 640 × 480 pixels |
| Label file | `street.json` |
| Format version | `5.0.1` |

The image contains two annotated objects:

| Object | Shape Type | Geometry | Class |
|--------|------------|----------|-------|
| Person | `rectangle` | Top-left `(100, 80)`, bottom-right `(300, 320)` | `person` |
| Car | `rectangle` | Top-left `(400, 300)`, bottom-right `(520, 460)` | `car` |

### 5.2 Complete JSON File

```json
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [
    {
      "label": "person",
      "points": [
        [100, 80],
        [300, 320]
      ],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    },
    {
      "label": "car",
      "points": [
        [400, 300],
        [520, 460]
      ],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    }
  ],
  "imagePath": "street.jpg",
  "imageData": null,
  "imageHeight": 480,
  "imageWidth": 640
}
```

### 5.3 Verification

To verify correct interpretation of the `person` shape:

```
shape_type: "rectangle"
points[0]  = [100, 80]    → top-left corner at pixel (100, 80)
points[1]  = [300, 320]   → bottom-right corner at pixel (300, 320)

Bounding box: x ∈ [100, 300], y ∈ [80, 320]
Width:  300 - 100 = 200 px
Height: 320 - 80  = 240 px
Area:   200 × 240 = 48,000 px²
```

### 5.4 Polygon Example (Reference)

For completeness, the same `person` region expressed as a polygon:

```json
{
  "label": "person",
  "points": [
    [100, 80],
    [300, 80],
    [300, 320],
    [100, 320]
  ],
  "group_id": null,
  "shape_type": "polygon",
  "flags": {}
}
```

Note the four vertices tracing the perimeter in clockwise order. The polygon implicitly closes from `[100, 320]` back to `[100, 80]`.
