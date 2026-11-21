# LabelMe Annotation Format Specification

> **Version:** v1.3 | **Last Updated:** 2026-11-21
> **Status:** Canonical — this document defines the authoritative LabelMe format contract for DataFlow-CV.

## 1. File Organization

### 1.1 File Structure

- **One `.json` file per image**, using the same stem name as the corresponding image file.
- Example: `image_001.jpg` → `image_001.json`
- By default (`recursive=False`), label files reside in a single flat directory (no subdirectories). When `recursive=True`, the handler uses `rglob` for file discovery, traversing all subdirectories recursively.
- Encoding: **UTF-8**.

### 1.2 Directory Layout

```
dataset/
├── image_001.jpg
├── image_001.json
├── image_002.jpg
├── image_002.json
└── ...
```

Images and labels may be in the same directory or separate directories.

## 2. JSON Structure

### 2.1 Root-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | **Yes** | LabelMe version (e.g., "5.0.1") |
| `flags` | object | **Yes** | Global flags (typically empty `{}`) |
| `shapes` | array | **Yes** | List of annotation shapes |
| `imagePath` | string | **Yes** | Path to the image file (relative to JSON location by default) |
| `imageData` | string\|null | No | Base64-encoded image data (usually `null` for external images) |
| `imageHeight` | int | No | Image height in pixels (inferred from image if missing) |
| `imageWidth` | int | No | Image width in pixels (inferred from image if missing) |

### 2.2 `imagePath` Resolution

- If `imagePath` is a relative path, it is resolved relative to the JSON file's directory.
- If `imagePath` is absolute, it is used as-is.

### 2.3 `imageData`

- **Not required on read**: Valid LabelMe files may omit `imageData` (set to `null`) when images are stored as external files.
- On **write**, `imageData` is always set to `null` — DataFlow-CV does not embed image data.

### 2.4 `imageHeight` / `imageWidth`

- Optional on read. If missing, the handler reads the actual image file to determine dimensions via OpenCV.
- If the image file is not found AND `imageHeight`/`imageWidth` are absent from JSON, an `ImageError` is raised, causing the image to be skipped with a warning.
- If the image file is not found BUT `imageHeight`/`imageWidth` are present in JSON, those dimensions are used — the image file is not required in this case (no warning).
- Both fields are validated to be `> 0` when present.
- On write, both fields are always included with the actual dimensions.

## 3. `shapes` Array

Each shape object represents one annotation:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `label` | string | **Yes** | Class/category name |
| `shape_type` | string | **Yes** | `"rectangle"`, `"polygon"`, or `"point"` (pose keypoint, §3.4) |
| `points` | [[float, float], ...] | **Yes** | List of `[x, y]` coordinate pairs in absolute pixels |
| `group_id` | int\|null | No | Group identifier for related shapes (nullable) |
| `flags` | object | No | Shape-level flags (typically empty `{}`) |
| `difficult` | bool | No | X-AnyLabeling extension on point shapes — marks an occluded keypoint (reads as v=1, §3.5). Written for v=1 on export |

> Other ecosystem extras (`score`, `attributes`, `description` on shapes) are
> ignored on read and not written on export.

### 3.1 Rectangle Shape

- `shape_type`: `"rectangle"`
- `points`: Exactly **2 points** — `[[x1, y1], [x2, y2]]`
- Semantics: The two points define opposite corners of the rectangle (typically top-left and bottom-right, but corner order is not guaranteed).

**Corner-order agnosticism**: The handler uses `min`/`max` to determine the actual bounding box:
```
actual_top_left = (min(x1, x2), min(y1, y2))
actual_bottom_right = (max(x1, x2), max(y1, y2))
width = abs(x2 - x1)
height = abs(y2 - y1)
```

### 3.2 Polygon Shape

- `shape_type`: `"polygon"`
- `points`: At least **3 points** — `[[x1, y1], [x2, y2], ..., [xn, yn]]`
- The polygon is implicitly closed (the last point connects to the first).

### 3.3 Unsupported Shape Types

The following LabelMe shape types are **not supported** by DataFlow-CV and will cause a parse error:
- `circle`
- `line`
- `point` **without** a `group_id` (a point shape with a non-null `group_id` is a pose keypoint — see §3.4)

### 3.4 Point Shape (Pose Keypoint)

- `shape_type`: `"point"`
- `points`: Exactly **1 point** — `[[x, y]]`
- `group_id`: **Required** (non-null) — groups the keypoints of one pose instance
- `label`: The **keypoint name** (e.g., `"nose"`, `"left_eye"`) — not the class name

This is the de-facto ecosystem convention (X-AnyLabeling and third-party pose
converters — the official labelme `labelme2coco` example script handles only
segmentation shapes and has no pose support. X-AnyLabeling:
"group_id 用于关联同一组内的多个对象 (e.g., pose keypoints)"). LabelMe has no native
keypoint shape type; keypoint annotation is performed as per-keypoint point shapes.

### 3.5 Pose Instance Convention

One pose instance = all point shapes sharing a `group_id`, plus optionally one `rectangle`
shape with the same `group_id`:

| Element | Shape | Role |
|---------|-------|------|
| Class + bbox | `rectangle` (same `group_id`) | `label` = class name; provides the instance bbox. **Required** — keypoint labels carry no class information, so a point group without a rectangle is invalid |
| Keypoints | `point` shapes (same `group_id`) | `label` = keypoint name, `points` = one `[x, y]` pair |
| Visibility | Implicit | A point shape present = labeled and visible (v=2); a keypoint absent from the file = not labeled (v=0). v=1 (labeled but occluded) is represented X-AnyLabeling-style via `"difficult": true` on the point shape — written for v=1 on export, read back as v=1. Point shapes without `difficult` read as v=2 |

**Keypoint ordering**: DataFlow-CV orders keypoints by the keypoint-definition file
(`kpts_config`, see `spec_label.md` §4.4): each instance produces exactly `k = len(names)`
keypoints, with absent names becoming v=0 (0, 0) entries. Without a definition file, count
and order follow the shapes' appearance order in the JSON file (with a WARNING that
ordering is file-order-based).

## 4. Coordinate System

### 4.1 Absolute Pixel Coordinates

All LabelMe coordinates are in **absolute pixels** with the origin at the **top-left corner**:
- X-axis: increases to the right
- Y-axis: increases downward
- Origin: `(0, 0)` = top-left pixel

### 4.2 Coordinate System Comparison

| Property | LabelMe | YOLO | COCO |
|----------|---------|------|------|
| Bbox representation | 2 corner points | Center + size (5 values) | `[x, y, w, h]` |
| Coordinate space | Absolute pixels | Normalized [0,1] | Absolute pixels |
| Bbox origin | Two corners (agnostic) | Center | Top-left |
| Polygon format | `[[x,y], ...]` | Flattened on one line | `[[x1,y1,x2,y2,...], ...]` |
| Keypoints | `point` shapes + `group_id` | `5+3k` tokens flattened | `[x1,y1,v1,...]` triplets |

## 5. Category Management

### 5.1 Category Discovery

Categories can be sourced from two places:

1. **`classes.txt` file** (if provided): One class name per line. Line number = class_id (0-indexed). This takes priority.
2. **Auto-extracted from annotations**: If no class file is provided, categories are built from `shape.label` values encountered during reading. New labels are assigned incrementing IDs.

### 5.2 Label to class_id Mapping

During reading:
- If `shape.label` matches a known category name → use its `class_id`
- If `shape.label` is new → assign `class_id = len(categories)` and add to categories

## 6. Validation Rules

A valid LabelMe JSON file must satisfy:

1. Root object contains `version`, `flags`, `shapes`, `imagePath`
2. Each shape in `shapes` has non-empty `label`, valid `shape_type`, and `points`
3. Rectangle shapes have exactly 2 points
4. Polygon shapes have at least 3 points
5. Point shapes have exactly 1 point and a non-null `group_id`
6. Pose instances: every point group has at least 1 labeled point; a group's rectangle (if present) carries the class label
7. `imageHeight` > 0 and `imageWidth` > 0 (if present)
8. `imagePath` points to an existing image file (warning if not found, but processing continues). **Exception**: When the JSON file contains valid `imageHeight` and `imageWidth` fields, the image file itself is not required — its absence in this case produces no warning, as dimensions are read directly from the JSON.

## 7. Reference Example

```json
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [
    {
      "label": "person",
      "points": [[320, 135], [480, 315]],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    },
    {
      "label": "car",
      "points": [[540, 240], [660, 360]],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    },
    {
      "label": "person",
      "points": [[100, 200], [150, 180], [200, 220], [180, 280], [120, 270]],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    },
    {
      "label": "person",
      "points": [[300, 150], [360, 380]],
      "group_id": 1,
      "shape_type": "rectangle",
      "flags": {}
    },
    {
      "label": "nose",
      "points": [[320, 200]],
      "group_id": 1,
      "shape_type": "point",
      "flags": {}
    },
    {
      "label": "left_eye",
      "points": [[310, 190]],
      "group_id": 1,
      "shape_type": "point",
      "flags": {}
    },
    {
      "label": "right_eye",
      "points": [[330, 190]],
      "group_id": 1,
      "shape_type": "point",
      "flags": {}
    }
  ],
  "imagePath": "image_001.jpg",
  "imageData": null,
  "imageHeight": 600,
  "imageWidth": 800
}
```

**Verification for the first rectangle** (person):
- Points: (320, 135) and (480, 315) → opposite corners
- On an 800×600 image:
  - Normalized center: x = ((320+480)/2)/800 = 0.5, y = ((135+315)/2)/600 = 0.375
  - Normalized size: w = |480-320|/800 = 0.2, h = |315-135|/600 = 0.3
- YOLO equivalent: `0 0.500000 0.375000 0.200000 0.300000`
- COCO equivalent bbox: `[320, 135, 160, 180]`
