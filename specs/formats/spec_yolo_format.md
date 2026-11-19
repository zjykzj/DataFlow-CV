# YOLO Annotation Format Specification

> **Version:** v1.2 | **Last Updated:** 2026-11-19
> **Status:** Canonical — this document defines the authoritative YOLO format contract for DataFlow-CV.

## 1. File Organization

### 1.1 File Structure

- **One `.txt` file per image**, using the same stem name as the corresponding image file.
- Example: `image_001.jpg` → `image_001.txt`
- By default (`recursive=False`), label files reside in a single flat directory (no subdirectories). When `recursive=True`, the handler uses `rglob` for file discovery, traversing all subdirectories recursively.
- Encoding: **UTF-8**.

### 1.2 Category Mapping

- **`classes.txt`**: A plain-text file with one class name per line.
- Line number (0-indexed) corresponds to `class_id`.
- Empty lines are skipped during parsing.
- Example:
  ```
  person
  car
  bicycle
  dog
  ```

### 1.3 Directory Layout

```
dataset/
├── images/
│   ├── train_001.jpg
│   ├── train_002.jpg
│   └── ...
├── labels/
│   ├── train_001.txt
│   ├── train_002.txt
│   └── ...
└── classes.txt
```

## 2. Line Format

### 2.1 Object Detection (Bounding Box)

Each line contains exactly **5 tokens** separated by spaces:

```
<class_id> <x_center> <y_center> <width> <height>
```

| Token | Type | Range | Description |
|-------|------|-------|-------------|
| `class_id` | int | 0..N-1 | Zero-based class index into `classes.txt` |
| `x_center` | float | [0, 1] | Normalized x-coordinate of bounding box **center** |
| `y_center` | float | [0, 1] | Normalized y-coordinate of bounding box **center** |
| `width` | float | [0, 1] | Normalized width of bounding box |
| `height` | float | [0, 1] | Normalized height of bounding box |

**Key semantics**: `(x_center, y_center)` is the **center point** of the bounding box, NOT the top-left corner. This is a critical distinction from COCO and LabelMe formats.

### 2.2 Instance Segmentation (Polygon)

Each line contains an **odd number > 5** of tokens:

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

| Token | Type | Range | Description |
|-------|------|-------|-------------|
| `class_id` | int | 0..N-1 | Zero-based class index |
| `x1, y1, ...` | float | [0, 1] | Normalized polygon vertex coordinates |

**Constraints**:
- The polygon must have at least **3 points** (6 coordinate values + class_id = 7 tokens minimum).
- Polygon vertices are stored as flattened (x, y) pairs.
- The polygon is implicitly closed (the last point connects to the first).

### 2.3 Format Detection

Format detection is **mode-sensitive**. The same token count has different meanings depending on whether the file is being read as a label (ground truth) or a prediction (model output). The mode is controlled by a flag (`prediction=False` for labels, `prediction=True` for predictions) and applies to the entire batch of files being processed.

#### Label Mode (`prediction=False`)

| Token Count | Format |
|------------|--------|
| `len == 5` | Detection label |
| `len > 5 AND len % 2 == 1` | Segmentation label |
| Any other | **Invalid** |

#### Prediction Mode (`prediction=True`)

| Token Count | Format |
|------------|--------|
| `len == 6` | Detection prediction |
| `len > 6 AND len % 2 == 0` | Segmentation prediction |
| Any other | **Invalid** |

**Why this is unambiguous**: Label files always have an odd number of tokens (class_id + even number of coordinates). Prediction files always have an even number of tokens (class_id + even number of coordinates + 1 confidence). The two sets are disjoint — no token count can be valid in both modes simultaneously:

| Token Count | Label Mode | Prediction Mode |
|------------|-----------|----------------|
| 5 | Detection label | **Invalid** |
| 6 | **Invalid** | Detection prediction |
| 7 | Segmentation label (3 points) | **Invalid** |
| 8 | **Invalid** | Segmentation prediction (3 points) |
| 9 | Segmentation label (4 points) | **Invalid** |
| 10 | **Invalid** | Segmentation prediction (4 points) |

#### Pose Mode (`pose=True`)

| Token Count | Pose Label Mode | Pose Prediction Mode |
|------------|-----------------|----------------------|
| `5 + 3k` (k ≥ 1) | Pose label (k keypoints) | **Invalid** |
| `6 + 3k` (k ≥ 1) | **Invalid** | Pose prediction (k keypoints) |
| Any other | **Invalid** | **Invalid** |

**Pose mode is explicit, never auto-detected.** A pose row has `5 + 3k` tokens (label) or
`6 + 3k` (prediction); when 3k is even these lengths **collide** with segmentation rows —
e.g., 11 tokens is both a 2-keypoint pose label row and a 3-point segmentation label row,
and a 17-keypoint row (56 tokens) silently parses as a 29-point segmentation row. The
`pose` flag (handler parameter / CLI `--pose`) selects the interpretation; within one
dataset k is fixed — the first valid row sets k, rows with a different k are invalid
(strict: error; non-strict: skip with warning).

### 2.4 Prediction Format

When reading YOLO files in **prediction mode** (`prediction=True`), each line contains an additional trailing `confidence` token beyond the standard label format.

#### 2.4.1 Detection Prediction

Exactly **6 tokens**:

```
<class_id> <x_center> <y_center> <width> <height> <confidence>
```

| Token | Type | Range | Description |
|-------|------|-------|-------------|
| `class_id` | int | 0..N-1 | Zero-based class index |
| `x_center` | float | [0, 1] | Normalized x-coordinate of bbox center |
| `y_center` | float | [0, 1] | Normalized y-coordinate of bbox center |
| `width` | float | [0, 1] | Normalized bbox width |
| `height` | float | [0, 1] | Normalized bbox height |
| `confidence` | float | [0, 1] | Model confidence score |

The first 5 tokens are identical to the detection label format (§2.1). The 6th token is the prediction confidence.

#### 2.4.2 Segmentation Prediction

An **even number > 6** of tokens:

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn> <confidence>
```

| Token | Type | Range | Description |
|-------|------|-------|-------------|
| `class_id` | int | 0..N-1 | Zero-based class index |
| `x1, y1, ...` | float | [0, 1] | Normalized polygon vertex coordinates |
| `confidence` | float | [0, 1] | Model confidence score (the **last token**) |

All coordinate tokens except the last are identical to the segmentation label format (§2.2). The final token is the prediction confidence.

#### 2.4.3 Confidence Validation

- `confidence` must be a finite float in the range `[0, 1]`
- Values outside this range are rejected in strict mode; in non-strict mode, they are clamped with a warning
- Missing confidence in prediction mode is treated as a format error

### 2.5 Empty Lines

Empty lines (whitespace-only) are ignored during parsing. An empty file indicates an image with no objects.

### 2.6 Keypoints (Pose)

Each pose row carries a detection-style bbox **plus** k keypoint triplets (coordinate
pair + visibility flag):

```
<class_id> <x_center> <y_center> <width> <height> <px1> <py1> <pv1> <px2> <py2> <pv2> ... <pxk> <pyk> <pvk>
```

| Token | Type | Range | Description |
|-------|------|-------|-------------|
| `class_id` | int | 0..N-1 | Zero-based class index |
| `x_center`, `y_center`, `width`, `height` | float | [0, 1] | Bbox, same semantics as §2.1 |
| `px, py` | float | [0, 1] | Normalized keypoint coordinates |
| `pv` | int | {0, 1, 2} | Visibility: 0 = not labeled (px=py=0), 1 = labeled but not visible, 2 = labeled and visible |

**Constraints**:
- k ≥ 1 and fixed per dataset (see §2.3 Pose Mode)
- `pv` ∈ {0, 1, 2}; a point with `pv == 0` is conventionally written as `0 0 0`
- Pose prediction rows append a `confidence` token after the last `pv` (§2.4 rules apply)
- The row carries no segmentation — segmentation cannot be represented in a pose row

## 3. Coordinate System

### 3.1 Normalization

All coordinates are **normalized to [0, 1]** relative to image dimensions:

```
x_normalized = x_pixel / image_width
y_normalized = y_pixel / image_height
w_normalized = w_pixel / image_width
h_normalized = h_pixel / image_height
```

### 3.2 Bounding Box Center Convention

The bounding box is defined by its **center point** and **dimensions**:

```
Top-left corner (pixels):     ( (x_center - width/2)  * W,  (y_center - height/2) * H )
Bottom-right corner (pixels):  ( (x_center + width/2)  * W,  (y_center + height/2) * H )
```

Where `W` = image_width, `H` = image_height.

### 3.3 Validation Constraints

A valid YOLO annotation line must satisfy ALL of the following:

1. `class_id` is a valid integer and exists in `classes.txt`. Integer-valued floats (e.g., `1.00`, `0.0`) are accepted and parsed as their integer equivalent (`1`, `0`). Non-integer floats (e.g., `0.5`) are rejected.
2. All coordinate values are finite floats in [0, 1]
3. For detection: exactly 5 tokens
4. For segmentation: odd number of tokens, at least 7 (class_id + 3 point pairs)
5. Polygon has at least 3 vertices
6. For pose (label): `5 + 3k` tokens for the dataset's fixed k ≥ 1
7. For pose (prediction): `6 + 3k` tokens for the dataset's fixed k ≥ 1
8. Pose visibility values are integers in {0, 1, 2}

### 3.4 Common Misinterpretations

| **Wrong** | **Correct** |
|-----------|-------------|
| `(x, y)` is top-left corner | `(x, y)` is the **center** of the box |
| Coordinates can exceed [0, 1] | Coordinates must be in [0, 1] |
| Width/height can be zero | Width/height must be in (0, 1] |
| class_id can be arbitrary | class_id is a **zero-based** contiguous index |

## 4. Precision

- Coordinate values are written with **6 decimal places** (`.6f` format).
- Internal storage uses Python `float` (IEEE 754 double precision).

## 5. Reference Example

### Detection

```
# classes.txt
person
car

# image_001.txt (for an 800×600 image)
0 0.500000 0.375000 0.200000 0.300000
1 0.750000 0.500000 0.150000 0.200000
```

**Verification**:
- Object 0 (person): center=(400, 225), size=(160×180), bbox=[320, 135, 480, 315] in pixels
- Object 1 (car): center=(600, 300), size=(120×120), bbox=[540, 240, 660, 360] in pixels

### Segmentation

```
# For an 800×600 image
0 0.400000 0.300000 0.500000 0.250000 0.600000 0.350000 0.550000 0.450000 0.400000 0.400000
```

This defines a 5-vertex polygon (10 coordinate values + class_id = 11 tokens).

### Keypoints (Pose)

```
# For an 800×600 image — 3-keypoint pose (k=3): 5 + 3×3 = 14 tokens
0 0.500000 0.375000 0.200000 0.300000 0.480000 0.350000 2 0.520000 0.340000 2 0.000000 0.000000 0
```

**Verification**:
- Bbox: center=(400, 225), size=(160×180) — same as the detection example
- Keypoint 0: (384, 210), v=2 (labeled and visible)
- Keypoint 1: (416, 204), v=2 (labeled and visible)
- Keypoint 2: (0, 0), v=0 (not labeled)
