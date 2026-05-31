# YOLO Annotation Format Specification

> **Version:** 1.0
> **Status:** Canonical — this document defines the authoritative YOLO format contract for DataFlow-CV.

## 1. File Organization

### 1.1 File Structure

- **One `.txt` file per image**, using the same stem name as the corresponding image file.
- Example: `image_001.jpg` → `image_001.txt`
- All label files reside in a single flat directory (no subdirectories).
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

The handler determines annotation type by token count:
- `len(tokens) == 5` → Detection (bbox)
- `len(tokens) > 5 and len(tokens) % 2 == 1` → Segmentation (polygon)
- Any other token count → Invalid

### 2.4 Empty Lines

Empty lines (whitespace-only) are ignored during parsing. An empty file indicates an image with no objects.

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

1. `class_id` is a valid integer and exists in `classes.txt`
2. All coordinate values are finite floats in [0, 1]
3. For detection: exactly 5 tokens
4. For segmentation: odd number of tokens, at least 7 (class_id + 3 point pairs)
5. Polygon has at least 3 vertices

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
- Original line text is preserved in `OriginalData` for lossless round-trip conversion.

## 5. Original Data Preservation

For lossless A→B→A round-trip conversion, each parsed YOLO annotation stores:

```python
OriginalData(
    format="yolo",
    raw_data={
        "line": "<original line text>",
        "line_number": <int>,
        "items": [class_id_str, x_float, y_float, w_float, h_float],
        "is_detection": <bool>,
        "is_segmentation": <bool>,
    }
)
```

**Critical detail**: The first element of `items` is a **string** (class_id), while the remaining elements are **floats**. When reconstructing a line from original data, `items` must be copied with `list()` before mutation.

## 6. Reference Example

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
