# YOLO Plain-Text Label Format Specification

> **Role:** Single source of truth for YOLO `.txt` annotation format in DataFlow-CV. All handlers, converters, and visualizers MUST conform to this spec.

## 1. File Overview

YOLO annotations are stored as **plain-text files**, with one file per image.

| Property | Specification |
|----------|---------------|
| File extension | `.txt` |
| Encoding | UTF-8 |
| Line ending | LF (`\n`) or CRLF (`\r\n`) |
| File naming | `<image_stem>.txt` — must match the corresponding image file name, excluding the image extension |

**Naming rule:**

```
image_stem = basename_of_image_file_without_extension
label_file  = image_stem + ".txt"
```

Example: image `frame_001.jpg` → label `frame_001.txt`, image `scene.png` → label `scene.txt`.

Each line in the file represents exactly **one annotated object**. Empty lines are permitted and MUST be ignored by parsers. The number of objects in an image equals the number of non-empty lines in its label file.

## 2. Line Structure Definition

### 2.1 Format

Each non-empty line MUST follow this whitespace-separated structure:

```
<class_id> <x_center> <y_center> <width> <height>
```

A line MUST contain exactly **5 tokens**. Any line with a token count other than 5 is malformed.

### 2.2 Field Definitions

| Index | Field | Type | Required | Description |
|-------|-------|------|----------|-------------|
| 0 | `class_id` | `integer` | **Required** | Zero-based class index. Must be a non-negative integer. |
| 1 | `x_center` | `float` | **Required** | Normalized x-coordinate of the bounding box **center**. Range: `[0.0, 1.0]`. |
| 2 | `y_center` | `float` | **Required** | Normalized y-coordinate of the bounding box **center**. Range: `[0.0, 1.0]`. |
| 3 | `width` | `float` | **Required** | Normalized width of the bounding box. Range: `[0.0, 1.0]`. |
| 4 | `height` | `float` | `Required` | Normalized height of the bounding box. Range: `[0.0, 1.0]`. |

**Token separator:** One or more whitespace characters (space ` ` or tab `\t`). Leading and trailing whitespace on each line MUST be stripped before parsing.

**Precision:** Floating-point values SHOULD be written with sufficient precision to avoid coordinate drift on round-trip conversion. Implementations in this project use 6 decimal places (e.g., `0.312500`).

### 2.3 Validation Constraints

A compliant parser MUST reject (or warn, depending on strictness mode) any line meeting any of these conditions:

- `class_id` is not a valid integer, or is negative.
- `x_center`, `y_center`, `width`, or `height` cannot be parsed as a float.
- Any coordinate value falls outside the closed interval `[0.0, 1.0]`.
- The resulting bounding box has zero or negative area (`width <= 0` or `height <= 0`).

## 3. Coordinate System (Critical)

### 3.1 Normalization Requirement

**All four coordinate values are normalized to the image dimensions.** The range is the closed interval `[0, 1]`:

- `0.0` represents the left/top edge of the image.
- `1.0` represents the right/bottom edge of the image.
- Values at exactly `0.0` or `1.0` are valid and indicate that the box center or boundary touches (or lies exactly on) the corresponding image edge.

### 3.2 Center-Origin Semantics

This is the single most important definition in the spec:

> `(x_center, y_center)` is the **center point** of the bounding box, **NOT** the top-left corner.

### 3.3 Mathematical Definitions

Given an image of dimensions `W` × `H` pixels, and a bounding box defined by pixel-coordinate corners `(x_min, y_min)` (top-left) and `(x_max, y_max)` (bottom-right):

```
x_center = (x_min + x_max) / (2 × W)
y_center = (y_min + y_max) / (2 × H)
width    = (x_max - x_min) / W
height   = (y_max - y_min) / H
```

Conversely, to reconstruct pixel-coordinate corners from normalized YOLO values:

```
x_min = (x_center - width / 2) × W
y_min = (y_center - height / 2) × H
x_max = (x_center + width / 2) × W
y_max = (y_center + height / 2) × H
```

### 3.4 Coordinate Bounds

For the bounding box to be fully contained within the image, the following inequalities MUST hold:

```
0 ≤ x_center - width / 2    (left edge within image)
x_center + width / 2 ≤ 1    (right edge within image)
0 ≤ y_center - height / 2   (top edge within image)
y_center + height / 2 ≤ 1   (bottom edge within image)
```

These constraints are equivalent to requiring the bounding box corners to lie within `[0, 1]` when converted to `(x_min, y_min, x_max, y_max)` form. Implementations MAY enforce these bounds and reject boxes that extend beyond the image boundary.

### 3.5 Common Misinterpretations to Avoid

| Incorrect Interpretation | Correct Interpretation |
|--------------------------|----------------------|
| `(x_center, y_center)` = top-left corner | `(x_center, y_center)` = center point |
| Coordinates in absolute pixels | Coordinates normalized to `[0, 1]` |
| `width`, `height` are absolute pixels | `width`, `height` are fractions of image dimensions |
| Origin at bottom-left | Origin at top-left (consistent with pixel coordinate conventions) |

## 4. Category Mapping

### 4.1 `class_id` Definition

`class_id` is a **0-based integer index** that maps to a class name through an external mapping file. The spec is concerned only with the integer ID itself; the mapping mechanism is out of scope for a single label file.

### 4.2 Mapping Files (External Context)

Two common mapping approaches exist (described here for reference only — the label file itself contains only the integer `class_id`):

| File | Format | Role |
|------|--------|------|
| `classes.txt` | One class name per line; line number = `class_id` | Simple ordered list |
| `data.yaml` | YAML with `names` key mapping IDs to names | Extended configuration |

Example `classes.txt` content:

```
person
bicycle
car
```

In this example, `class_id = 0` maps to `"person"`, `class_id = 1` maps to `"bicycle"`, and `class_id = 2` maps to `"car"`. The label file records only the integer — `0`, `1`, or `2` — and never the human-readable name.

## 5. Standard Example

### 5.1 Scenario

| Parameter | Value |
|-----------|-------|
| Image file | `street.jpg` |
| Image dimensions | 640 × 480 pixels |
| Label file | `street.txt` |

The image contains two annotated objects:

| Object | Pixel Bounding Box `(x_min, y_min, x_max, y_max)` | Class |
|--------|---------------------------------------------------|-------|
| Person | (100, 80, 300, 320) | `person` (id=0) |
| Car | (400, 300, 520, 460) | `car` (id=2) |

### 5.2 Normalization

**Person (class_id=0):**

```
x_center = (100 + 300) / (2 × 640) = 400 / 1280 = 0.312500
y_center = (80 + 320)  / (2 × 480) = 400 / 960  = 0.416667
width    = (300 - 100) / 640        = 200 / 640  = 0.312500
height   = (320 - 80)  / 480        = 240 / 480  = 0.500000
```

**Car (class_id=2):**

```
x_center = (400 + 520) / (2 × 640) = 920 / 1280 = 0.718750
y_center = (300 + 460) / (2 × 480) = 760 / 960  = 0.791667
width    = (520 - 400) / 640        = 120 / 640  = 0.187500
height   = (460 - 300) / 480        = 160 / 480  = 0.333333
```

### 5.3 Resulting Label File

`street.txt`:

```
0 0.312500 0.416667 0.312500 0.500000
2 0.718750 0.791667 0.187500 0.333333
```

### 5.4 Verification

To verify correctness, reconstruct the pixel bounding box for the first line:

```
x_min = (0.312500 - 0.312500 / 2) × 640 = 0.15625 × 640 = 100 ✓
y_min = (0.416667 - 0.500000 / 2) × 480 = 0.166667 × 480 = 80 ✓
x_max = (0.312500 + 0.312500 / 2) × 640 = 0.46875 × 640 = 300 ✓
y_max = (0.416667 + 0.500000 / 2) × 480 = 0.666667 × 480 = 320 ✓
```

The reconstruction matches the original pixel bounding box `(100, 80, 300, 320)`. Rounding differences from floating-point precision are expected and acceptable.
