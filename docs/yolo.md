# YOLO Format Documentation

## Overview

YOLO (You Only Look Once) format is used for object detection annotations in YOLO-based models. It uses normalized coordinates and simple text file format.

## File Format

YOLO annotations are stored in plain text files (`.txt`) with one annotation per line.

### Line Format

```
<class_index> <x_center> <y_center> <width> <height>
```

**Fields:**
- `class_index`: Integer class ID (0-based)
- `x_center`: Normalized x-coordinate of bounding box center (0.0 to 1.0)
- `y_center`: Normalized y-coordinate of bounding box center (0.0 to 1.0)
- `width`: Normalized width of bounding box (0.0 to 1.0)
- `height`: Normalized height of bounding box (0.0 to 1.0)

### Example File Content

```
0 0.512500 0.312500 0.187500 0.166667
1 0.703125 0.562500 0.234375 0.208333
0 0.296875 0.604167 0.156250 0.166667
```

## Coordinate System

- **Format**: Normalized `(x_center, y_center, width, height)`
- **Normalization**: All values divided by image dimensions (width/height)
- **Range**: All values between 0.0 and 1.0
- **Center-based**: Coordinates represent bounding box center, not corner

### Conversion Formula

```python
# From absolute coordinates to YOLO format
x_center = (x1 + width/2) / image_width
y_center = (y1 + height/2) / image_height
width_norm = width / image_width
height_norm = height / image_height

# From YOLO format to absolute coordinates
x1 = (x_center - width_norm/2) * image_width
y1 = (y_center - height_norm/2) * image_height
width = width_norm * image_width
height = height_norm * image_height
```

## Class Names File

YOLO format requires a separate class names file (typically `classes.txt`) that maps class indices to class names.

### `classes.txt` Format

One class name per line:

```
person
car
dog
cat
bicycle
```

**Mapping:**
- Line 0: `person` → class index 0
- Line 1: `car` → class index 1
- Line 2: `dog` → class index 2
- etc.

## Usage with DataFlow-CV

### Visualization

```bash
# Visualize YOLO annotations on an image
dataflow visualize yolo image.jpg labels.txt classes.txt --show

# Save visualization
dataflow visualize yolo image.jpg labels.txt classes.txt --save output.jpg

# Batch visualization of directory
dataflow visualize yolo images/ labels/ classes.txt --batch --show
```

### Conversion

```bash
# YOLO to COCO format
dataflow convert yolo2coco image.jpg labels.txt classes.txt output.json

# YOLO to LabelMe format
dataflow convert yolo2labelme image.jpg labels.txt classes.txt output.json
```

### Python API

```python
import dataflow

# Visualize YOLO annotations
image = dataflow.visualize.visualize_yolo(
    "image.jpg",
    "labels.txt",
    ["person", "car", "dog"]  # class names list
)

# Convert YOLO to COCO
dataflow.convert.yolo_to_coco(
    "image.jpg",
    "labels.txt",
    ["person", "car", "dog"],
    "coco.json"
)
```

## Example Files

### Annotation File (`image1.txt`)

```
0 0.512500 0.312500 0.187500 0.166667
1 0.703125 0.562500 0.234375 0.208333
0 0.296875 0.604167 0.156250 0.166667
```

### Class Names File (`classes.txt`)

```
person
car
dog
cat
bicycle
traffic_light
```

### Complete Example

For an 800×600 image with:
1. A person at center (400, 300) with size 150×100
2. A car at (500, 400) with size 200×150

**Calculations:**
```
# Person (class 0)
x_center = 400 / 800 = 0.5
y_center = 300 / 600 = 0.5
width = 150 / 800 = 0.1875
height = 100 / 600 = 0.166667

# Car (class 1)
x_center = 500 / 800 = 0.625
y_center = 400 / 600 = 0.666667
width = 200 / 800 = 0.25
height = 150 / 600 = 0.25
```

**Annotation file content:**
```
0 0.500000 0.500000 0.187500 0.166667
1 0.625000 0.666667 0.250000 0.250000
```

## File Organization

YOLO datasets typically follow this structure:

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       ├── image3.jpg
│       └── image4.jpg
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   └── image2.txt
│   └── val/
│       ├── image3.txt
│       └── image4.txt
└── classes.txt
```

**Naming convention:** Annotation files have the same basename as their corresponding images (e.g., `image1.jpg` → `image1.txt`).

## Notes

- **0-based indexing**: Class indices start at 0
- **Normalized coordinates**: All values must be between 0.0 and 1.0
- **Precision**: Typically 6 decimal places are used
- **Image dimensions**: Required for conversion to/from other formats
- **No header**: Text files contain only annotation lines, no headers or metadata
- **Empty files**: Images with no objects should have empty annotation files (0 bytes)

## Common Issues

1. **Coordinates out of range**: Values >1.0 or <0.0 will cause errors
2. **Missing class names**: Class indices must exist in the class names list
3. **Inconsistent naming**: Annotation files must match image basenames
4. **Wrong image dimensions**: Using incorrect image size will distort annotations