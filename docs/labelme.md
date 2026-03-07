# LabelMe Format Documentation

## Overview

LabelMe is a web-based annotation tool for image annotation. The LabelMe format stores annotations in JSON files with support for various shape types including rectangles and polygons.

## File Format

LabelMe annotations are stored in JSON files with the following structure:

### Top-Level Structure

```json
{
  "version": "5.3.1",
  "flags": {},
  "shapes": [...],
  "imagePath": "image.jpg",
  "imageData": null,
  "imageHeight": 480,
  "imageWidth": 640
}
```

### 1. `shapes` Section

List of annotation shapes:

```json
{
  "shapes": [
    {
      "label": "person",
      "points": [[x1, y1], [x2, y2]],
      "group_id": null,
      "description": "",
      "shape_type": "rectangle",
      "flags": {}
    }
  ]
}
```

**Shape Types Supported:**
- `"rectangle"`: Two points defining top-left and bottom-right corners
- `"polygon"`: Three or more points defining a polygon outline

**Key Fields:**
- `label`: Object class name (string)
- `points`: Array of coordinate points `[[x, y], ...]`
- `shape_type`: Type of shape (`"rectangle"` or `"polygon"`)
- `group_id`: For grouping related shapes (optional)
- `description`: Additional description (optional)

### 2. Image Information

```json
{
  "imagePath": "relative/path/to/image.jpg",
  "imageData": null,  // Base64 encoded image (optional)
  "imageHeight": 480,
  "imageWidth": 640
}
```

**Note:** `imageData` is typically `null` in files, as images are stored separately.

## Coordinate System

### Rectangle Format
- **Format**: Two points `[[x1, y1], [x2, y2]]`
- **Points**: Top-left and bottom-right corners
- **Units**: Pixels (integer values)
- **Example**: `[[100, 50], [220, 130]]` defines a rectangle from (100,50) to (220,130)

### Polygon Format
- **Format**: Array of points `[[x1, y1], [x2, y2], [x3, y3], ...]`
- **Points**: Sequential points defining the polygon boundary
- **Units**: Pixels (integer values)
- **Minimum**: 3 points required

## Usage with DataFlow-CV

### Visualization

```bash
# Visualize LabelMe annotations on an image
dataflow visualize labelme image.jpg annotation.json --show

# Save visualization
dataflow visualize labelme image.jpg annotation.json --save output.jpg

# Batch visualization of directory
dataflow visualize labelme images/ annotations/ --batch --show
```

### Conversion

```bash
# LabelMe to COCO format
dataflow convert labelme2coco labelme.json output.json

# LabelMe to YOLO format
dataflow convert labelme2yolo image.jpg labelme.json output.txt --class-names classes.txt
```

### Python API

```python
import dataflow

# Visualize LabelMe annotations
image = dataflow.visualize.visualize_labelme("image.jpg", "annotation.json")

# Convert LabelMe to COCO
dataflow.convert.labelme_to_coco("labelme.json", "coco.json")

# Convert LabelMe to YOLO
dataflow.convert.labelme_to_yolo(
    "image.jpg",
    "labelme.json",
    "yolo.txt",
    class_names=["person", "car", "dog"]
)
```

## Example Files

### Rectangle Annotation Example

```json
{
  "version": "5.3.1",
  "flags": {},
  "shapes": [
    {
      "label": "person",
      "points": [[100, 50], [220, 130]],
      "group_id": null,
      "description": "",
      "shape_type": "rectangle",
      "flags": {}
    },
    {
      "label": "car",
      "points": [[300, 200], [450, 280]],
      "group_id": null,
      "description": "Red car",
      "shape_type": "rectangle",
      "flags": {}
    }
  ],
  "imagePath": "example.jpg",
  "imageData": null,
  "imageHeight": 480,
  "imageWidth": 640
}
```

### Polygon Annotation Example

```json
{
  "version": "5.3.1",
  "flags": {},
  "shapes": [
    {
      "label": "building",
      "points": [[100, 50], [150, 30], [200, 50], [200, 100], [100, 100]],
      "group_id": null,
      "description": "Office building",
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "imagePath": "building.jpg",
  "imageData": null,
  "imageHeight": 480,
  "imageWidth": 640
}
```

## Notes

- **Image dimensions** (`imageHeight`, `imageWidth`) are required for proper conversion
- **Label names** are case-sensitive and used directly as class names
- **Polygon annotations** are converted to their bounding boxes during format conversion
- **Multiple shapes** can have the same label (multiple instances of same class)
- **Empty shapes arrays** are allowed (images with no objects)
- **Relative paths** in `imagePath` should be resolvable from the annotation file location

## Best Practices

1. **Consistent labeling**: Use the same label names across all annotations
2. **Accurate points**: Ensure rectangle points are actually top-left and bottom-right
3. **Image dimensions**: Always include correct `imageHeight` and `imageWidth`
4. **File organization**: Keep images and annotation files in the same directory or use relative paths