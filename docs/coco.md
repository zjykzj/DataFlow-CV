# COCO Format Documentation

## Overview

COCO (Common Objects in Context) is a large-scale object detection, segmentation, and captioning dataset format. DataFlow-CV supports COCO format for object detection annotations.

## File Format

COCO annotations are stored in JSON files with the following structure:

### Top-Level Structure

```json
{
  "info": {...},
  "licenses": [...],
  "images": [...],
  "annotations": [...],
  "categories": [...]
}
```

### 1. `info` Section

Contains metadata about the dataset:

```json
{
  "info": {
    "description": "Dataset description",
    "url": "http://example.com",
    "version": "1.0",
    "year": 2026,
    "contributor": "Data contributor",
    "date_created": "2026-03-07T10:30:00"
  }
}
```

### 2. `images` Section

List of images in the dataset:

```json
{
  "images": [
    {
      "id": 1,
      "width": 640,
      "height": 480,
      "file_name": "image1.jpg",
      "license": 1,
      "date_captured": "2026-03-07T10:30:00"
    }
  ]
}
```

### 3. `annotations` Section

List of object annotations:

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1000.0,
      "iscrowd": 0,
      "segmentation": []
    }
  ]
}
```

**Key Fields:**
- `bbox`: `[x1, y1, width, height]` where `(x1, y1)` is the top-left corner
- `area`: Bounding box area (width × height)
- `iscrowd`: 0 for single object, 1 for crowd regions
- `segmentation`: Empty array for detection-only annotations

### 4. `categories` Section

List of object categories:

```json
{
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "person"
    }
  ]
}
```

## Coordinate System

- **Format**: `[x1, y1, width, height]`
- **Origin**: Top-left corner of image
- **Units**: Pixels (integer values)
- **Example**: `[100, 50, 120, 80]` means:
  - Top-left corner at (100, 50)
  - Width: 120 pixels
  - Height: 80 pixels

## Usage with DataFlow-CV

### Visualization

```bash
# Visualize COCO annotations on an image
dataflow visualize coco image.jpg annotation.json --show

# Save visualization
dataflow visualize coco image.jpg annotation.json --save output.jpg

# Batch visualization of directory
dataflow visualize coco images/ annotations/ --batch --show
```

### Conversion

```bash
# COCO to YOLO format
dataflow convert coco2yolo image.jpg annotation.json output.txt --class-names classes.txt

# COCO to LabelMe format
dataflow convert coco2labelme annotation.json image.jpg output.json
```

### Python API

```python
import dataflow

# Visualize COCO annotations
image = dataflow.visualize.visualize_coco("image.jpg", "annotation.json")

# Convert COCO to YOLO
dataflow.convert.coco_to_yolo(
    "image.jpg",
    "coco.json",
    "yolo.txt",
    class_names=["person", "car", "dog"]
)
```

## Example File

```json
{
  "info": {
    "description": "Example COCO dataset",
    "url": "",
    "version": "1.0",
    "year": 2026,
    "contributor": "DataFlow",
    "date_created": "2026-03-07T10:30:00"
  },
  "licenses": [{
    "id": 1,
    "name": "Unknown",
    "url": ""
  }],
  "images": [{
    "id": 1,
    "width": 640,
    "height": 480,
    "file_name": "example.jpg",
    "license": 1,
    "date_captured": "2026-03-07T10:30:00"
  }],
  "annotations": [{
    "id": 1,
    "image_id": 1,
    "category_id": 1,
    "bbox": [100.0, 50.0, 120.0, 80.0],
    "area": 9600.0,
    "iscrowd": 0,
    "segmentation": []
  }],
  "categories": [{
    "id": 1,
    "name": "person",
    "supercategory": "person"
  }]
}
```

## Notes

- COCO IDs start from 1 (not 0)
- `iscrowd=1` annotations are skipped during conversion to other formats
- Image dimensions are required for proper coordinate conversion
- File paths in `images.file_name` should match actual image filenames