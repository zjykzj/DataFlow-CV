"""
Convert LabelMe annotation to YOLO format.
"""

import json
import os
from pathlib import Path
from typing import List


def labelme_to_yolo(labelme_json_path: str, output_txt_path: str,
                    class_names: List[str] = None) -> None:
    """
    Convert LabelMe annotation to YOLO format.

    Args:
        labelme_json_path: Path to LabelMe JSON annotation file
        output_txt_path: Path to save YOLO format annotation
        class_names: List of class names (optional, will be extracted from LabelMe if not provided)
    """
    # Validate path
    if not Path(labelme_json_path).exists():
        raise FileNotFoundError(f"LabelMe annotation file not found: {labelme_json_path}")

    # Load LabelMe annotation
    with open(labelme_json_path, 'r') as f:
        labelme_data = json.load(f)

    # Extract image dimensions
    image_height = labelme_data.get('imageHeight', 0)
    image_width = labelme_data.get('imageWidth', 0)

    if image_height == 0 or image_width == 0:
        raise ValueError("Invalid image dimensions in LabelMe annotation")

    # Process shapes
    shapes = labelme_data.get('shapes', [])
    if not shapes:
        print("Warning: No shapes found in LabelMe annotation")

    # Extract class names from shapes if not provided
    if class_names is None or len(class_names) == 0:
        class_names = []
        for shape in shapes:
            label = shape.get('label', '')
            if label and label not in class_names:
                class_names.append(label)

    if not class_names:
        raise ValueError("No class names found in LabelMe annotation")

    # Create class name to index mapping
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Process annotations
    yolo_annotations = []
    for shape in shapes:
        label = shape.get('label', '')
        shape_type = shape.get('shape_type', '')
        points = shape.get('points', [])

        if not label:
            print("Warning: Shape has no label, skipping")
            continue

        # Get class index
        if label not in class_to_idx:
            print(f"Warning: Label '{label}' not in class names list, assigning new index")
            class_to_idx[label] = len(class_names)
            class_names.append(label)

        class_idx = class_to_idx[label]

        # Get bounding box based on shape type
        if shape_type == 'rectangle' and len(points) == 2:
            # Rectangle: points are [top-left, bottom-right]
            x1 = min(points[0][0], points[1][0])
            y1 = min(points[0][1], points[1][1])
            x2 = max(points[0][0], points[1][0])
            y2 = max(points[0][1], points[1][1])

        elif shape_type == 'polygon' and len(points) >= 3:
            # Polygon: compute bounding box
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            x1 = min(x_coords)
            y1 = min(y_coords)
            x2 = max(x_coords)
            y2 = max(y_coords)
        else:
            print(f"Warning: Unsupported shape type '{shape_type}' or invalid points, skipping")
            continue

        # Ensure coordinates are within image bounds
        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(0, min(image_width - 1, x2))
        y2 = max(0, min(image_height - 1, y2))

        # Convert to YOLO format (xc, yc, w, h normalized)
        width = x2 - x1
        height = y2 - y1
        xc = x1 + width / 2
        yc = y1 + height / 2

        # Normalize
        xc_norm = xc / image_width
        yc_norm = yc / image_height
        w_norm = width / image_width
        h_norm = height / image_height

        # Ensure values are within [0, 1]
        xc_norm = max(0, min(1, xc_norm))
        yc_norm = max(0, min(1, yc_norm))
        w_norm = max(0, min(1, w_norm))
        h_norm = max(0, min(1, h_norm))

        yolo_annotations.append([class_idx, xc_norm, yc_norm, w_norm, h_norm])

    # Save YOLO format
    output_dir = Path(output_txt_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_txt_path, 'w') as f:
        for ann in yolo_annotations:
            line = f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}"
            f.write(line + '\n')

    print(f"Converted {len(yolo_annotations)} annotations to YOLO format")
    print(f"Class names used: {class_names}")