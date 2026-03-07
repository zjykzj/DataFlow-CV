"""
Visualize LabelMe annotations.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from .base import BaseVisualizer


def visualize_labelme(image_path: str, labelme_json_path: str) -> np.ndarray:
    """
    Visualize LabelMe annotations on image.

    Args:
        image_path: Path to image file
        labelme_json_path: Path to LabelMe JSON annotation file

    Returns:
        Image with annotations drawn
    """
    # Validate paths
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not Path(labelme_json_path).exists():
        raise FileNotFoundError(f"LabelMe annotation file not found: {labelme_json_path}")

    # Load image
    image = BaseVisualizer.load_image(image_path)
    img_height, img_width = image.shape[:2]

    # Load LabelMe annotation
    with open(labelme_json_path, 'r') as f:
        labelme_data = json.load(f)

    # Get shapes
    shapes = labelme_data.get('shapes', [])
    if not shapes:
        print("Warning: No shapes found in LabelMe annotation")
        return image

    # Extract unique class names for color mapping
    class_names = []
    for shape in shapes:
        label = shape.get('label', '')
        if label and label not in class_names:
            class_names.append(label)

    # Draw shapes
    for shape in shapes:
        label = shape.get('label', '')
        shape_type = shape.get('shape_type', '')
        points = shape.get('points', [])

        if not label:
            print("Warning: Shape has no label, skipping")
            continue

        # Get class index for color
        class_idx = class_names.index(label) if label in class_names else 0
        color = BaseVisualizer.get_color(class_idx, len(class_names))

        if shape_type == 'rectangle' and len(points) == 2:
            # Rectangle: points are [top-left, bottom-right]
            x1 = min(points[0][0], points[1][0])
            y1 = min(points[0][1], points[1][1])
            x2 = max(points[0][0], points[1][0])
            y2 = max(points[0][1], points[1][1])

            # Ensure coordinates are within image bounds
            x1 = max(0, min(img_width - 1, x1))
            y1 = max(0, min(img_height - 1, y1))
            x2 = max(0, min(img_width - 1, x2))
            y2 = max(0, min(img_height - 1, y2))

            # Draw rectangle
            image = BaseVisualizer.draw_bbox(
                image,
                [x1, y1, x2, y2],
                color=color,
                thickness=2,
                label=label
            )

        elif shape_type == 'polygon' and len(points) >= 3:
            # Polygon: draw polygon outline
            points_array = np.array(points, dtype=np.int32)

            # Draw polygon
            cv2.polylines(
                image,
                [points_array],
                isClosed=True,
                color=color,
                thickness=2
            )

            # Draw label at first point
            if len(points) > 0:
                x, y = int(points[0][0]), int(points[0][1])
                cv2.putText(
                    image,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
        else:
            print(f"Warning: Unsupported shape type '{shape_type}', skipping")

    print(f"Visualized {len(shapes)} LabelMe annotations")
    return image


def visualize_labelme_from_data(image_path: str, labelme_data: Dict[str, Any]) -> np.ndarray:
    """
    Visualize LabelMe annotations from data dictionary.

    Args:
        image_path: Path to image file
        labelme_data: LabelMe annotation data dictionary

    Returns:
        Image with annotations drawn
    """
    # This function allows visualization without loading from file
    # For now, just save to temp file and use the main function
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(labelme_data, f)
        temp_path = f.name

    try:
        result = visualize_labelme(image_path, temp_path)
    finally:
        os.unlink(temp_path)

    return result