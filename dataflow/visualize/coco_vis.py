"""
Visualize COCO annotations.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from .base import BaseVisualizer


def visualize_coco(image_path: str, coco_json_path: str) -> np.ndarray:
    """
    Visualize COCO annotations on image.

    Args:
        image_path: Path to image file
        coco_json_path: Path to COCO JSON annotation file

    Returns:
        Image with annotations drawn
    """
    # Validate paths
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not Path(coco_json_path).exists():
        raise FileNotFoundError(f"COCO annotation file not found: {coco_json_path}")

    # Load image
    image = BaseVisualizer.load_image(image_path)
    img_height, img_width = image.shape[:2]

    # Load COCO annotation
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Get image information
    image_filename = Path(image_path).name
    image_id = None

    # Try to find image by filename
    for img in coco_data.get('images', []):
        if img.get('file_name') == image_filename:
            image_id = img['id']
            break

    # If not found by filename, use first image
    if image_id is None and coco_data.get('images') and len(coco_data['images']) > 0:
        image_id = coco_data['images'][0]['id']

    if image_id is None:
        print("Warning: Could not find image information in COCO annotation")
        return image

    # Build category ID to name mapping
    category_map = {}
    for cat in coco_data.get('categories', []):
        category_map[cat['id']] = cat['name']

    # Find annotations for this image
    annotations = []
    for ann in coco_data.get('annotations', []):
        if ann.get('image_id') != image_id:
            continue

        # Skip crowd annotations
        if ann.get('iscrowd', 0) != 0:
            continue

        annotations.append(ann)

    if not annotations:
        print("Warning: No annotations found for this image in COCO file")
        return image

    # Draw annotations
    for ann in annotations:
        # Get category name
        cat_id = ann.get('category_id')
        label = category_map.get(cat_id, f"class_{cat_id}")

        # Get bounding box
        bbox = ann.get('bbox', [])
        if len(bbox) != 4:
            continue

        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h

        # Ensure coordinates are within image bounds
        x1 = max(0, min(img_width - 1, x1))
        y1 = max(0, min(img_height - 1, y1))
        x2 = max(0, min(img_width - 1, x2))
        y2 = max(0, min(img_height - 1, y2))

        # Get color for this class
        class_idx = list(category_map.keys()).index(cat_id) if cat_id in category_map else 0
        color = BaseVisualizer.get_color(class_idx, len(category_map))

        # Draw bounding box
        image = BaseVisualizer.draw_bbox(
            image,
            [x1, y1, x2, y2],
            color=color,
            thickness=2,
            label=label
        )

    print(f"Visualized {len(annotations)} COCO annotations")
    return image


def visualize_coco_from_data(image_path: str, coco_data: Dict[str, Any]) -> np.ndarray:
    """
    Visualize COCO annotations from data dictionary.

    Args:
        image_path: Path to image file
        coco_data: COCO annotation data dictionary

    Returns:
        Image with annotations drawn
    """
    # This function allows visualization without loading from file
    # For now, just save to temp file and use the main function
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coco_data, f)
        temp_path = f.name

    try:
        result = visualize_coco(image_path, temp_path)
    finally:
        os.unlink(temp_path)

    return result