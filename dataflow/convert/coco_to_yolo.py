"""
Convert COCO annotation to YOLO format.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from .base import BaseConverter


def coco_to_yolo(coco_json_path: str, output_dir: str) -> None:
    """
    Convert COCO annotation to YOLO format.

    Args:
        coco_json_path: Path to COCO JSON annotation file (contains all images and annotations)
        output_dir: Output directory for YOLO .txt files and class.names file

    The function reads the COCO JSON file, extracts category information, and creates:
    1. class.names file in output_dir with all category names
    2. One .txt file per image in output_dir with YOLO format annotations
    """
    # Validate input file
    BaseConverter.validate_paths(coco_json_path)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load COCO annotation
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Extract categories
    if 'categories' not in coco_data:
        raise ValueError("COCO JSON does not contain 'categories' field")

    # Sort categories by ID to ensure consistent ordering
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    class_names = [cat['name'] for cat in categories]

    # Save class names file
    class_names_path = output_path / "class.names"
    with open(class_names_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Saved class names to {class_names_path}")

    # Create mapping from category_id to class index (0-based for YOLO)
    cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(categories)}

    # Build image_id -> image_info mapping
    images_by_id = {}
    for img in coco_data.get('images', []):
        images_by_id[img['id']] = {
            'width': img.get('width', 0),
            'height': img.get('height', 0),
            'file_name': img.get('file_name', ''),
        }

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        image_id = ann.get('image_id')
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Process each image
    total_images = 0
    total_annotations = 0

    for image_id, image_info in images_by_id.items():
        img_width = image_info['width']
        img_height = image_info['height']
        file_name = image_info['file_name']

        if img_width <= 0 or img_height <= 0:
            print(f"Warning: Image {file_name} has invalid dimensions {img_width}x{img_height}, skipping")
            continue

        # Get stem from file_name (without extension)
        stem = Path(file_name).stem
        output_txt_path = output_path / f"{stem}.txt"

        # Get annotations for this image
        image_annotations = annotations_by_image.get(image_id, [])

        yolo_lines = []
        for ann in image_annotations:
            # Skip crowd annotations
            if ann.get('iscrowd', 0) != 0:
                continue

            # Get category ID and map to class index
            cat_id = ann.get('category_id')
            if cat_id not in cat_id_to_idx:
                print(f"Warning: Category ID {cat_id} not in categories, skipping annotation")
                continue

            class_idx = cat_id_to_idx[cat_id]

            # Get bounding box
            bbox = ann.get('bbox', [])
            if len(bbox) != 4:
                print(f"Warning: Invalid bbox {bbox}, skipping")
                continue

            x1, y1, w, h = bbox

            # Convert COCO format (x1, y1, w, h) to YOLO format (xc, yc, w, h normalized)
            xc = (x1 + w / 2) / img_width
            yc = (y1 + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            # Ensure values are within [0, 1]
            xc = max(0, min(1, xc))
            yc = max(0, min(1, yc))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            yolo_lines.append(f"{class_idx} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Save YOLO file (even if empty, to mark images with no annotations)
        with open(output_txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        total_images += 1
        total_annotations += len(yolo_lines)

        if len(yolo_lines) > 0:
            print(f"  {stem}.txt: {len(yolo_lines)} annotations")

    print(f"\nConversion complete:")
    print(f"  Images processed: {total_images}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Classes: {len(class_names)}")
    print(f"  Output directory: {output_dir}")


