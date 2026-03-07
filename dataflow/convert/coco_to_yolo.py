"""
Convert COCO annotation to YOLO format.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from .base import BaseConverter


def coco_to_yolo(coco_json_path: str, image_path: str, output_txt_path: str,
                 class_names: List[str] = None) -> None:
    """
    Convert COCO annotation to YOLO format.

    Args:
        coco_json_path: Path to COCO JSON annotation file
        image_path: Path to corresponding image file
        output_txt_path: Path to save YOLO format annotation
        class_names: List of class names (optional, will be extracted from COCO if not provided)
    """
    # Validate paths
    BaseConverter.validate_paths(coco_json_path, image_path)

    # Load COCO annotation
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Get image size
    img_width, img_height = BaseConverter.get_image_size(image_path)

    # Extract class names from COCO if not provided
    if class_names is None or len(class_names) == 0:
        if 'categories' in coco_data:
            class_names = [cat['name'] for cat in sorted(coco_data['categories'], key=lambda x: x['id'])]
        else:
            raise ValueError("No class names provided and cannot extract from COCO annotation")

    # Create mapping from category_id to class index
    cat_id_to_idx = {}
    if 'categories' in coco_data:
        for cat in coco_data['categories']:
            if cat['name'] in class_names:
                cat_id_to_idx[cat['id']] = class_names.index(cat['name'])
            else:
                # If category not in class_names, assign next available index
                cat_id_to_idx[cat['id']] = len(class_names)
                class_names.append(cat['name'])

    # Find annotations for this image
    image_filename = Path(image_path).name
    image_id = None

    # Try to find image by filename
    for img in coco_data.get('images', []):
        if img.get('file_name') == image_filename:
            image_id = img['id']
            break

    # If not found by filename, use first image (assuming single image annotation)
    if image_id is None and 'images' in coco_data and len(coco_data['images']) > 0:
        image_id = coco_data['images'][0]['id']

    if image_id is None:
        raise ValueError(f"Could not find image information in COCO annotation for {image_filename}")

    # Collect annotations for this image
    annotations = []
    for ann in coco_data.get('annotations', []):
        if ann.get('image_id') == image_id:
            # Skip crowd annotations
            if ann.get('iscrowd', 0) != 0:
                continue

            # Get class index
            cat_id = ann.get('category_id')
            if cat_id not in cat_id_to_idx:
                # Create new mapping if category not seen before
                cat_id_to_idx[cat_id] = len(class_names)
                class_names.append(f"class_{cat_id}")

            class_idx = cat_id_to_idx[cat_id]

            # Get bounding box
            bbox = ann.get('bbox', [])
            if len(bbox) != 4:
                continue

            # Convert COCO format (x1, y1, w, h) to YOLO format (xc, yc, w, h normalized)
            x1, y1, w, h = bbox

            # Normalize
            xc = (x1 + w / 2) / img_width
            yc = (y1 + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            # Ensure values are within [0, 1]
            xc = max(0, min(1, xc))
            yc = max(0, min(1, yc))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            annotations.append([class_idx, xc, yc, w_norm, h_norm])

    # Save YOLO format
    output_dir = Path(output_txt_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_txt_path, 'w') as f:
        for ann in annotations:
            line = f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}"
            f.write(line + '\n')

    print(f"Converted {len(annotations)} annotations to YOLO format")
    print(f"Class names used: {class_names}")


def batch_coco_to_yolo(pairs: List[Tuple[str, str]], class_names: List[str] = None,
                       output_dir: str = None) -> None:
    """
    Convert multiple COCO annotations to YOLO format.

    Args:
        pairs: List of (image_path, coco_json_path) tuples
        class_names: List of class names (optional, will be extracted from COCO if not provided)
        output_dir: Output directory for YOLO files (optional, required for per-file mode)
    """
    if not pairs:
        raise ValueError("No image-annotation pairs provided")

    successful = 0
    errors = 0

    for idx, (image_path, coco_json_path) in enumerate(pairs):
        try:
            # Generate output filename based on image name
            image_stem = Path(image_path).stem
            output_txt_path = Path(output_dir) / f"{image_stem}.txt"

            # Call single conversion function
            coco_to_yolo(coco_json_path, image_path, str(output_txt_path), class_names)

            print(f"[{idx + 1}/{len(pairs)}] Converted: {Path(image_path).name} → {output_txt_path.name}")
            successful += 1

        except Exception as e:
            print(f"[{idx + 1}/{len(pairs)}] Error processing {Path(image_path).name}: {e}")
            print("  Skipping...")
            errors += 1
            continue

    print(f"\nBatch conversion complete.")
    print(f"  Successfully converted: {successful}")
    if errors > 0:
        print(f"  Errors: {errors}")