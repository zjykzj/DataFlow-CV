"""
Convert COCO annotation to LabelMe format.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import base64


def coco_to_labelme(coco_json_path: str, image_path: str, output_json_path: str) -> None:
    """
    Convert COCO annotation to LabelMe format.

    Args:
        coco_json_path: Path to COCO JSON annotation file
        image_path: Path to corresponding image file
        output_json_path: Path to save LabelMe format annotation
    """
    # Validate paths
    if not Path(coco_json_path).exists():
        raise FileNotFoundError(f"COCO annotation file not found: {coco_json_path}")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load COCO annotation
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Get image size
    from .base import BaseConverter
    img_width, img_height = BaseConverter.get_image_size(image_path)

    # Find image information
    image_filename = Path(image_path).name
    image_info = None

    for img in coco_data.get('images', []):
        if img.get('file_name') == image_filename:
            image_info = img
            break

    # If not found, use first image or create default
    if image_info is None and coco_data.get('images'):
        image_info = coco_data['images'][0]

    if image_info:
        # Use dimensions from COCO if available
        img_width = image_info.get('width', img_width)
        img_height = image_info.get('height', img_height)

    # Build category ID to name mapping
    category_map = {}
    for cat in coco_data.get('categories', []):
        category_map[cat['id']] = cat['name']

    # Find annotations for this image
    image_id = image_info.get('id') if image_info else 1
    shapes = []

    for ann in coco_data.get('annotations', []):
        if ann.get('image_id') != image_id:
            continue

        # Skip crowd annotations
        if ann.get('iscrowd', 0) != 0:
            continue

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

        # Create rectangle points (top-left, bottom-right)
        points = [[float(x1), float(y1)], [float(x2), float(y2)]]

        shapes.append({
            "label": label,
            "points": points,
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None  # LabelMe 5.x format
        })

    # Build LabelMe format data
    labelme_data = {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,  # Can be set to base64 encoded image if needed
        "imageHeight": img_height,
        "imageWidth": img_width
    }

    # Optionally encode image data (commented out for now)
    # with open(image_path, 'rb') as f:
    #     image_data = base64.b64encode(f.read()).decode('utf-8')
    # labelme_data['imageData'] = image_data

    # Save to file
    output_dir = Path(output_json_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_data, f, ensure_ascii=False, indent=4)

    print(f"Converted {len(shapes)} annotations to LabelMe format")


def batch_coco_to_labelme(pairs: List[Tuple[str, str]], output_dir: str) -> None:
    """
    Convert multiple COCO annotations to LabelMe format.

    Args:
        pairs: List of (image_path, coco_json_path) tuples
        output_dir: Output directory for LabelMe JSON files
    """
    if not pairs:
        raise ValueError("No image-annotation pairs provided")

    successful = 0
    errors = 0

    for idx, (image_path, coco_json_path) in enumerate(pairs):
        try:
            # Generate output filename based on image name
            image_stem = Path(image_path).stem
            output_json_path = Path(output_dir) / f"{image_stem}.json"

            # Call single conversion function
            coco_to_labelme(coco_json_path, image_path, str(output_json_path))

            print(f"[{idx + 1}/{len(pairs)}] Converted: {Path(image_path).name} → {output_json_path.name}")
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