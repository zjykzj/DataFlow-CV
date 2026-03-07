"""
Convert LabelMe annotation to COCO format.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime


def labelme_to_coco(labelme_json_path: str, output_json_path: str) -> None:
    """
    Convert LabelMe annotation to COCO format.

    Args:
        labelme_json_path: Path to LabelMe JSON annotation file
        output_json_path: Path to save COCO format annotation
    """
    # Validate path
    if not Path(labelme_json_path).exists():
        raise FileNotFoundError(f"LabelMe annotation file not found: {labelme_json_path}")

    # Load LabelMe annotation
    with open(labelme_json_path, 'r') as f:
        labelme_data = json.load(f)

    # Extract image information
    image_path = labelme_data.get('imagePath', '')
    image_height = labelme_data.get('imageHeight', 0)
    image_width = labelme_data.get('imageWidth', 0)

    if image_height == 0 or image_width == 0:
        raise ValueError("Invalid image dimensions in LabelMe annotation")

    # Process shapes
    shapes = labelme_data.get('shapes', [])
    if not shapes:
        print("Warning: No shapes found in LabelMe annotation")

    # Collect unique class names
    class_names = []
    class_name_to_id = {}

    # Process annotations
    coco_annotations = []
    for idx, shape in enumerate(shapes):
        label = shape.get('label', '')
        shape_type = shape.get('shape_type', '')
        points = shape.get('points', [])

        if not label:
            print(f"Warning: Shape {idx} has no label, skipping")
            continue

        # Register class name
        if label not in class_name_to_id:
            class_name_to_id[label] = len(class_names) + 1  # COCO IDs start from 1
            class_names.append(label)

        # Get bounding box based on shape type
        if shape_type == 'rectangle' and len(points) == 2:
            # Rectangle: points are [top-left, bottom-right]
            x1 = min(points[0][0], points[1][0])
            y1 = min(points[0][1], points[1][1])
            x2 = max(points[0][0], points[1][0])
            y2 = max(points[0][1], points[1][1])

            width = x2 - x1
            height = y2 - y1

        elif shape_type == 'polygon' and len(points) >= 3:
            # Polygon: compute bounding box
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            x1 = min(x_coords)
            y1 = min(y_coords)
            x2 = max(x_coords)
            y2 = max(y_coords)

            width = x2 - x1
            height = y2 - y1
        else:
            print(f"Warning: Unsupported shape type '{shape_type}' or invalid points, skipping")
            continue

        # Ensure coordinates are within image bounds
        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        width = max(1, min(image_width - x1, width))
        height = max(1, min(image_height - y1, height))

        # Create COCO annotation
        coco_ann = {
            'id': idx + 1,
            'image_id': 1,  # Single image
            'category_id': class_name_to_id[label],
            'bbox': [float(x1), float(y1), float(width), float(height)],
            'area': float(width * height),
            'iscrowd': 0,
            'segmentation': [],  # Empty for detection
        }
        coco_annotations.append(coco_ann)

    # Build categories
    categories = []
    for idx, name in enumerate(class_names):
        categories.append({
            'id': idx + 1,
            'name': name,
            'supercategory': name
        })

    # Build images entry
    image_filename = Path(image_path).name if image_path else Path(labelme_json_path).stem + '.jpg'
    images = [{
        'id': 1,
        'width': image_width,
        'height': image_height,
        'file_name': image_filename,
        'license': 1,
        'date_captured': datetime.now().isoformat()
    }]

    # Build full COCO structure
    coco_data = {
        'info': {
            'description': 'Converted from LabelMe format',
            'url': '',
            'version': '1.0',
            'year': datetime.now().year,
            'contributor': 'DataFlow',
            'date_created': datetime.now().isoformat()
        },
        'licenses': [{
            'id': 1,
            'name': 'Unknown',
            'url': ''
        }],
        'images': images,
        'annotations': coco_annotations,
        'categories': categories
    }

    # Save to file
    output_dir = Path(output_json_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"Converted {len(coco_annotations)} annotations to COCO format")
    print(f"Classes found: {class_names}")


def batch_labelme_to_coco(pairs: List[Tuple[str, str]], output_json_path: str) -> None:
    """
    Convert multiple LabelMe annotations to a single COCO format file.

    Args:
        pairs: List of (annotation_path, annotation_path) tuples (both same for LabelMe)
        output_json_path: Path to save combined COCO format annotation file
    """
    if not pairs:
        raise ValueError("No annotation files provided")

    all_images = []
    all_annotations = []
    all_categories = []

    image_id = 1
    annotation_id = 1
    category_name_to_id = {}
    next_category_id = 1

    # Process each annotation file
    for idx, (ann_path, _) in enumerate(pairs):
        try:
            # Load LabelMe annotation
            with open(ann_path, 'r') as f:
                labelme_data = json.load(f)

            # Extract image information
            image_filename = labelme_data.get('imagePath', '')
            if not image_filename:
                image_filename = Path(ann_path).stem + '.jpg'

            image_height = labelme_data.get('imageHeight', 0)
            image_width = labelme_data.get('imageWidth', 0)

            if image_height == 0 or image_width == 0:
                print(f"Warning: Invalid image dimensions in {Path(ann_path).name}, skipping")
                continue

            # Create image entry
            image_entry = {
                'id': image_id,
                'width': image_width,
                'height': image_height,
                'file_name': image_filename,
                'license': 1,
                'date_captured': datetime.now().isoformat()
            }
            all_images.append(image_entry)

            # Process shapes
            shapes = labelme_data.get('shapes', [])
            for shape in shapes:
                label = shape.get('label', '')
                shape_type = shape.get('shape_type', '')
                points = shape.get('points', [])

                if not label:
                    continue

                # Register category if new
                if label not in category_name_to_id:
                    category_name_to_id[label] = next_category_id
                    all_categories.append({
                        'id': next_category_id,
                        'name': label,
                        'supercategory': label
                    })
                    next_category_id += 1

                # Get bounding box
                if shape_type == 'rectangle' and len(points) == 2:
                    x1 = min(points[0][0], points[1][0])
                    y1 = min(points[0][1], points[1][1])
                    x2 = max(points[0][0], points[1][0])
                    y2 = max(points[0][1], points[1][1])
                elif shape_type == 'polygon' and len(points) >= 3:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x1 = min(x_coords)
                    y1 = min(y_coords)
                    x2 = max(x_coords)
                    y2 = max(y_coords)
                else:
                    continue

                width = x2 - x1
                height = y2 - y1

                # Ensure coordinates are within image bounds
                x1 = max(0, min(image_width - 1, x1))
                y1 = max(0, min(image_height - 1, y1))
                width = max(1, min(image_width - x1, width))
                height = max(1, min(image_height - y1, height))

                # Create COCO annotation
                coco_ann = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_name_to_id[label],
                    'bbox': [float(x1), float(y1), float(width), float(height)],
                    'area': float(width * height),
                    'iscrowd': 0,
                    'segmentation': []
                }
                all_annotations.append(coco_ann)
                annotation_id += 1

            print(f"[{idx + 1}/{len(pairs)}] Processed: {Path(ann_path).name} ({len(shapes)} shapes)")
            image_id += 1

        except Exception as e:
            print(f"[{idx + 1}/{len(pairs)}] Error processing {Path(ann_path).name}: {e}")
            print("  Skipping...")
            continue

    if not all_images:
        raise ValueError("No valid images processed")

    # Build full COCO structure
    coco_data = {
        'info': {
            'description': 'Converted from LabelMe format (batch)',
            'url': '',
            'version': '1.0',
            'year': datetime.now().year,
            'contributor': 'DataFlow',
            'date_created': datetime.now().isoformat()
        },
        'licenses': [{
            'id': 1,
            'name': 'Unknown',
            'url': ''
        }],
        'images': all_images,
        'annotations': all_annotations,
        'categories': all_categories
    }

    # Save to file
    output_dir = Path(output_json_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"\nBatch conversion complete.")
    print(f"  Images: {len(all_images)}")
    print(f"  Annotations: {len(all_annotations)}")
    print(f"  Categories: {len(all_categories)}")
    print(f"  Saved to: {output_json_path}")