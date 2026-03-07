"""
Convert YOLO annotation to COCO format.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime


def yolo_to_coco(yolo_txt_path: str, image_path: str, class_names: List[str],
                 output_json_path: str) -> None:
    """
    Convert YOLO annotation to COCO format.

    Args:
        yolo_txt_path: Path to YOLO format annotation file
        image_path: Path to corresponding image file
        class_names: List of class names
        output_json_path: Path to save COCO format annotation
    """
    # Validate paths
    if not Path(yolo_txt_path).exists():
        raise FileNotFoundError(f"YOLO annotation file not found: {yolo_txt_path}")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Get image size
    from .base import BaseConverter
    img_width, img_height = BaseConverter.get_image_size(image_path)

    # Read YOLO annotations
    annotations = []
    with open(yolo_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                class_idx = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue

            # Validate class index
            if class_idx < 0 or class_idx >= len(class_names):
                print(f"Warning: Class index {class_idx} out of range for class names list")
                class_name = f"class_{class_idx}"
            else:
                class_name = class_names[class_idx]

            # Denormalize coordinates
            xc_abs = xc * img_width
            yc_abs = yc * img_height
            w_abs = w * img_width
            h_abs = h * img_height

            # Convert from (xc, yc, w, h) to COCO format (x1, y1, w, h)
            x1 = xc_abs - w_abs / 2
            y1 = yc_abs - h_abs / 2

            # Ensure coordinates are within image bounds
            x1 = max(0, min(img_width - 1, x1))
            y1 = max(0, min(img_height - 1, y1))
            w_abs = max(1, min(img_width - x1, w_abs))
            h_abs = max(1, min(img_height - y1, h_abs))

            annotations.append({
                'class_idx': class_idx,
                'class_name': class_name,
                'bbox': [float(x1), float(y1), float(w_abs), float(h_abs)],
                'area': float(w_abs * h_abs),
                'iscrowd': 0
            })

    # Create COCO format data
    image_filename = Path(image_path).name
    image_id = 1  # Single image, use ID 1

    # Build categories
    categories = []
    for idx, name in enumerate(class_names):
        categories.append({
            'id': idx + 1,  # COCO category IDs start from 1
            'name': name,
            'supercategory': name
        })

    # Build annotations with unique IDs
    coco_annotations = []
    for idx, ann in enumerate(annotations):
        coco_ann = {
            'id': idx + 1,  # Annotation ID
            'image_id': image_id,
            'category_id': ann['class_idx'] + 1,  # COCO category IDs start from 1
            'bbox': ann['bbox'],
            'area': ann['area'],
            'iscrowd': ann['iscrowd'],
            'segmentation': [],  # Empty for detection
        }
        coco_annotations.append(coco_ann)

    # Build images entry
    images = [{
        'id': image_id,
        'width': img_width,
        'height': img_height,
        'file_name': image_filename,
        'license': 1,
        'date_captured': datetime.now().isoformat()
    }]

    # Build full COCO structure
    coco_data = {
        'info': {
            'description': 'Converted from YOLO format',
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

    print(f"Converted {len(annotations)} annotations to COCO format")
    print(f"Saved to {output_json_path}")


def process_yolo_file(yolo_txt_path: str, image_path: str, class_names: List[str],
                      image_id: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Process a single YOLO file and return image info and annotations.

    Args:
        yolo_txt_path: Path to YOLO format annotation file
        image_path: Path to corresponding image file
        class_names: List of class names
        image_id: Unique image ID for COCO format

    Returns:
        Tuple of (image_info, annotations) for COCO format
    """
    if not Path(yolo_txt_path).exists():
        raise FileNotFoundError(f"YOLO annotation file not found: {yolo_txt_path}")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Get image size
    from .base import BaseConverter
    img_width, img_height = BaseConverter.get_image_size(image_path)

    # Read YOLO annotations
    annotations = []
    with open(yolo_txt_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                class_idx = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue

            # Validate class index
            if class_idx < 0 or class_idx >= len(class_names):
                class_name = f"class_{class_idx}"
            else:
                class_name = class_names[class_idx]

            # Denormalize coordinates
            xc_abs = xc * img_width
            yc_abs = yc * img_height
            w_abs = w * img_width
            h_abs = h * img_height

            # Convert from (xc, yc, w, h) to COCO format (x1, y1, w, h)
            x1 = xc_abs - w_abs / 2
            y1 = yc_abs - h_abs / 2

            # Ensure coordinates are within image bounds
            x1 = max(0, min(img_width - 1, x1))
            y1 = max(0, min(img_height - 1, y1))
            w_abs = max(1, min(img_width - x1, w_abs))
            h_abs = max(1, min(img_height - y1, h_abs))

            annotations.append({
                'class_idx': class_idx,
                'class_name': class_name,
                'bbox': [float(x1), float(y1), float(w_abs), float(h_abs)],
                'area': float(w_abs * h_abs),
                'iscrowd': 0
            })

    # Build image info
    image_filename = Path(image_path).name
    image_info = {
        'id': image_id,
        'width': img_width,
        'height': img_height,
        'file_name': image_filename,
        'license': 1,
        'date_captured': datetime.now().isoformat()
    }

    # Build annotations with unique IDs (will be set in batch function)
    coco_annotations = []
    for ann_idx, ann in enumerate(annotations):
        coco_ann = {
            'class_idx': ann['class_idx'],  # Store for reference
            'bbox': ann['bbox'],
            'area': ann['area'],
            'iscrowd': ann['iscrowd']
        }
        coco_annotations.append(coco_ann)

    return image_info, coco_annotations


def batch_yolo_to_coco(pairs: List[Tuple[str, str]], class_names: List[str],
                       output_json_path: str) -> None:
    """
    Convert multiple YOLO annotations to a single COCO format file.

    Args:
        pairs: List of (image_path, yolo_txt_path) tuples
        class_names: List of class names
        output_json_path: Path to save COCO format annotation file
    """
    if not pairs:
        raise ValueError("No image-annotation pairs provided")

    all_images = []
    all_annotations = []
    annotation_id = 1

    # Process each pair
    for idx, (image_path, yolo_txt_path) in enumerate(pairs):
        try:
            image_id = idx + 1  # COCO image IDs start from 1
            image_info, annotations = process_yolo_file(
                yolo_txt_path, image_path, class_names, image_id
            )

            all_images.append(image_info)

            # Add annotations with proper IDs
            for ann in annotations:
                coco_ann = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': ann['class_idx'] + 1,  # COCO category IDs start from 1
                    'bbox': ann['bbox'],
                    'area': ann['area'],
                    'iscrowd': ann['iscrowd'],
                    'segmentation': []  # Empty for detection
                }
                all_annotations.append(coco_ann)
                annotation_id += 1

            print(f"Processed: {Path(image_path).name} ({len(annotations)} annotations)")

        except Exception as e:
            print(f"Error processing {Path(image_path).name}: {e}")
            print("Skipping...")
            continue

    if not all_images:
        raise ValueError("No valid images processed")

    # Build categories
    categories = []
    for idx, name in enumerate(class_names):
        categories.append({
            'id': idx + 1,  # COCO category IDs start from 1
            'name': name,
            'supercategory': name
        })

    # Build full COCO structure
    coco_data = {
        'info': {
            'description': 'Converted from YOLO format (batch)',
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
        'categories': categories
    }

    # Save to file
    output_dir = Path(output_json_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"\nBatch conversion complete.")
    print(f"  Images: {len(all_images)}")
    print(f"  Annotations: {len(all_annotations)}")
    print(f"  Categories: {len(categories)}")
    print(f"  Saved to: {output_json_path}")