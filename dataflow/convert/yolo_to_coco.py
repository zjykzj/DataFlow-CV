"""
Convert YOLO annotation to COCO format.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime


def yolo_to_coco(images_dir: str, labels_dir: str, class_names_path: str,
                 output_json_path: str) -> None:
    """
    Convert YOLO annotations to COCO format.

    Args:
        images_dir: Directory containing image files
        labels_dir: Directory containing YOLO .txt annotation files
        class_names_path: Path to class names file (one per line)
        output_json_path: Path to save COCO format annotation file
    """
    from pathlib import Path
    from .base import BaseConverter

    # Validate directories
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Read class names
    if not Path(class_names_path).exists():
        raise FileNotFoundError(f"Class names file not found: {class_names_path}")

    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]

    if not class_names:
        raise ValueError("Class names file is empty or contains no valid lines")

    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_path.glob(f"*{ext}"))
        image_files.extend(images_path.glob(f"*{ext.upper()}"))

    if not image_files:
        raise ValueError(f"No image files found in {images_dir}. Supported extensions: {image_extensions}")

    # Prepare COCO data structures
    images = []
    annotations = []
    categories = []

    # Build categories (COCO IDs start from 1)
    for idx, name in enumerate(class_names):
        categories.append({
            'id': idx + 1,
            'name': name,
            'supercategory': name
        })

    image_id = 1
    annotation_id = 1
    processed_count = 0
    skipped_count = 0

    print(f"Found {len(image_files)} image files")

    for image_file in sorted(image_files):
        # Find corresponding label file
        label_file = labels_path / f"{image_file.stem}.txt"

        if not label_file.exists():
            # Try alternative extensions
            found = False
            for alt_ext in ['.txt', '.TXT']:
                alt_file = labels_path / f"{image_file.stem}{alt_ext}"
                if alt_file.exists():
                    label_file = alt_file
                    found = True
                    break

            if not found:
                skipped_count += 1
                continue

        try:
            # Get image dimensions
            img_width, img_height = BaseConverter.get_image_size(str(image_file))

            # Read YOLO annotations
            with open(label_file, 'r') as f:
                yolo_lines = f.readlines()

            image_annotations = []
            for line in yolo_lines:
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
                    print(f"Warning: Class index {class_idx} out of range for {image_file.name}, skipping annotation")
                    continue

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

                # Create annotation
                image_annotations.append({
                    'class_idx': class_idx,
                    'bbox': [float(x1), float(y1), float(w_abs), float(h_abs)],
                    'area': float(w_abs * h_abs),
                    'iscrowd': 0
                })

            # Add image entry
            images.append({
                'id': image_id,
                'width': img_width,
                'height': img_height,
                'file_name': image_file.name,
                'license': 1,
                'date_captured': datetime.now().isoformat()
            })

            # Add annotations with unique IDs
            for ann in image_annotations:
                annotations.append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': ann['class_idx'] + 1,  # COCO category IDs start from 1
                    'bbox': ann['bbox'],
                    'area': ann['area'],
                    'iscrowd': ann['iscrowd'],
                    'segmentation': []  # Empty for detection
                })
                annotation_id += 1

            processed_count += 1
            if len(image_annotations) > 0:
                print(f"  {image_file.name}: {len(image_annotations)} annotations")
            else:
                print(f"  {image_file.name}: no annotations")

        except Exception as e:
            print(f"Warning: Failed to process {image_file.name}: {e}")
            skipped_count += 1
            continue

        image_id += 1

    if not images:
        raise ValueError("No valid images processed. Check that image and label files match.")

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
        'annotations': annotations,
        'categories': categories
    }

    # Save to file
    output_dir = Path(output_json_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"\nConversion complete:")
    print(f"  Images processed: {processed_count}")
    print(f"  Images skipped: {skipped_count}")
    print(f"  Total annotations: {len(annotations)}")
    print(f"  Classes: {len(class_names)}")
    print(f"  Saved to: {output_json_path}")


