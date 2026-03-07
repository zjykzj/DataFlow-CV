"""
Convert YOLO annotation to LabelMe format.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple


def yolo_to_labelme(yolo_txt_path: str, image_path: str, class_names: List[str],
                    output_json_path: str) -> None:
    """
    Convert YOLO annotation to LabelMe format.

    Args:
        yolo_txt_path: Path to YOLO format annotation file
        image_path: Path to corresponding image file
        class_names: List of class names
        output_json_path: Path to save LabelMe format annotation
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
    shapes = []
    with open(yolo_txt_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                print(f"Warning: Line {line_num} has invalid format: {line}")
                continue

            try:
                class_idx = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                print(f"Warning: Line {line_num} has invalid values: {line}")
                continue

            # Validate class index
            if class_idx < 0 or class_idx >= len(class_names):
                print(f"Warning: Class index {class_idx} out of range for class names list")
                label = f"class_{class_idx}"
            else:
                label = class_names[class_idx]

            # Denormalize coordinates
            xc_abs = xc * img_width
            yc_abs = yc * img_height
            w_abs = w * img_width
            h_abs = h * img_height

            # Convert from (xc, yc, w, h) to rectangle corners
            x1 = xc_abs - w_abs / 2
            y1 = yc_abs - h_abs / 2
            x2 = x1 + w_abs
            y2 = y1 + h_abs

            # Ensure coordinates are within image bounds
            x1 = max(0, min(img_width - 1, x1))
            y1 = max(0, min(img_height - 1, y1))
            x2 = max(0, min(img_width - 1, x2))
            y2 = max(0, min(img_height - 1, y2))

            # Create rectangle points
            points = [[float(x1), float(y1)], [float(x2), float(y2)]]

            shapes.append({
                "label": label,
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            })

    # Build LabelMe format data
    image_filename = Path(image_path).name
    labelme_data = {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width
    }

    # Save to file
    output_dir = Path(output_json_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_data, f, ensure_ascii=False, indent=4)

    print(f"Converted {len(shapes)} annotations to LabelMe format")


def batch_yolo_to_labelme(pairs: List[Tuple[str, str]], class_names: List[str],
                          output_dir: str) -> None:
    """
    Convert multiple YOLO annotations to LabelMe format.

    Args:
        pairs: List of (image_path, yolo_txt_path) tuples
        class_names: List of class names
        output_dir: Output directory for LabelMe JSON files
    """
    if not pairs:
        raise ValueError("No image-annotation pairs provided")

    successful = 0
    errors = 0

    for idx, (image_path, yolo_txt_path) in enumerate(pairs):
        try:
            # Generate output filename based on image name
            image_stem = Path(image_path).stem
            output_json_path = Path(output_dir) / f"{image_stem}.json"

            # Call single conversion function
            yolo_to_labelme(yolo_txt_path, image_path, class_names, str(output_json_path))

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