"""
Visualize YOLO annotations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List

from .base import BaseVisualizer


def visualize_yolo(image_path: str, yolo_txt_path: str, class_names: List[str]) -> np.ndarray:
    """
    Visualize YOLO annotations on image.

    Args:
        image_path: Path to image file
        yolo_txt_path: Path to YOLO format annotation file
        class_names: List of class names

    Returns:
        Image with annotations drawn
    """
    import sys
    print(f"DEBUG visualize_yolo ENTER: image='{image_path}', yolo='{yolo_txt_path}'", file=sys.stderr, flush=True)
    print(f"DEBUG visualize_yolo ENTER stdout", flush=True)
    # Validate paths
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not Path(yolo_txt_path).exists():
        raise FileNotFoundError(f"YOLO annotation file not found: {yolo_txt_path}")

    # Load image
    image = BaseVisualizer.load_image(image_path)
    img_height, img_width = image.shape[:2]

    # Read YOLO annotations
    annotations = []
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

            annotations.append({
                'class_idx': class_idx,
                'xc': xc,
                'yc': yc,
                'w': w,
                'h': h
            })

    if not annotations:
        print("Warning: No annotations found in YOLO file")
        return image

    # Draw annotations
    for ann in annotations:
        class_idx = ann['class_idx']
        xc = ann['xc']
        yc = ann['yc']
        w = ann['w']
        h = ann['h']

        # Validate class index
        if class_idx < 0 or class_idx >= len(class_names):
            label = f"class_{class_idx}"
            print(f"Warning: Class index {class_idx} out of range for class names list")
        else:
            label = class_names[class_idx]

        # Denormalize coordinates
        xc_abs = xc * img_width
        yc_abs = yc * img_height
        w_abs = w * img_width
        h_abs = h * img_height

        # Convert from (xc, yc, w, h) to (x1, y1, x2, y2)
        x1 = xc_abs - w_abs / 2
        y1 = yc_abs - h_abs / 2
        x2 = x1 + w_abs
        y2 = y1 + h_abs

        # Ensure coordinates are within image bounds
        x1 = max(0, min(img_width - 1, x1))
        y1 = max(0, min(img_height - 1, y1))
        x2 = max(0, min(img_width - 1, x2))
        y2 = max(0, min(img_height - 1, y2))

        # Get color for this class
        color = BaseVisualizer.get_color(class_idx, len(class_names))

        # Debug output
        print(f"  YOLO bbox: class={class_idx}({label}), xyxy=[{int(x1)},{int(y1)},{int(x2)},{int(y2)}] from normalized: {xc:.6f},{yc:.6f},{w:.6f},{h:.6f}")

        # Draw bounding box
        image = BaseVisualizer.draw_bbox(
            image,
            [x1, y1, x2, y2],
            color=color,
            thickness=2,
            label=label
        )

    print(f"Visualized {len(annotations)} YOLO annotations")
    return image