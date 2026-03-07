#!/usr/bin/env python3
"""
Basic conversion examples for DataFlow convert module.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataflow import convert


def create_test_image(width=400, height=300):
    """Create a test image with colored rectangles."""
    import cv2
    import numpy as np

    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw some colored rectangles
    cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(image, (200, 100), (300, 200), (0, 255, 0), -1)  # Green rectangle
    cv2.rectangle(image, (100, 200), (250, 250), (0, 0, 255), -1)  # Red rectangle

    return image


def example_labelme_to_coco():
    """Example: Convert LabelMe to COCO format."""
    print("Example: LabelMe to COCO conversion")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test image
        image = create_test_image()
        image_path = str(tmpdir / "test_image.jpg")
        import cv2
        cv2.imwrite(image_path, image)

        # Create LabelMe annotation
        labelme_data = {
            "version": "5.3.1",
            "flags": {},
            "shapes": [
                {
                    "label": "blue_box",
                    "points": [[50, 50], [150, 150]],
                    "shape_type": "rectangle"
                },
                {
                    "label": "green_box",
                    "points": [[200, 100], [300, 200]],
                    "shape_type": "rectangle"
                }
            ],
            "imagePath": "test_image.jpg",
            "imageData": None,
            "imageHeight": 300,
            "imageWidth": 400
        }

        labelme_path = str(tmpdir / "labelme.json")
        with open(labelme_path, 'w') as f:
            json.dump(labelme_data, f, indent=2)

        print(f"Created LabelMe file: {labelme_path}")

        # Convert to COCO
        coco_path = str(tmpdir / "coco.json")
        convert.labelme_to_coco(labelme_path, coco_path)
        print(f"Converted to COCO: {coco_path}")

        # Show COCO structure
        with open(coco_path, 'r') as f:
            coco_data = json.load(f)
            print(f"COCO contains {len(coco_data.get('annotations', []))} annotations")
            print(f"Classes: {[cat['name'] for cat in coco_data.get('categories', [])]}")

        print()


def example_coco_to_yolo():
    """Example: Convert COCO to YOLO format."""
    print("Example: COCO to YOLO conversion")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test image
        image = create_test_image()
        image_path = str(tmpdir / "test_image.jpg")
        import cv2
        cv2.imwrite(image_path, image)

        # Create COCO annotation
        coco_data = {
            "images": [{
                "id": 1,
                "width": 400,
                "height": 300,
                "file_name": "test_image.jpg"
            }],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [50, 50, 100, 100],  # x, y, w, h
                    "area": 10000,
                    "iscrowd": 0
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [200, 100, 100, 100],
                    "area": 10000,
                    "iscrowd": 0
                }
            ],
            "categories": [
                {"id": 1, "name": "blue_box", "supercategory": "object"},
                {"id": 2, "name": "green_box", "supercategory": "object"}
            ]
        }

        coco_path = str(tmpdir / "coco.json")
        with open(coco_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"Created COCO file: {coco_path}")

        # Convert to YOLO
        yolo_path = str(tmpdir / "yolo.txt")
        convert.coco_to_yolo(coco_path, image_path, yolo_path, ["blue_box", "green_box"])
        print(f"Converted to YOLO: {yolo_path}")

        # Show YOLO content
        with open(yolo_path, 'r') as f:
            print("YOLO format content:")
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_idx, xc, yc, w, h = parts
                    print(f"  Class {class_idx}: center=({xc}, {yc}), size=({w}, {h})")

        print()


def example_yolo_to_coco():
    """Example: Convert YOLO to COCO format."""
    print("Example: YOLO to COCO conversion")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test image
        image = create_test_image()
        image_path = str(tmpdir / "test_image.jpg")
        import cv2
        cv2.imwrite(image_path, image)

        # Create YOLO annotation
        yolo_path = str(tmpdir / "yolo.txt")
        with open(yolo_path, 'w') as f:
            # class_idx, x_center, y_center, width, height (all normalized)
            f.write("0 0.25 0.25 0.25 0.333\n")  # Blue box
            f.write("1 0.625 0.375 0.25 0.333\n")  # Green box

        class_names = ["blue_box", "green_box"]

        print(f"Created YOLO file: {yolo_path}")
        print(f"Class names: {class_names}")

        # Convert to COCO
        coco_path = str(tmpdir / "coco.json")
        convert.yolo_to_coco(yolo_path, image_path, class_names, coco_path)
        print(f"Converted to COCO: {coco_path}")

        # Show COCO structure
        with open(coco_path, 'r') as f:
            coco_data = json.load(f)
            print(f"COCO contains {len(coco_data.get('annotations', []))} annotations")

        print()


def example_coco_to_labelme():
    """Example: Convert COCO to LabelMe format."""
    print("Example: COCO to LabelMe conversion")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test image
        image = create_test_image()
        image_path = str(tmpdir / "test_image.jpg")
        import cv2
        cv2.imwrite(image_path, image)

        # Create COCO annotation
        coco_data = {
            "images": [{
                "id": 1,
                "width": 400,
                "height": 300,
                "file_name": "test_image.jpg"
            }],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [50, 50, 100, 100],
                    "area": 10000,
                    "iscrowd": 0
                }
            ],
            "categories": [
                {"id": 1, "name": "blue_box", "supercategory": "object"}
            ]
        }

        coco_path = str(tmpdir / "coco.json")
        with open(coco_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"Created COCO file: {coco_path}")

        # Convert to LabelMe
        labelme_path = str(tmpdir / "labelme.json")
        convert.coco_to_labelme(coco_path, image_path, labelme_path)
        print(f"Converted to LabelMe: {labelme_path}")

        # Show LabelMe structure
        with open(labelme_path, 'r') as f:
            labelme_data = json.load(f)
            print(f"LabelMe contains {len(labelme_data.get('shapes', []))} shapes")

        print()


def example_labelme_to_yolo():
    """Example: Convert LabelMe to YOLO format."""
    print("Example: LabelMe to YOLO conversion")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create LabelMe annotation (no image needed for this conversion)
        labelme_data = {
            "version": "5.3.1",
            "flags": {},
            "shapes": [
                {
                    "label": "blue_box",
                    "points": [[50, 50], [150, 150]],
                    "shape_type": "rectangle"
                }
            ],
            "imagePath": "test_image.jpg",
            "imageData": None,
            "imageHeight": 300,
            "imageWidth": 400
        }

        labelme_path = str(tmpdir / "labelme.json")
        with open(labelme_path, 'w') as f:
            json.dump(labelme_data, f, indent=2)

        print(f"Created LabelMe file: {labelme_path}")

        # Convert to YOLO
        yolo_path = str(tmpdir / "yolo.txt")
        convert.labelme_to_yolo(labelme_path, yolo_path, ["blue_box"])
        print(f"Converted to YOLO: {yolo_path}")

        # Show YOLO content
        with open(yolo_path, 'r') as f:
            print("YOLO format content:")
            for line in f:
                print(f"  {line.strip()}")

        print()


def example_yolo_to_labelme():
    """Example: Convert YOLO to LabelMe format."""
    print("Example: YOLO to LabelMe conversion")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test image
        image = create_test_image()
        image_path = str(tmpdir / "test_image.jpg")
        import cv2
        cv2.imwrite(image_path, image)

        # Create YOLO annotation
        yolo_path = str(tmpdir / "yolo.txt")
        with open(yolo_path, 'w') as f:
            f.write("0 0.25 0.25 0.25 0.333\n")  # Blue box

        class_names = ["blue_box"]

        print(f"Created YOLO file: {yolo_path}")
        print(f"Class names: {class_names}")

        # Convert to LabelMe
        labelme_path = str(tmpdir / "labelme.json")
        convert.yolo_to_labelme(yolo_path, image_path, class_names, labelme_path)
        print(f"Converted to LabelMe: {labelme_path}")

        # Show LabelMe structure
        with open(labelme_path, 'r') as f:
            labelme_data = json.load(f)
            print(f"LabelMe contains {len(labelme_data.get('shapes', []))} shapes")

        print()


def main():
    """Run all conversion examples."""
    print("=" * 60)
    print("DataFlow Conversion Examples")
    print("=" * 60)
    print()

    examples = [
        example_labelme_to_coco,
        example_coco_to_yolo,
        example_yolo_to_coco,
        example_coco_to_labelme,
        example_labelme_to_yolo,
        example_yolo_to_labelme
    ]

    for example_func in examples:
        example_func()

    print("=" * 60)
    print("Conversion examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()