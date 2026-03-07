#!/usr/bin/env python3
"""
Basic visualization examples for DataFlow visualize module.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataflow import visualize


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


def example_coco_visualization():
    """Example: Visualize COCO annotations."""
    print("Example: COCO visualization")
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

        # Visualize
        result = visualize.visualize_coco(image_path, coco_path)
        vis_path = str(tmpdir / "coco_visualization.jpg")
        cv2.imwrite(vis_path, result)
        print(f"Visualization saved: {vis_path}")

        print()


def example_yolo_visualization():
    """Example: Visualize YOLO annotations."""
    print("Example: YOLO visualization")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test image
        image = create_test_image()
        image_path = str(tmpdir / "test_image.jpg")
        import cv2
        cv2.imwrite(image_path, image)

        # Create YOLO annotation
        yolo_path = str(tmpdir / "labels.txt")
        with open(yolo_path, 'w') as f:
            # class_idx, x_center, y_center, width, height (all normalized)
            f.write("0 0.25 0.25 0.25 0.333\n")  # Blue box
            f.write("1 0.625 0.375 0.25 0.333\n")  # Green box

        class_names = ["blue_box", "green_box"]

        print(f"Created YOLO file: {yolo_path}")
        print(f"Class names: {class_names}")

        # Visualize
        result = visualize.visualize_yolo(image_path, yolo_path, class_names)
        vis_path = str(tmpdir / "yolo_visualization.jpg")
        cv2.imwrite(vis_path, result)
        print(f"Visualization saved: {vis_path}")

        print()


def example_labelme_visualization():
    """Example: Visualize LabelMe annotations."""
    print("Example: LabelMe visualization")
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

        # Visualize
        result = visualize.visualize_labelme(image_path, labelme_path)
        vis_path = str(tmpdir / "labelme_visualization.jpg")
        cv2.imwrite(vis_path, result)
        print(f"Visualization saved: {vis_path}")

        print()


def example_labelme_polygon_visualization():
    """Example: Visualize LabelMe polygon annotations."""
    print("Example: LabelMe polygon visualization")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test image
        image = create_test_image()
        image_path = str(tmpdir / "test_image.jpg")
        import cv2
        cv2.imwrite(image_path, image)

        # Create LabelMe annotation with polygon
        labelme_data = {
            "version": "5.3.1",
            "flags": {},
            "shapes": [
                {
                    "label": "triangle",
                    "points": [[200, 50], [150, 150], [250, 150]],
                    "shape_type": "polygon"
                },
                {
                    "label": "pentagon",
                    "points": [[300, 100], [280, 150], [300, 200], [320, 200], [340, 150]],
                    "shape_type": "polygon"
                }
            ],
            "imagePath": "test_image.jpg",
            "imageData": None,
            "imageHeight": 300,
            "imageWidth": 400
        }

        labelme_path = str(tmpdir / "labelme_polygon.json")
        with open(labelme_path, 'w') as f:
            json.dump(labelme_data, f, indent=2)

        print(f"Created LabelMe polygon file: {labelme_path}")

        # Visualize
        result = visualize.visualize_labelme(image_path, labelme_path)
        vis_path = str(tmpdir / "labelme_polygon_visualization.jpg")
        cv2.imwrite(vis_path, result)
        print(f"Visualization saved: {vis_path}")

        print()


def example_cli_visualization():
    """Example: Using CLI for visualization."""
    print("Example: CLI visualization commands")
    print("-" * 40)

    print("Available visualization commands:")
    print("  dataflow visualize coco image.jpg annotation.json --save output.jpg")
    print("  dataflow visualize yolo image.jpg label.txt classes.txt --save output.jpg")
    print("  dataflow visualize labelme image.jpg annotation.json --save output.jpg")
    print()
    print("Options:")
    print("  --save PATH     Save visualization to file")
    print("  --show/--no-show  Show visualization window (default: --show)")
    print()


def main():
    """Run all visualization examples."""
    print("=" * 60)
    print("DataFlow Visualization Examples")
    print("=" * 60)
    print()

    examples = [
        example_coco_visualization,
        example_yolo_visualization,
        example_labelme_visualization,
        example_labelme_polygon_visualization,
        example_cli_visualization
    ]

    for example_func in examples:
        example_func()

    print("=" * 60)
    print("Visualization examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()