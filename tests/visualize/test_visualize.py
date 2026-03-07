#!/usr/bin/env python3
"""
Test for DataFlow visualize module.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path to import dataflow
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataflow import visualize


def test_coco_visualization():
    """Test COCO visualization."""
    print("Testing COCO visualization...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test image
        import cv2
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_path = str(tmpdir / "test.jpg")
        cv2.imwrite(image_path, test_image)

        # Create COCO annotation
        coco_data = {
            "images": [{
                "id": 1,
                "width": 100,
                "height": 100,
                "file_name": "test.jpg"
            }],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 20, 20],
                    "area": 400,
                    "iscrowd": 0
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [50, 50, 30, 30],
                    "area": 900,
                    "iscrowd": 0
                }
            ],
            "categories": [
                {"id": 1, "name": "cat", "supercategory": "cat"},
                {"id": 2, "name": "dog", "supercategory": "dog"}
            ]
        }

        coco_path = str(tmpdir / "coco.json")
        with open(coco_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"  Created test COCO file: {coco_path}")

        # Test visualization
        try:
            result = visualize.visualize_coco(image_path, coco_path)
            print(f"  ✓ COCO visualization successful")

            # Save result
            vis_path = str(tmpdir / "coco_vis.jpg")
            cv2.imwrite(vis_path, result)
            print(f"  ✓ Saved visualization to {vis_path}")

            # Check result dimensions
            assert result.shape == (100, 100, 3), f"Unexpected shape: {result.shape}"
            print(f"  ✓ Output image has correct shape: {result.shape}")

        except Exception as e:
            print(f"  ✗ COCO visualization failed: {e}")
            return False

    return True


def test_yolo_visualization():
    """Test YOLO visualization."""
    print("Testing YOLO visualization...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test image
        import cv2
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_path = str(tmpdir / "test.jpg")
        cv2.imwrite(image_path, test_image)

        # Create YOLO annotation
        yolo_path = str(tmpdir / "yolo.txt")
        with open(yolo_path, 'w') as f:
            f.write("0 0.2 0.2 0.1 0.1\n")  # cat at (20,20) with size 10x10
            f.write("1 0.6 0.6 0.1 0.1\n")  # dog at (60,60) with size 10x10

        print(f"  Created test YOLO file: {yolo_path}")

        # Test visualization
        try:
            result = visualize.visualize_yolo(image_path, yolo_path, ["cat", "dog"])
            print(f"  ✓ YOLO visualization successful")

            # Save result
            vis_path = str(tmpdir / "yolo_vis.jpg")
            cv2.imwrite(vis_path, result)
            print(f"  ✓ Saved visualization to {vis_path}")

            # Check result dimensions
            assert result.shape == (100, 100, 3), f"Unexpected shape: {result.shape}"
            print(f"  ✓ Output image has correct shape: {result.shape}")

        except Exception as e:
            print(f"  ✗ YOLO visualization failed: {e}")
            return False

    return True


def test_labelme_visualization():
    """Test LabelMe visualization."""
    print("Testing LabelMe visualization...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test image
        import cv2
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_path = str(tmpdir / "test.jpg")
        cv2.imwrite(image_path, test_image)

        # Create LabelMe annotation
        labelme_data = {
            "version": "5.3.1",
            "flags": {},
            "shapes": [
                {
                    "label": "cat",
                    "points": [[10, 10], [30, 30]],
                    "shape_type": "rectangle"
                },
                {
                    "label": "dog",
                    "points": [[50, 50], [80, 80]],
                    "shape_type": "rectangle"
                }
            ],
            "imagePath": "test.jpg",
            "imageData": None,
            "imageHeight": 100,
            "imageWidth": 100
        }

        labelme_path = str(tmpdir / "labelme.json")
        with open(labelme_path, 'w') as f:
            json.dump(labelme_data, f, indent=2)

        print(f"  Created test LabelMe file: {labelme_path}")

        # Test visualization
        try:
            result = visualize.visualize_labelme(image_path, labelme_path)
            print(f"  ✓ LabelMe visualization successful")

            # Save result
            vis_path = str(tmpdir / "labelme_vis.jpg")
            cv2.imwrite(vis_path, result)
            print(f"  ✓ Saved visualization to {vis_path}")

            # Check result dimensions
            assert result.shape == (100, 100, 3), f"Unexpected shape: {result.shape}"
            print(f"  ✓ Output image has correct shape: {result.shape}")

        except Exception as e:
            print(f"  ✗ LabelMe visualization failed: {e}")
            return False

    return True


def test_labelme_polygon_visualization():
    """Test LabelMe polygon visualization."""
    print("Testing LabelMe polygon visualization...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test image
        import cv2
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_path = str(tmpdir / "test.jpg")
        cv2.imwrite(image_path, test_image)

        # Create LabelMe annotation with polygon
        labelme_data = {
            "version": "5.3.1",
            "flags": {},
            "shapes": [
                {
                    "label": "triangle",
                    "points": [[50, 10], [30, 50], [70, 50]],
                    "shape_type": "polygon"
                }
            ],
            "imagePath": "test.jpg",
            "imageData": None,
            "imageHeight": 100,
            "imageWidth": 100
        }

        labelme_path = str(tmpdir / "labelme_polygon.json")
        with open(labelme_path, 'w') as f:
            json.dump(labelme_data, f, indent=2)

        print(f"  Created test LabelMe polygon file: {labelme_path}")

        # Test visualization
        try:
            result = visualize.visualize_labelme(image_path, labelme_path)
            print(f"  ✓ LabelMe polygon visualization successful")

            # Save result
            vis_path = str(tmpdir / "labelme_polygon_vis.jpg")
            cv2.imwrite(vis_path, result)
            print(f"  ✓ Saved visualization to {vis_path}")

        except Exception as e:
            print(f"  ✗ LabelMe polygon visualization failed: {e}")
            return False

    return True


def main():
    """Run all visualize tests."""
    print("=" * 60)
    print("DataFlow Visualize Module Tests")
    print("=" * 60)

    all_passed = True

    # Test imports
    print("\nTesting imports...")
    try:
        from dataflow import visualize
        print("  ✓ Visualize module imports successfully")
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        all_passed = False
        return

    # Run tests
    tests = [
        test_coco_visualization,
        test_yolo_visualization,
        test_labelme_visualization,
        test_labelme_polygon_visualization
    ]

    for test_func in tests:
        if test_func():
            print(f"  ✓ {test_func.__name__} PASSED")
        else:
            print(f"  ✗ {test_func.__name__} FAILED")
            all_passed = False
        print()

    print("=" * 60)
    if all_passed:
        print("All visualize tests PASSED! 🎉")
    else:
        print("Some visualize tests FAILED! ❌")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)