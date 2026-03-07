#!/usr/bin/env python3
"""
Test for DataFlow convert module.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path to import dataflow
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataflow import convert


def test_labelme_to_coco():
    """Test LabelMe to COCO conversion."""
    print("Testing LabelMe to COCO conversion...")

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
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {}
                },
                {
                    "label": "dog",
                    "points": [[50, 50], [80, 80]],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {}
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

        # Convert to COCO
        coco_path = str(tmpdir / "coco.json")
        try:
            convert.labelme_to_coco(labelme_path, coco_path)
            print(f"  ✓ LabelMe to COCO conversion successful")

            # Verify COCO file exists and has content
            with open(coco_path, 'r') as f:
                coco_data = json.load(f)
                assert 'annotations' in coco_data
                assert len(coco_data['annotations']) == 2
                print(f"  ✓ COCO file has {len(coco_data['annotations'])} annotations")
        except Exception as e:
            print(f"  ✗ LabelMe to COCO conversion failed: {e}")
            return False

    return True


def test_coco_to_yolo():
    """Test COCO to YOLO conversion."""
    print("Testing COCO to YOLO conversion...")

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

        # Convert to YOLO
        yolo_path = str(tmpdir / "yolo.txt")
        try:
            convert.coco_to_yolo(coco_path, image_path, yolo_path, ["cat", "dog"])
            print(f"  ✓ COCO to YOLO conversion successful")

            # Verify YOLO file exists and has content
            with open(yolo_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 2
                print(f"  ✓ YOLO file has {len(lines)} annotations")
        except Exception as e:
            print(f"  ✗ COCO to YOLO conversion failed: {e}")
            return False

    return True


def test_yolo_to_coco():
    """Test YOLO to COCO conversion."""
    print("Testing YOLO to COCO conversion...")

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

        # Convert to COCO
        coco_path = str(tmpdir / "coco.json")
        try:
            convert.yolo_to_coco(yolo_path, image_path, ["cat", "dog"], coco_path)
            print(f"  ✓ YOLO to COCO conversion successful")

            # Verify COCO file exists
            with open(coco_path, 'r') as f:
                coco_data = json.load(f)
                assert 'annotations' in coco_data
                assert len(coco_data['annotations']) == 2
                print(f"  ✓ COCO file has {len(coco_data['annotations'])} annotations")
        except Exception as e:
            print(f"  ✗ YOLO to COCO conversion failed: {e}")
            return False

    return True


def test_coco_to_labelme():
    """Test COCO to LabelMe conversion."""
    print("Testing COCO to LabelMe conversion...")

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
                }
            ],
            "categories": [
                {"id": 1, "name": "cat", "supercategory": "cat"}
            ]
        }

        coco_path = str(tmpdir / "coco.json")
        with open(coco_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"  Created test COCO file: {coco_path}")

        # Convert to LabelMe
        labelme_path = str(tmpdir / "labelme.json")
        try:
            convert.coco_to_labelme(coco_path, image_path, labelme_path)
            print(f"  ✓ COCO to LabelMe conversion successful")
        except Exception as e:
            print(f"  ✗ COCO to LabelMe conversion failed: {e}")
            return False

    return True


def test_labelme_to_yolo():
    """Test LabelMe to YOLO conversion."""
    print("Testing LabelMe to YOLO conversion...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create LabelMe annotation (no image needed for this conversion)
        labelme_data = {
            "version": "5.3.1",
            "flags": {},
            "shapes": [
                {
                    "label": "cat",
                    "points": [[10, 10], [30, 30]],
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

        # Convert to YOLO
        yolo_path = str(tmpdir / "yolo.txt")
        try:
            convert.labelme_to_yolo(labelme_path, yolo_path, ["cat"])
            print(f"  ✓ LabelMe to YOLO conversion successful")

            # Verify YOLO file exists
            with open(yolo_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                print(f"  ✓ YOLO file has {len(lines)} annotations")
        except Exception as e:
            print(f"  ✗ LabelMe to YOLO conversion failed: {e}")
            return False

    return True


def test_yolo_to_labelme():
    """Test YOLO to LabelMe conversion."""
    print("Testing YOLO to LabelMe conversion...")

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

        print(f"  Created test YOLO file: {yolo_path}")

        # Convert to LabelMe
        labelme_path = str(tmpdir / "labelme.json")
        try:
            convert.yolo_to_labelme(yolo_path, image_path, ["cat"], labelme_path)
            print(f"  ✓ YOLO to LabelMe conversion successful")
        except Exception as e:
            print(f"  ✗ YOLO to LabelMe conversion failed: {e}")
            return False

    return True


def main():
    """Run all convert tests."""
    print("=" * 60)
    print("DataFlow Convert Module Tests")
    print("=" * 60)

    all_passed = True

    # Test imports
    print("\nTesting imports...")
    try:
        from dataflow import convert
        print("  ✓ Convert module imports successfully")
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        all_passed = False
        return

    # Run tests
    tests = [
        test_labelme_to_coco,
        test_coco_to_yolo,
        test_yolo_to_coco,
        test_coco_to_labelme,
        test_labelme_to_yolo,
        test_yolo_to_labelme
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
        print("All convert tests PASSED! 🎉")
    else:
        print("Some convert tests FAILED! ❌")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)