# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/14 23:01
@File    : test_labelme_handler.py
@Author  : zj
@Description: Tests for LabelMeHandler
"""

import os
import json
import tempfile
import unittest
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataflow.label import LabelMeHandler


class TestLabelMeHandler(unittest.TestCase):
    """Test cases for LabelMeHandler."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp(prefix="test_labelme_handler_")
        self.json_path = os.path.join(self.test_dir, "image1.json")

        # Sample LabelMe data
        self.labelme_data = {
            "version": "5.3.1",
            "flags": {},
            "shapes": [
                {
                    "label": "person",
                    "points": [[100, 150], [150, 150], [150, 230], [100, 230]],
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                },
                {
                    "label": "car",
                    "points": [[300, 200], [400, 260]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
            ],
            "imagePath": "image1.jpg",
            "imageData": None,
            "imageHeight": 480,
            "imageWidth": 640
        }

        # Write LabelMe JSON file
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.labelme_data, f, indent=2)

        # Create dummy image file
        self.image_path = os.path.join(self.test_dir, "image1.jpg")
        open(self.image_path, 'w').close()

        # Initialize handler
        self.handler = LabelMeHandler(verbose=False)

    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_read(self):
        """Test reading LabelMe JSON file."""
        result = self.handler.read(self.json_path)

        self.assertEqual(result["image_id"], "image1")
        self.assertEqual(result["image_path"], self.image_path)
        self.assertEqual(result["width"], 640)
        self.assertEqual(result["height"], 480)
        self.assertEqual(len(result["annotations"]), 2)

        # Check polygon annotation
        polygon_ann = result["annotations"][0]
        self.assertEqual(polygon_ann["category_name"], "person")
        self.assertEqual(polygon_ann["category_id"], 0)  # Auto-assigned ID
        self.assertIsNotNone(polygon_ann.get("segmentation"))
        self.assertEqual(len(polygon_ann["segmentation"][0]), 8)  # 4 points

        # Check rectangle annotation
        rect_ann = result["annotations"][1]
        self.assertEqual(rect_ann["category_name"], "car")
        self.assertEqual(rect_ann["category_id"], 0)
        self.assertIsNotNone(rect_ann.get("bbox"))
        self.assertIsNotNone(rect_ann.get("segmentation"))
        self.assertEqual(len(rect_ann["segmentation"][0]), 8)  # rectangle polygon has 4 points

    def test_read_with_require_segmentation(self):
        """Test reading with require_segmentation flag."""
        # With require_segmentation=True, only polygon shapes should be included
        result = self.handler.read(self.json_path, require_segmentation=True)

        self.assertEqual(len(result["annotations"]), 1)  # Only polygon
        self.assertEqual(result["annotations"][0]["category_name"], "person")

    def test_read_nonexistent_file(self):
        """Test reading non-existent JSON file."""
        with self.assertRaises(FileNotFoundError):
            self.handler.read(os.path.join(self.test_dir, "nonexistent.json"))

    def test_read_invalid_json(self):
        """Test reading invalid JSON file."""
        invalid_path = os.path.join(self.test_dir, "invalid.json")
        with open(invalid_path, 'w') as f:
            f.write("invalid json")
        with self.assertRaises(ValueError):
            self.handler.read(invalid_path)

    def test_read_missing_required_fields(self):
        """Test reading JSON with missing required fields."""
        incomplete_data = {
            "shapes": [],
            # Missing imagePath, imageHeight, imageWidth
        }
        incomplete_path = os.path.join(self.test_dir, "incomplete.json")
        with open(incomplete_path, 'w', encoding='utf-8') as f:
            json.dump(incomplete_data, f)

        with self.assertRaises(ValueError):
            self.handler.read(incomplete_path)

    def test_write(self):
        """Test writing LabelMe JSON file."""
        # Create unified format data
        data = {
            "image_id": "test_image",
            "image_path": os.path.join(self.test_dir, "test_image.jpg"),
            "width": 640,
            "height": 480,
            "annotations": [
                {
                    "category_id": 0,
                    "category_name": "person",
                    "segmentation": [[100, 150, 150, 150, 150, 230, 100, 230]]
                },
                {
                    "category_id": 1,
                    "category_name": "car",
                    "bbox": [300, 200, 100, 60]
                }
            ]
        }

        output_path = os.path.join(self.test_dir, "output.json")
        result = self.handler.write(data, output_path)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_path))

        # Verify written content
        with open(output_path, 'r', encoding='utf-8') as f:
            written_data = json.load(f)

        self.assertEqual(written_data["imagePath"], "test_image.jpg")
        self.assertEqual(written_data["imageWidth"], 640)
        self.assertEqual(written_data["imageHeight"], 480)
        self.assertEqual(len(written_data["shapes"]), 2)

        # Check shape types
        shape_types = {shape["shape_type"] for shape in written_data["shapes"]}
        self.assertEqual(shape_types, {"polygon", "rectangle"})

    def test_write_invalid_data(self):
        """Test writing invalid data."""
        # Missing required fields
        data = {
            "image_id": "test",
            # Missing width, height, annotations
        }

        output_path = os.path.join(self.test_dir, "output.json")
        with self.assertRaises(ValueError):
            self.handler.write(data, output_path)

    def test_read_batch(self):
        """Test batch reading of LabelMe JSON files."""
        # Create multiple JSON files
        json_dir = os.path.join(self.test_dir, "jsons")
        os.makedirs(json_dir)

        for i in range(3):
            data = self.labelme_data.copy()
            data["imagePath"] = f"image{i}.jpg"
            data["shapes"][0]["label"] = f"object{i}"

            json_path = os.path.join(json_dir, f"image{i}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            # Create dummy image file
            open(os.path.join(self.test_dir, f"image{i}.jpg"), 'w').close()

        results = self.handler.read_batch(json_dir)

        self.assertEqual(len(results), 3)
        # Check all expected image_ids are present (order may vary)
        image_ids = {result["image_id"] for result in results}
        self.assertEqual(image_ids, {"image0", "image1", "image2"})
        for result in results:
            self.assertEqual(len(result["annotations"]), 2)

    def test_write_batch(self):
        """Test batch writing of LabelMe JSON files."""
        # Create data list
        data_list = []
        for i in range(3):
            data_list.append({
                "image_id": f"image{i}",
                "image_path": os.path.join(self.test_dir, f"image{i}.jpg"),
                "width": 640,
                "height": 480,
                "annotations": [
                    {
                        "category_id": 0,
                        "category_name": "person",
                        "bbox": [100 + i*50, 150 + i*30, 50, 80]
                    }
                ]
            })

        output_dir = os.path.join(self.test_dir, "output_jsons")
        result = self.handler.write_batch(data_list, output_dir)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_dir))

        # Check that files were created
        for i in range(3):
            expected_file = os.path.join(output_dir, f"image{i}.json")
            self.assertTrue(os.path.exists(expected_file))

    def test_handler_verbose_mode(self):
        """Test handler initialization with verbose mode."""
        handler = LabelMeHandler(verbose=True)
        self.assertTrue(handler.verbose)

        handler = LabelMeHandler(verbose=False)
        self.assertFalse(handler.verbose)

    def test_shape_parsing_edge_cases(self):
        """Test edge cases in shape parsing."""
        # Create a LabelMe JSON with empty shapes
        data = self.labelme_data.copy()
        data["shapes"] = []

        empty_path = os.path.join(self.test_dir, "empty_shapes.json")
        with open(empty_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        result = self.handler.read(empty_path)
        self.assertEqual(len(result["annotations"]), 0)

        # Test shape with invalid points
        data["shapes"] = [
            {
                "label": "test",
                "points": [],  # Empty points
                "shape_type": "polygon"
            }
        ]

        invalid_path = os.path.join(self.test_dir, "invalid_shapes.json")
        with open(invalid_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        result = self.handler.read(invalid_path)
        self.assertEqual(len(result["annotations"]), 0)


if __name__ == "__main__":
    unittest.main()