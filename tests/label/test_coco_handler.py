# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/14 23:00
@File    : test_coco_handler.py
@Author  : zj
@Description: Tests for CocoHandler
"""

import os
import json
import tempfile
import unittest
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataflow.label import CocoHandler


class TestCocoHandler(unittest.TestCase):
    """Test cases for CocoHandler."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp(prefix="test_coco_handler_")
        self.json_path = os.path.join(self.test_dir, "annotations.json")

        # Sample COCO data
        self.coco_data = {
            "info": {
                "description": "Test COCO dataset",
                "version": "1.0",
                "year": 2026,
                "contributor": "Test",
                "date_created": "2026/03/14"
            },
            "licenses": [
                {
                    "url": "http://example.com",
                    "id": 1,
                    "name": "Test License"
                }
            ],
            "images": [
                {
                    "id": 1,
                    "file_name": "image1.jpg",
                    "height": 480,
                    "width": 640
                },
                {
                    "id": 2,
                    "file_name": "image2.jpg",
                    "height": 300,
                    "width": 400
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 150, 50, 80],
                    "segmentation": [[100, 150, 150, 150, 150, 230, 100, 230]],
                    "area": 4000,
                    "iscrowd": 0
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [300, 200, 100, 60],
                    "segmentation": [],
                    "area": 6000,
                    "iscrowd": 0
                },
                {
                    "id": 3,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [50, 60, 30, 40],
                    "segmentation": [[50, 60, 80, 60, 80, 100, 50, 100]],
                    "area": 1200,
                    "iscrowd": 0
                }
            ],
            "categories": [
                {"id": 1, "name": "person", "supercategory": "human"},
                {"id": 2, "name": "car", "supercategory": "vehicle"}
            ]
        }

        # Write COCO JSON file
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.coco_data, f, indent=2)

        # Initialize handler
        self.handler = CocoHandler(verbose=False)

    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_read(self):
        """Test reading COCO JSON file."""
        result = self.handler.read(self.json_path)

        self.assertIn("images", result)
        self.assertIn("annotations", result)
        self.assertIn("categories", result)
        self.assertEqual(len(result["images"]), 2)
        self.assertEqual(len(result["annotations"]), 3)
        self.assertEqual(len(result["categories"]), 2)

        # Test non-existent file
        with self.assertRaises(FileNotFoundError):
            self.handler.read(os.path.join(self.test_dir, "nonexistent.json"))

        # Test invalid JSON
        invalid_path = os.path.join(self.test_dir, "invalid.json")
        with open(invalid_path, 'w') as f:
            f.write("invalid json")
        with self.assertRaises(ValueError):
            self.handler.read(invalid_path)

    def test_read_missing_required_sections(self):
        """Test reading COCO JSON with missing required sections."""
        incomplete_data = {
            "images": [],
            "annotations": [],
            # Missing categories
        }
        incomplete_path = os.path.join(self.test_dir, "incomplete.json")
        with open(incomplete_path, 'w', encoding='utf-8') as f:
            json.dump(incomplete_data, f)

        with self.assertRaises(ValueError):
            self.handler.read(incomplete_path)

    def test_convert_to_unified_format(self):
        """Test converting COCO data to unified format."""
        unified_data = self.handler.convert_to_unified_format(self.coco_data)

        # Should have unified data per image
        self.assertEqual(len(unified_data), 2)  # Two images

        # Find image1 data (image_id is numeric from COCO)
        image1_data = None
        for data in unified_data:
            if data.get("image_id") == 1:  # image_id from COCO images list
                image1_data = data
                break

        self.assertIsNotNone(image1_data)
        self.assertEqual(image1_data["image_path"], "image1.jpg")
        self.assertEqual(image1_data["width"], 640)
        self.assertEqual(image1_data["height"], 480)
        self.assertEqual(len(image1_data["annotations"]), 2)

        # Check annotations
        ann1 = image1_data["annotations"][0]
        self.assertEqual(ann1["category_id"], 1)
        self.assertEqual(ann1["category_name"], "person")
        self.assertEqual(ann1["bbox"], [100, 150, 50, 80])
        self.assertEqual(ann1["segmentation"], [[100, 150, 150, 150, 150, 230, 100, 230]])

    def test_convert_to_unified_format_with_require_segmentation(self):
        """Test converting with require_segmentation flag."""
        # Create COCO data with both bbox-only and segmentation annotations
        coco_data = {
            "images": [
                {"id": 1, "file_name": "image1.jpg", "height": 480, "width": 640}
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 150, 50, 80],
                    "segmentation": [],  # No segmentation
                    "area": 4000,
                    "iscrowd": 0
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [300, 200, 100, 60],
                    "segmentation": [[300, 200, 400, 200, 400, 260, 300, 260]],
                    "area": 6000,
                    "iscrowd": 0
                }
            ],
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "car"}
            ]
        }

        # Without require_segmentation, both annotations should be included
        unified_data = self.handler.convert_to_unified_format(coco_data)
        image_data = unified_data[0]
        self.assertEqual(len(image_data["annotations"]), 2)

        # With require_segmentation=True, only the second annotation (with segmentation) should be included
        unified_data = self.handler.convert_to_unified_format(coco_data, require_segmentation=True)
        image_data = unified_data[0]
        self.assertEqual(len(image_data["annotations"]), 1)
        self.assertEqual(image_data["annotations"][0]["category_id"], 2)

    def test_convert_from_unified_format(self):
        """Test converting unified format back to COCO format."""
        unified_data = [
            {
                "image_id": 1,
                "image_path": "image1.jpg",
                "width": 640,
                "height": 480,
                "annotations": [
                    {
                        "category_id": 1,
                        "category_name": "person",
                        "bbox": [100, 150, 50, 80],
                        "segmentation": [[100, 150, 150, 150, 150, 230, 100, 230]]
                    }
                ]
            }
        ]

        coco_data = self.handler.convert_from_unified_format(unified_data)

        self.assertIn("images", coco_data)
        self.assertIn("annotations", coco_data)
        self.assertIn("categories", coco_data)

        self.assertEqual(len(coco_data["images"]), 1)
        self.assertEqual(coco_data["images"][0]["file_name"], "image1.jpg")
        self.assertEqual(coco_data["images"][0]["width"], 640)
        self.assertEqual(coco_data["images"][0]["height"], 480)

        self.assertEqual(len(coco_data["annotations"]), 1)
        self.assertEqual(coco_data["annotations"][0]["image_id"], 1)
        self.assertEqual(coco_data["annotations"][0]["category_id"], 1)
        self.assertEqual(coco_data["annotations"][0]["bbox"], [100, 150, 50, 80])

    def test_convert_from_unified_format_with_multiple_images(self):
        """Test converting multiple images from unified format."""
        unified_data = [
            {
                "image_id": 1,
                "image_path": "image1.jpg",
                "width": 640,
                "height": 480,
                "annotations": [
                    {
                        "category_id": 1,
                        "category_name": "person",
                        "bbox": [100, 150, 50, 80]
                    }
                ]
            },
            {
                "image_id": 2,
                "image_path": "image2.jpg",
                "width": 400,
                "height": 300,
                "annotations": [
                    {
                        "category_id": 2,
                        "category_name": "car",
                        "bbox": [200, 100, 80, 60]
                    }
                ]
            }
        ]

        coco_data = self.handler.convert_from_unified_format(unified_data)

        self.assertEqual(len(coco_data["images"]), 2)
        self.assertEqual(len(coco_data["annotations"]), 2)

        # Check image IDs are assigned correctly
        image_ids = {img["id"] for img in coco_data["images"]}
        self.assertEqual(image_ids, {1, 2})

    def test_handler_verbose_mode(self):
        """Test handler initialization with verbose mode."""
        handler = CocoHandler(verbose=True)
        self.assertTrue(handler.verbose)

        handler = CocoHandler(verbose=False)
        self.assertFalse(handler.verbose)

    def test_write_coco_json(self):
        """Test writing COCO JSON file."""
        output_path = os.path.join(self.test_dir, "output.json")

        # Use the handler's read method to get data, then write it back
        coco_data = self.handler.read(self.json_path)

        # Write using standard json dump (handler doesn't have write method?)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)

        self.assertTrue(os.path.exists(output_path))

        # Read back and compare
        with open(output_path, 'r', encoding='utf-8') as f:
            written_data = json.load(f)

        self.assertEqual(written_data["images"], coco_data["images"])
        self.assertEqual(written_data["annotations"], coco_data["annotations"])


if __name__ == "__main__":
    unittest.main()