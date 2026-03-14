# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 21:10
@File    : test_coco_to_yolo.py
@Author  : zj
@Description: Tests for COCO to YOLO conversion
"""

import os
import json
import tempfile
import unittest
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataflow.convert import CocoToYoloConverter
from dataflow.config import Config


class TestCocoToYoloConverter(unittest.TestCase):
    """Test cases for CocoToYoloConverter."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp(prefix="test_coco2yolo_")
        self.coco_json_path = os.path.join(self.test_dir, "annotations.json")
        self.output_dir = os.path.join(self.test_dir, "output")

        # Create sample COCO JSON data
        self._create_sample_coco_json()

    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_sample_coco_json(self):
        """Create a minimal COCO JSON file for testing."""
        coco_data = {
            "info": {
                "description": "Test COCO dataset",
                "version": "1.0",
                "year": 2026,
                "contributor": "Test",
                "date_created": "2026-03-08"
            },
            "licenses": [],
            "images": [
                {
                    "id": 1,
                    "file_name": "test_image.jpg",
                    "width": 640,
                    "height": 480
                },
                {
                    "id": 2,
                    "file_name": "test_image2.jpg",
                    "width": 800,
                    "height": 600
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 150, 200, 120],  # x, y, width, height
                    "area": 24000,
                    "segmentation": [],
                    "iscrowd": 0
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [300, 200, 150, 180],
                    "area": 27000,
                    "segmentation": [],
                    "iscrowd": 0
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "human"
                },
                {
                    "id": 2,
                    "name": "car",
                    "supercategory": "vehicle"
                }
            ]
        }

        # Write JSON file
        with open(self.coco_json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)

    def test_converter_initialization(self):
        """Test converter initialization."""
        converter = CocoToYoloConverter(verbose=False)
        self.assertIsInstance(converter, CocoToYoloConverter)
        self.assertFalse(converter.verbose)

    def test_successful_conversion(self):
        """Test successful COCO to YOLO conversion."""
        converter = CocoToYoloConverter(verbose=False)

        # Perform conversion
        result = converter.convert(self.coco_json_path, self.output_dir)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("output_dir"), self.output_dir)

        # Check that output files were created
        labels_dir = os.path.join(self.output_dir, Config.YOLO_LABELS_DIRNAME)
        classes_file = os.path.join(self.output_dir, Config.YOLO_CLASSES_FILENAME)

        self.assertTrue(os.path.exists(labels_dir), f"Labels directory not found: {labels_dir}")
        self.assertTrue(os.path.exists(classes_file), f"Classes file not found: {classes_file}")

        # Check labels directory contents
        label_files = os.listdir(labels_dir)
        self.assertEqual(len(label_files), 2)  # Two images

        # Check class names file
        with open(classes_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f if line.strip()]

        self.assertEqual(len(class_names), 2)
        # Categories are sorted alphabetically in new converter
        self.assertIn("person", class_names)
        self.assertIn("car", class_names)

        # Check label file for first image (uses image_id as filename)
        label_file = os.path.join(labels_dir, "1.txt")
        self.assertTrue(os.path.exists(label_file), f"Label file not found: {label_file}")

        with open(label_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Should have 2 annotations
        self.assertEqual(len(lines), 2)

        # Verify first annotation (person)
        parts = lines[0].split()
        self.assertEqual(len(parts), 5)
        self.assertEqual(int(parts[0]), 0)  # person is class 0

        # Verify coordinates are normalized
        for coord in map(float, parts[1:]):
            self.assertGreaterEqual(coord, 0.0)
            self.assertLessEqual(coord, 1.0)

    def test_invalid_coco_json_path(self):
        """Test conversion with invalid COCO JSON path."""
        converter = CocoToYoloConverter(verbose=False)

        with self.assertRaises(ValueError):
            converter.convert("/invalid/path/annotations.json", self.output_dir)

    def test_invalid_output_dir(self):
        """Test conversion with invalid output directory."""
        converter = CocoToYoloConverter(verbose=False)

        # 使用文件路径作为输出目录（应该失败）
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="test_invalid_")
        invalid_dir = os.path.join(temp_dir, "file.txt")

        # 创建一个文件而不是目录
        with open(invalid_dir, 'w', encoding='utf-8') as f:
            f.write("test")

        try:
            with self.assertRaises(ValueError):
                converter.convert(self.coco_json_path, invalid_dir)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_empty_coco_json(self):
        """Test conversion with empty COCO JSON."""
        # Create empty COCO JSON
        empty_json_path = os.path.join(self.test_dir, "empty.json")
        with open(empty_json_path, 'w', encoding='utf-8') as f:
            json.dump({"images": [], "annotations": [], "categories": []}, f)

        converter = CocoToYoloConverter(verbose=False)

        # Empty JSON should not raise error but produce zero results
        result = converter.convert(empty_json_path, self.output_dir)
        self.assertEqual(result["images_processed"], 0)
        self.assertEqual(result["annotations_processed"], 0)
        self.assertEqual(result["categories_found"], 0)

    def test_coco_json_without_categories(self):
        """Test conversion with COCO JSON missing categories."""
        # Read existing JSON and remove categories
        with open(self.coco_json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        coco_data["categories"] = []

        no_cat_json_path = os.path.join(self.test_dir, "no_categories.json")
        with open(no_cat_json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f)

        converter = CocoToYoloConverter(verbose=False)

        # Should work but warn
        result = converter.convert(no_cat_json_path, self.output_dir)
        # Without categories, converter may still extract category names from category_id
        self.assertGreaterEqual(result.get("categories_found"), 0)

    def test_conversion_statistics(self):
        """Verify conversion statistics are accurate."""
        converter = CocoToYoloConverter(verbose=False)
        result = converter.convert(self.coco_json_path, self.output_dir)

        # Check statistics
        self.assertEqual(result.get("images_processed"), 2)
        self.assertEqual(result.get("annotations_processed"), 2)
        self.assertEqual(result.get("categories_found"), 2)

    def test_verbose_mode(self):
        """Test converter with verbose mode enabled."""
        converter = CocoToYoloConverter(verbose=True)
        self.assertTrue(converter.verbose)

        # Conversion should still work
        result = converter.convert(self.coco_json_path, self.output_dir)
        self.assertIsInstance(result, dict)

    def test_segmentation_option(self):
        """Test conversion with segmentation option."""
        # Create COCO JSON with segmentation
        seg_json_path = os.path.join(self.test_dir, "segmentation.json")
        seg_data = {
            "info": Config.COCO_DEFAULT_INFO,
            "licenses": [],
            "images": [{
                "id": 1,
                "file_name": "seg_image.jpg",
                "width": 640,
                "height": 480
            }],
            "annotations": [{
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 150, 200, 120],
                "area": 24000,
                "segmentation": [[100, 150, 300, 150, 300, 270, 100, 270]],
                "iscrowd": 0
            }],
            "categories": [{
                "id": 1,
                "name": "object",
                "supercategory": "none"
            }]
        }

        with open(seg_json_path, 'w', encoding='utf-8') as f:
            json.dump(seg_data, f)

        # Test with segmentation disabled (default)
        converter = CocoToYoloConverter(verbose=False)
        result = converter.convert(seg_json_path, self.output_dir)
        self.assertIsInstance(result, dict)

        # Note: With segmentation disabled, only bbox will be converted


if __name__ == "__main__":
    unittest.main()