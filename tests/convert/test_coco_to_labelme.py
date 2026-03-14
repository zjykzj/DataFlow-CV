# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/10 22:00
@File    : test_coco_to_labelme.py
@Author  : zj
@Description: Tests for COCO to LabelMe conversion
"""

import os
import json
import tempfile
import unittest
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataflow.convert import CocoToLabelMeConverter
from dataflow.config import Config


class TestCocoToLabelMeConverter(unittest.TestCase):
    """Test cases for CocoToLabelMeConverter."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp(prefix="test_coco2labelme_")
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
        converter = CocoToLabelMeConverter(verbose=False)
        self.assertIsInstance(converter, CocoToLabelMeConverter)
        self.assertFalse(converter.verbose)

    def test_successful_conversion(self):
        """Test successful COCO to LabelMe conversion."""
        converter = CocoToLabelMeConverter(verbose=False)

        # Perform conversion
        result = converter.convert(self.coco_json_path, self.output_dir)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("output_dir"), self.output_dir)
        self.assertEqual(result.get("images_processed"), 2)
        self.assertEqual(result.get("annotations_processed"), 2)
        self.assertEqual(result.get("categories_found"), 2)

        # Check that output directory was created
        self.assertTrue(os.path.exists(self.output_dir),
                       f"Output directory not found: {self.output_dir}")

        # Check that LabelMe JSON files were created
        labelme_files = [f for f in os.listdir(self.output_dir)
                        if f.endswith('.json') and not f.endswith('classes.json')]
        self.assertEqual(len(labelme_files), 2)

        # Check first LabelMe file
        self.assertTrue(len(labelme_files) > 0, "No LabelMe JSON files created")
        labelme_file = os.path.join(self.output_dir, labelme_files[0])
        self.assertTrue(os.path.exists(labelme_file),
                       f"LabelMe JSON not created: {labelme_file}")

        # Load and verify LabelMe JSON
        with open(labelme_file, 'r', encoding='utf-8') as f:
            labelme_data = json.load(f)

        # Check basic structure
        self.assertIn("version", labelme_data)
        self.assertIn("flags", labelme_data)
        self.assertIn("shapes", labelme_data)
        self.assertIn("imagePath", labelme_data)
        self.assertIn("imageData", labelme_data)
        self.assertIn("imageHeight", labelme_data)
        self.assertIn("imageWidth", labelme_data)

        # Check shapes
        shapes = labelme_data["shapes"]
        self.assertEqual(len(shapes), 2)

        # Verify first shape (person)
        shape1 = shapes[0]
        self.assertEqual(shape1["label"], "person")
        self.assertIn(shape1["shape_type"], ["rectangle", "polygon"])
        self.assertIn("points", shape1)

        # Check class.names file (optional)
        classes_file = os.path.join(self.output_dir, Config.YOLO_CLASSES_FILENAME)
        if os.path.exists(classes_file):
            with open(classes_file, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f if line.strip()]
            self.assertEqual(len(class_names), 2)
            self.assertEqual(class_names[0], "person")
            self.assertEqual(class_names[1], "car")

    def test_invalid_coco_json_path(self):
        """Test conversion with invalid COCO JSON path."""
        converter = CocoToLabelMeConverter(verbose=False)

        with self.assertRaises(ValueError):
            converter.convert("/invalid/path/annotations.json", self.output_dir)

    def test_invalid_output_dir(self):
        """Test conversion with invalid output directory."""
        converter = CocoToLabelMeConverter(verbose=False)

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

        converter = CocoToLabelMeConverter(verbose=False)

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

        converter = CocoToLabelMeConverter(verbose=False)

        # Should work but warn
        result = converter.convert(no_cat_json_path, self.output_dir)
        # Without categories, converter may still extract category names from category_id
        self.assertGreaterEqual(result.get("categories_found"), 0)

    def test_conversion_statistics(self):
        """Verify conversion statistics are accurate."""
        converter = CocoToLabelMeConverter(verbose=False)
        result = converter.convert(self.coco_json_path, self.output_dir)

        # Check statistics
        self.assertEqual(result.get("images_processed"), 2)
        self.assertEqual(result.get("annotations_processed"), 2)
        self.assertEqual(result.get("categories_found"), 2)

    def test_verbose_mode(self):
        """Test converter with verbose mode enabled."""
        converter = CocoToLabelMeConverter(verbose=True)
        self.assertTrue(converter.verbose)

        # Conversion should still work
        result = converter.convert(self.coco_json_path, self.output_dir)
        self.assertIsInstance(result, dict)

    def test_converter_verbose_mode(self):
        """Test converter verbose mode."""
        # Test with verbose=False (default)
        converter = CocoToLabelMeConverter(verbose=False)
        self.assertFalse(converter.verbose)

        # Test with verbose=True
        converter = CocoToLabelMeConverter(verbose=True)
        self.assertTrue(converter.verbose)

        # Test with verbose=None (should use Config.VERBOSE)
        original_verbose = Config.VERBOSE

        Config.VERBOSE = True
        converter = CocoToLabelMeConverter(verbose=None)
        self.assertTrue(converter.verbose)

        Config.VERBOSE = False
        converter = CocoToLabelMeConverter(verbose=None)
        self.assertFalse(converter.verbose)

        # Restore original value
        Config.VERBOSE = original_verbose

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
        converter = CocoToLabelMeConverter(verbose=False)
        result = converter.convert(seg_json_path, self.output_dir)
        self.assertIsInstance(result, dict)

        # Test with segmentation enabled
        result = converter.convert(seg_json_path, self.output_dir, segmentation=True)
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()