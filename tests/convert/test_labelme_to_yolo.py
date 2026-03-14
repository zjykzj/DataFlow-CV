# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/10 22:00
@File    : test_labelme_to_yolo.py
@Author  : zj
@Description: Tests for LabelMe to YOLO conversion
"""

import os
import json
import tempfile
import unittest
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataflow.convert import LabelMeToYoloConverter
from dataflow.config import Config


class TestLabelMeToYoloConverter(unittest.TestCase):
    """Test cases for LabelMeToYoloConverter."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp(prefix="test_labelme2yolo_")
        self.label_dir = os.path.join(self.test_dir, "labels")
        self.output_dir = os.path.join(self.test_dir, "output")
        self.classes_file = os.path.join(self.test_dir, "classes.names")

        # Create directories
        os.makedirs(self.label_dir, exist_ok=True)

        # Create classes file
        with open(self.classes_file, 'w', encoding='utf-8') as f:
            f.write("person\ncar\n")

        # Create sample LabelMe files
        self._create_sample_labelme_files()

    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_sample_labelme_files(self):
        """Create sample LabelMe JSON files."""
        # First image: two objects
        labelme1 = {
            "version": "5.3.1",
            "flags": {},
            "shapes": [
                {
                    "label": "person",
                    "points": [[100, 150], [300, 270]],  # rectangle: [x1,y1, x2,y2]
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                },
                {
                    "label": "car",
                    "points": [[300, 200], [450, 380]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
            ],
            "imagePath": "test_image1.jpg",
            "imageData": None,
            "imageHeight": 480,
            "imageWidth": 640
        }

        # Second image: one object
        labelme2 = {
            "version": "5.3.1",
            "flags": {},
            "shapes": [
                {
                    "label": "person",  # Same category as first image
                    "points": [[200, 200], [500, 500]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
            ],
            "imagePath": "test_image2.jpg",
            "imageData": None,
            "imageHeight": 600,
            "imageWidth": 800
        }

        # Write LabelMe JSON files
        labelme1_path = os.path.join(self.label_dir, "test_image1.json")
        with open(labelme1_path, 'w', encoding='utf-8') as f:
            json.dump(labelme1, f, indent=2)

        labelme2_path = os.path.join(self.label_dir, "test_image2.json")
        with open(labelme2_path, 'w', encoding='utf-8') as f:
            json.dump(labelme2, f, indent=2)

    def test_converter_initialization(self):
        """Test converter initialization."""
        converter = LabelMeToYoloConverter(verbose=False)
        self.assertIsInstance(converter, LabelMeToYoloConverter)
        self.assertFalse(converter.verbose)

    def test_successful_conversion(self):
        """Test successful LabelMe to YOLO conversion."""
        converter = LabelMeToYoloConverter(verbose=False)

        # Perform conversion
        result = converter.convert(self.label_dir, self.classes_file, self.output_dir)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("label_dir"), self.label_dir)
        self.assertEqual(result.get("classes_file"), self.classes_file)
        self.assertEqual(result.get("output_dir"), self.output_dir)
        self.assertEqual(result.get("images_processed"), 2)
        self.assertEqual(result.get("annotations_processed"), 3)
        self.assertEqual(result.get("categories_found"), 2)  # person, car in classes file
        self.assertEqual(result.get("categories_in_data"), 2)  # person, car in data

        # Check that output files were created (labels in output_dir/labels subdirectory)
        # Classes file should not be created (using provided one)
        self.assertTrue(os.path.exists(self.classes_file), f"Classes file not found: {self.classes_file}")

        # Check labels directory contents (label files should be in output_dir/labels)
        labels_dir = os.path.join(self.output_dir, Config.YOLO_LABELS_DIRNAME)
        self.assertTrue(os.path.exists(labels_dir), f"Labels directory not found: {labels_dir}")
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        self.assertEqual(len(label_files), 2)  # Two images

        # Check class names file (provided one)
        with open(self.classes_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f if line.strip()]

        self.assertEqual(len(class_names), 2)
        self.assertIn("person", class_names)
        self.assertIn("car", class_names)

        # Check label file for first image
        label_file = os.path.join(labels_dir, "test_image1.txt")
        self.assertTrue(os.path.exists(label_file))

        with open(label_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Should have 2 annotations
        self.assertEqual(len(lines), 2)

        # Verify first annotation
        parts = lines[0].split()
        # Could be bbox (5 parts) or segmentation (more parts)
        self.assertGreaterEqual(len(parts), 5)
        # Class index should be consistent with class.names order
        # Let's just verify coordinates are normalized
        for coord in map(float, parts[1:]):
            self.assertGreaterEqual(coord, 0.0)
            self.assertLessEqual(coord, 1.0)

    def test_invalid_label_dir(self):
        """Test conversion with invalid label directory."""
        converter = LabelMeToYoloConverter(verbose=False)

        with self.assertRaises(ValueError):
            converter.convert("/invalid/path/labels", self.classes_file, self.output_dir)

    def test_invalid_output_dir(self):
        """Test conversion with invalid output directory."""
        converter = LabelMeToYoloConverter(verbose=False)

        # 使用文件路径作为输出目录（应该失败）
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="test_invalid_")
        invalid_dir = os.path.join(temp_dir, "file.txt")

        # 创建一个文件而不是目录
        with open(invalid_dir, 'w', encoding='utf-8') as f:
            f.write("test")

        try:
            with self.assertRaises(ValueError):
                converter.convert(self.label_dir, self.classes_file, invalid_dir)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_empty_label_dir(self):
        """Test conversion with empty label directory."""
        empty_dir = os.path.join(self.test_dir, "empty_labels")
        os.makedirs(empty_dir, exist_ok=True)

        converter = LabelMeToYoloConverter(verbose=False)

        # Empty directory should not raise error but produce zero results
        result = converter.convert(empty_dir, self.classes_file, self.output_dir)
        self.assertEqual(result["images_processed"], 0)
        self.assertEqual(result["annotations_processed"], 0)
        self.assertEqual(result["categories_found"], 2)  # person, car from classes file

        # Check that output directory was created (labels subdirectory may exist)
        self.assertTrue(os.path.exists(self.output_dir))
        # Labels directory may exist but should be empty
        labels_dir = os.path.join(self.output_dir, Config.YOLO_LABELS_DIRNAME)
        if os.path.exists(labels_dir):
            # No label files should be created
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            self.assertEqual(len(label_files), 0)

    def test_conversion_statistics(self):
        """Verify conversion statistics are accurate."""
        converter = LabelMeToYoloConverter(verbose=False)
        result = converter.convert(self.label_dir, self.classes_file, self.output_dir)

        # Check statistics
        self.assertEqual(result.get("images_processed"), 2)
        self.assertEqual(result.get("annotations_processed"), 3)
        self.assertEqual(result.get("categories_found"), 2)  # person, car from classes file

    def test_verbose_mode(self):
        """Test converter with verbose mode enabled."""
        converter = LabelMeToYoloConverter(verbose=True)
        self.assertTrue(converter.verbose)

        # Conversion should still work
        result = converter.convert(self.label_dir, self.classes_file, self.output_dir)
        self.assertIsInstance(result, dict)

    def test_segmentation_option(self):
        """Test conversion with segmentation option."""
        # Create LabelMe JSON with polygon (segmentation)
        seg_label_dir = os.path.join(self.test_dir, "seg_labels")
        os.makedirs(seg_label_dir, exist_ok=True)

        labelme_seg = {
            "version": "5.3.1",
            "flags": {},
            "shapes": [
                {
                    "label": "person",
                    "points": [[100, 150], [300, 150], [300, 270], [100, 270]],  # polygon
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
            ],
            "imagePath": "seg_image.jpg",
            "imageData": None,
            "imageHeight": 480,
            "imageWidth": 640
        }

        seg_label_path = os.path.join(seg_label_dir, "seg_image.json")
        with open(seg_label_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_seg, f, indent=2)

        converter = LabelMeToYoloConverter(verbose=False)

        # Test with segmentation disabled (default)
        result = converter.convert(seg_label_dir, self.classes_file, self.output_dir)
        self.assertIsInstance(result, dict)

        # Test with segmentation enabled
        result = converter.convert(seg_label_dir, self.classes_file, self.output_dir, segmentation=True)
        self.assertIsInstance(result, dict)

    def test_labelme_with_polygon_shape(self):
        """Test conversion of LabelMe polygon shapes to YOLO segmentation format."""
        polygon_label_dir = os.path.join(self.test_dir, "polygon_labels")
        os.makedirs(polygon_label_dir, exist_ok=True)

        # Create a classes file that includes "object" category
        polygon_classes_file = os.path.join(self.test_dir, "polygon_classes.names")
        with open(polygon_classes_file, 'w', encoding='utf-8') as f:
            f.write("object\n")

        labelme_poly = {
            "version": "5.3.1",
            "flags": {},
            "shapes": [
                {
                    "label": "object",
                    "points": [[0.1, 0.1], [0.5, 0.1], [0.3, 0.5]],  # triangle
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
            ],
            "imagePath": "poly_image.jpg",
            "imageData": None,
            "imageHeight": 480,
            "imageWidth": 640
        }

        poly_label_path = os.path.join(polygon_label_dir, "poly_image.json")
        with open(poly_label_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_poly, f, indent=2)

        converter = LabelMeToYoloConverter(verbose=False)
        result = converter.convert(polygon_label_dir, polygon_classes_file, self.output_dir)
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()