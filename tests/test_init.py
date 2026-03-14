# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/14
@File    : test_init.py
@Author  : DataFlow Team
@Description: Tests for dataflow package public API
"""

import os
import sys
import tempfile
import unittest
import json
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import dataflow
from dataflow.config import Config


class TestInit(unittest.TestCase):
    """Test cases for dataflow package public API."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp(prefix="test_init_")

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_version(self):
        """Test package version."""
        self.assertIsInstance(dataflow.__version__, str)
        self.assertIsInstance(dataflow.__description__, str)

    def test_imports(self):
        """Test that main functions are importable from package."""
        # Check convenience functions
        self.assertTrue(hasattr(dataflow, 'coco_to_yolo'))
        self.assertTrue(hasattr(dataflow, 'yolo_to_coco'))
        self.assertTrue(hasattr(dataflow, 'coco_to_labelme'))
        self.assertTrue(hasattr(dataflow, 'labelme_to_coco'))
        self.assertTrue(hasattr(dataflow, 'labelme_to_yolo'))
        self.assertTrue(hasattr(dataflow, 'yolo_to_labelme'))
        self.assertTrue(hasattr(dataflow, 'visualize_yolo'))
        self.assertTrue(hasattr(dataflow, 'visualize_coco'))
        self.assertTrue(hasattr(dataflow, 'visualize_labelme'))

    def test_coco_to_yolo_function(self):
        """Test coco_to_yolo convenience function."""
        # Create minimal COCO JSON
        coco_json = os.path.join(self.test_dir, "test_coco.json")
        coco_data = {
            "info": {"description": "Test"},
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [100, 150, 200, 120], "area": 24000, "iscrowd": 0}
            ],
            "categories": [
                {"id": 1, "name": "person", "supercategory": "human"}
            ]
        }
        with open(coco_json, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)

        output_dir = os.path.join(self.test_dir, "output")
        result = dataflow.coco_to_yolo(coco_json, output_dir)
        self.assertIsInstance(result, dict)
        self.assertIn('images_processed', result)
        self.assertIn('annotations_processed', result)
        self.assertEqual(result['images_processed'], 1)

    def test_yolo_to_coco_function(self):
        """Test yolo_to_coco convenience function."""
        # Create minimal YOLO data
        images_dir = os.path.join(self.test_dir, "images")
        labels_dir = os.path.join(self.test_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Create dummy image file
        img_path = os.path.join(images_dir, "test.jpg")
        with open(img_path, 'wb') as f:
            f.write(b"")  # Empty file

        # Create label file
        label_path = os.path.join(labels_dir, "test.txt")
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write("0 0.3 0.4 0.2 0.3\n")

        # Create classes file
        classes_file = os.path.join(self.test_dir, "classes.names")
        with open(classes_file, 'w', encoding='utf-8') as f:
            f.write("person\n")

        output_json = os.path.join(self.test_dir, "output.json")
        result = dataflow.yolo_to_coco(images_dir, labels_dir, classes_file, output_json)
        self.assertIsInstance(result, dict)
        self.assertIn('images_processed', result)
        self.assertIn('annotations_processed', result)
        self.assertEqual(result['images_processed'], 1)

    def test_visualize_yolo_function(self):
        """Test visualize_yolo convenience function."""
        # Create minimal YOLO data
        images_dir = os.path.join(self.test_dir, "images")
        labels_dir = os.path.join(self.test_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Create dummy image file
        img_path = os.path.join(images_dir, "test.jpg")
        with open(img_path, 'wb') as f:
            f.write(b"")  # Empty file

        # Create label file
        label_path = os.path.join(labels_dir, "test.txt")
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write("0 0.3 0.4 0.2 0.3\n")

        # Create classes file
        classes_file = os.path.join(self.test_dir, "classes.names")
        with open(classes_file, 'w', encoding='utf-8') as f:
            f.write("person\n")

        result = dataflow.visualize_yolo(images_dir, labels_dir, classes_file)
        self.assertIsInstance(result, dict)
        self.assertIn('images_processed', result)

    def test_config_import(self):
        """Test that Config is accessible from dataflow.config."""
        from dataflow.config import Config
        self.assertTrue(hasattr(Config, 'VERBOSE'))
        self.assertTrue(hasattr(Config, 'OVERWRITE_EXISTING'))
        self.assertTrue(hasattr(Config, 'YOLO_CLASSES_FILENAME'))

    def test_module_imports(self):
        """Test that submodules are importable."""
        # Convert module
        from dataflow.convert import (
            CocoToYoloConverter,
            YoloToCocoConverter,
            CocoToLabelMeConverter,
            LabelMeToCocoConverter,
            LabelMeToYoloConverter,
            YoloToLabelMeConverter
        )
        # Visualize module
        from dataflow.visualize import (
            YoloVisualizer,
            CocoVisualizer,
            LabelMeVisualizer
        )
        # Label module
        from dataflow.label import (
            YoloHandler,
            CocoHandler,
            LabelMeHandler
        )
        self.assertTrue(True)  # If imports succeed, test passes


if __name__ == "__main__":
    unittest.main()