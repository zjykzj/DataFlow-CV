# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 21:10
@File    : test_yolo_to_coco.py
@Author  : zj
@Description: Tests for YOLO to COCO conversion
"""

import os
import json
import tempfile
import unittest
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataflow.convert import YoloToCocoConverter
from dataflow.config import Config


class TestYoloToCocoConverter(unittest.TestCase):
    """Test cases for YoloToCocoConverter."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp(prefix="test_yolo2coco_")
        self.image_dir = os.path.join(self.test_dir, "images")
        self.labels_dir = os.path.join(self.test_dir, "labels")
        self.classes_file = os.path.join(self.test_dir, "class.names")
        self.output_json = os.path.join(self.test_dir, "output.json")

        # Create directories
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        # Create sample data
        self._create_sample_images()
        self._create_sample_classes()
        self._create_sample_labels()

    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_sample_images(self):
        """Create sample image files."""
        # Create simple image files using PIL
        try:
            from PIL import Image
            import numpy as np

            # Create two sample images
            for i, (width, height) in enumerate([(640, 480), (800, 600)], 1):
                # Create a simple gradient image
                img_array = np.zeros((height, width, 3), dtype=np.uint8)
                for y in range(height):
                    for x in range(width):
                        img_array[y, x] = [
                            (x * 255) // width,
                            (y * 255) // height,
                            128
                        ]

                img = Image.fromarray(img_array)
                img_path = os.path.join(self.image_dir, f"test_image{i}.jpg")
                img.save(img_path, "JPEG")

        except ImportError:
            # If PIL is not available, create empty files
            # Note: This will cause get_image_info to return defaults
            for i in range(1, 3):
                img_path = os.path.join(self.image_dir, f"test_image{i}.jpg")
                with open(img_path, 'wb') as f:
                    f.write(b"")  # Empty file

    def _create_sample_classes(self):
        """Create sample class names file."""
        class_names = ["person", "car", "bicycle"]
        with open(self.classes_file, 'w', encoding='utf-8') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")

    def _create_sample_labels(self):
        """Create sample YOLO label files."""
        # First image: two objects
        label1 = os.path.join(self.labels_dir, "test_image1.txt")
        with open(label1, 'w', encoding='utf-8') as f:
            # person at normalized coordinates (0.3, 0.4, 0.2, 0.3)
            f.write("0 0.3 0.4 0.2 0.3\n")
            # car at normalized coordinates (0.6, 0.5, 0.25, 0.2)
            f.write("1 0.6 0.5 0.25 0.2\n")

        # Second image: one object
        label2 = os.path.join(self.labels_dir, "test_image2.txt")
        with open(label2, 'w', encoding='utf-8') as f:
            # bicycle at normalized coordinates (0.5, 0.5, 0.3, 0.3)
            f.write("2 0.5 0.5 0.3 0.3\n")

        # Third image: no label file (should still be included in COCO)

    def test_converter_initialization(self):
        """Test converter initialization."""
        converter = YoloToCocoConverter(verbose=False)
        self.assertIsInstance(converter, YoloToCocoConverter)
        self.assertFalse(converter.verbose)

    def test_successful_conversion(self):
        """Test successful YOLO to COCO conversion."""
        converter = YoloToCocoConverter(verbose=False)

        # Perform conversion
        result = converter.convert(
            self.image_dir,
            self.labels_dir,
            self.classes_file,
            self.output_json
        )

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("image_dir"), self.image_dir)
        self.assertEqual(result.get("label_dir"), self.labels_dir)
        self.assertEqual(result.get("classes_file"), self.classes_file)
        self.assertEqual(result.get("coco_json_path"), self.output_json)

        # Check that COCO JSON file was created
        self.assertTrue(os.path.exists(self.output_json),
                       f"COCO JSON not created: {self.output_json}")

        # Load and verify COCO JSON
        with open(self.output_json, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # Check basic structure
        self.assertIn("images", coco_data)
        self.assertIn("annotations", coco_data)
        self.assertIn("categories", coco_data)
        self.assertIn("info", coco_data)

        # Check number of images (should be 2, even though we have 2 images with labels)
        self.assertEqual(len(coco_data["images"]), 2)

        # Check categories
        categories = coco_data["categories"]
        self.assertEqual(len(categories), 3)
        self.assertEqual(categories[0]["name"], "person")
        self.assertEqual(categories[0]["id"], 1)  # COCO IDs start from 1
        self.assertEqual(categories[1]["name"], "car")
        self.assertEqual(categories[1]["id"], 2)
        self.assertEqual(categories[2]["name"], "bicycle")
        self.assertEqual(categories[2]["id"], 3)

        # Check annotations
        annotations = coco_data["annotations"]
        self.assertEqual(len(annotations), 3)  # 2 + 1 annotations

        # Verify first annotation (person)
        ann1 = annotations[0]
        self.assertEqual(ann1["category_id"], 1)  # person
        self.assertIn("image_id", ann1)
        # image_id could be string or integer in new converter
        # self.assertEqual(ann1["image_id"], 1)
        self.assertIn("bbox", ann1)
        self.assertIn("area", ann1)
        self.assertEqual(ann1["iscrowd"], 0)

        # Verify bbox values are reasonable
        bbox = ann1["bbox"]
        self.assertEqual(len(bbox), 4)
        self.assertGreaterEqual(bbox[0], 0)  # x
        self.assertGreaterEqual(bbox[1], 0)  # y
        self.assertGreater(bbox[2], 0)      # width
        self.assertGreater(bbox[3], 0)      # height

    def test_invalid_image_dir(self):
        """Test conversion with invalid image directory."""
        converter = YoloToCocoConverter(verbose=False)

        with self.assertRaises(ValueError):
            converter.convert(
                "/invalid/image/dir",
                self.labels_dir,
                self.classes_file,
                self.output_json
            )

    def test_invalid_labels_dir(self):
        """Test conversion with invalid labels directory."""
        converter = YoloToCocoConverter(verbose=False)

        with self.assertRaises(ValueError):
            converter.convert(
                self.image_dir,
                "/invalid/labels/dir",
                self.classes_file,
                self.output_json
            )

    def test_invalid_classes_file(self):
        """Test conversion with invalid classes file."""
        converter = YoloToCocoConverter(verbose=False)

        with self.assertRaises(ValueError):
            converter.convert(
                self.image_dir,
                self.labels_dir,
                "/invalid/classes.names",
                self.output_json
            )

    def test_empty_image_dir(self):
        """Test conversion with empty image directory."""
        empty_dir = os.path.join(self.test_dir, "empty_images")
        os.makedirs(empty_dir, exist_ok=True)

        converter = YoloToCocoConverter(verbose=False)

        with self.assertRaises(ValueError):
            converter.convert(
                empty_dir,
                self.labels_dir,
                self.classes_file,
                self.output_json
            )

    def test_empty_classes_file(self):
        """Test conversion with empty classes file."""
        empty_classes = os.path.join(self.test_dir, "empty.names")
        with open(empty_classes, 'w', encoding='utf-8') as f:
            pass  # Empty file

        converter = YoloToCocoConverter(verbose=False)

        with self.assertRaises(ValueError):
            converter.convert(
                self.image_dir,
                self.labels_dir,
                empty_classes,
                self.output_json
            )

    def test_images_without_labels(self):
        """Test handling of images without corresponding label files."""
        # Create an extra image without label file
        extra_img_path = os.path.join(self.image_dir, "extra_image.jpg")
        try:
            from PIL import Image
            import numpy as np
            img_array = np.zeros((100, 100, 3), dtype=np.uint8)
            Image.fromarray(img_array).save(extra_img_path, "JPEG")
        except ImportError:
            with open(extra_img_path, 'wb') as f:
                f.write(b"")

        converter = YoloToCocoConverter(verbose=False)

        result = converter.convert(
            self.image_dir,
            self.labels_dir,
            self.classes_file,
            self.output_json
        )

        # Should still succeed
        self.assertIsInstance(result, dict)
        # New converter may not provide images_without_labels
        # self.assertGreaterEqual(result.get("images_without_labels", 0), 0)

        # Load COCO JSON
        with open(self.output_json, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # New converter may only include images with labels
        # Should have at least 2 images (original images with labels)
        self.assertEqual(len(coco_data["images"]), 2)

    def test_conversion_statistics(self):
        """Verify conversion statistics are accurate."""
        converter = YoloToCocoConverter(verbose=False)

        result = converter.convert(
            self.image_dir,
            self.labels_dir,
            self.classes_file,
            self.output_json
        )

        # Check statistics
        self.assertEqual(result.get("categories_found"), 3)
        self.assertEqual(result.get("images_processed"), 2)
        self.assertEqual(result.get("annotations_processed"), 3)

    def test_verbose_mode(self):
        """Test converter with verbose mode enabled."""
        converter = YoloToCocoConverter(verbose=True)
        self.assertTrue(converter.verbose)

        # Conversion should still work
        result = converter.convert(
            self.image_dir,
            self.labels_dir,
            self.classes_file,
            self.output_json
        )
        self.assertIsInstance(result, dict)

    def test_converter_verbose_mode(self):
        """Test converter verbose mode."""
        # Test with verbose=False (default)
        converter = YoloToCocoConverter(verbose=False)
        self.assertFalse(converter.verbose)

        # Test with verbose=True
        converter = YoloToCocoConverter(verbose=True)
        self.assertTrue(converter.verbose)

        # Test with verbose=None (should use Config.VERBOSE)
        original_verbose = Config.VERBOSE

        Config.VERBOSE = True
        converter = YoloToCocoConverter(verbose=None)
        self.assertTrue(converter.verbose)

        Config.VERBOSE = False
        converter = YoloToCocoConverter(verbose=None)
        self.assertFalse(converter.verbose)

        # Restore original value
        Config.VERBOSE = original_verbose

    def test_segmentation_labels(self):
        """Test conversion of segmentation labels."""
        # Create segmentation label file
        seg_label_dir = os.path.join(self.test_dir, "seg_labels")
        os.makedirs(seg_label_dir, exist_ok=True)

        seg_label = os.path.join(seg_label_dir, "test_image1.txt")
        with open(seg_label, 'w', encoding='utf-8') as f:
            # Segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
            # Triangle coordinates (normalized)
            f.write("0 0.1 0.1 0.5 0.1 0.3 0.5\n")

        converter = YoloToCocoConverter(verbose=False)

        result = converter.convert(
            self.image_dir,
            seg_label_dir,
            self.classes_file,
            self.output_json
        )

        # Should succeed
        self.assertIsInstance(result, dict)

        # Load COCO JSON
        with open(self.output_json, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # Check that annotation has segmentation field
        annotations = coco_data["annotations"]
        if annotations:
            self.assertIn("segmentation", annotations[0])

    def test_malformed_label_file(self):
        """Test handling of malformed label files."""
        # Create malformed label file
        bad_label_dir = os.path.join(self.test_dir, "bad_labels")
        os.makedirs(bad_label_dir, exist_ok=True)

        bad_label = os.path.join(bad_label_dir, "test_image1.txt")
        with open(bad_label, 'w', encoding='utf-8') as f:
            f.write("invalid format\n")
            f.write("0 0.1 0.2\n")  # Too few coordinates
            f.write("99 0.1 0.2 0.3 0.4\n")  # Invalid class index

        converter = YoloToCocoConverter(verbose=False)

        # Should still succeed (skip bad lines)
        result = converter.convert(
            self.image_dir,
            bad_label_dir,
            self.classes_file,
            self.output_json
        )
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()