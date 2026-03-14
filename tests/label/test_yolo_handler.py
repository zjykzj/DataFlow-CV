# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/14 22:59
@File    : test_yolo_handler.py
@Author  : zj
@Description: Tests for YoloHandler
"""

import os
import tempfile
import unittest
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import PIL for creating test images
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from dataflow.label import YoloHandler


class TestYoloHandler(unittest.TestCase):
    """Test cases for YoloHandler."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp(prefix="test_yolo_handler_")
        self.classes = ["person", "car", "bicycle"]
        self.classes_path = os.path.join(self.test_dir, "classes.names")
        self._write_classes_file()

        # Create sample image dimensions
        self.image_width = 640
        self.image_height = 480

        # Initialize handler
        self.handler = YoloHandler(verbose=False)

    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _write_classes_file(self):
        """Write classes file."""
        with open(self.classes_path, 'w', encoding='utf-8') as f:
            for cls in self.classes:
                f.write(f"{cls}\n")

    def _create_sample_label_file(self, label_path, format_type="detection"):
        """Create a sample YOLO label file.

        Args:
            label_path: Path to label file
            format_type: "detection" or "segmentation"
        """
        with open(label_path, 'w', encoding='utf-8') as f:
            if format_type == "detection":
                # Detection format: class_id x_center y_center width height
                lines = [
                    "0 0.5 0.5 0.2 0.2",      # person
                    "1 0.3 0.3 0.1 0.1",      # car
                ]
            else:
                # Segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
                # Triangle polygon for person (class 0)
                lines = [
                    "0 0.4 0.4 0.6 0.4 0.5 0.6",
                    "1 0.2 0.2 0.3 0.2 0.3 0.3 0.2 0.3",  # square for car
                ]
            f.write("\n".join(lines))

    def test_read_classes(self):
        """Test reading classes file."""
        classes = self.handler.read_classes(self.classes_path)
        self.assertEqual(classes, self.classes)

        # Test non-existent file
        with self.assertRaises(FileNotFoundError):
            self.handler.read_classes(os.path.join(self.test_dir, "nonexistent.names"))

    def test_write_classes(self):
        """Test writing classes file."""
        new_classes = ["dog", "cat", "bird"]
        new_path = os.path.join(self.test_dir, "new_classes.names")

        result = self.handler.write_classes(new_classes, new_path)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(new_path))

        # Verify content
        with open(new_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        self.assertEqual(lines, new_classes)

        # Test empty classes list
        with self.assertRaises(ValueError):
            self.handler.write_classes([], new_path)

    def test_read_detection_format(self):
        """Test reading detection format label file."""
        label_path = os.path.join(self.test_dir, "labels.txt")
        image_path = os.path.join(self.test_dir, "image.jpg")
        self._create_sample_label_file(label_path, format_type="detection")

        # Create dummy image file (not needed for test, but path must exist for reading)
        self._create_image_file(image_path, self.image_width, self.image_height)

        result = self.handler.read(label_path, image_path, self.classes,
                                   image_size=(self.image_width, self.image_height))

        self.assertEqual(result["image_id"], "image")
        self.assertEqual(result["image_path"], image_path)
        self.assertEqual(result["width"], self.image_width)
        self.assertEqual(result["height"], self.image_height)
        self.assertEqual(len(result["annotations"]), 2)

        # Check first annotation (person)
        ann = result["annotations"][0]
        self.assertEqual(ann["category_id"], 0)
        self.assertEqual(ann["category_name"], "person")
        self.assertEqual(len(ann["bbox"]), 4)
        self.assertIsNone(ann.get("segmentation"))

    def test_read_segmentation_format(self):
        """Test reading segmentation format label file."""
        label_path = os.path.join(self.test_dir, "labels.txt")
        image_path = os.path.join(self.test_dir, "image.jpg")
        self._create_sample_label_file(label_path, format_type="segmentation")

        self._create_image_file(image_path, self.image_width, self.image_height)

        result = self.handler.read(label_path, image_path, self.classes,
                                   image_size=(self.image_width, self.image_height))

        self.assertEqual(len(result["annotations"]), 2)

        # Check segmentation annotation
        ann = result["annotations"][0]
        self.assertEqual(ann["category_id"], 0)
        self.assertEqual(ann["category_name"], "person")
        self.assertIsNotNone(ann.get("segmentation"))
        self.assertEqual(len(ann["segmentation"][0]), 6)  # 3 points -> 6 coordinates

    def test_read_with_require_segmentation(self):
        """Test reading with require_segmentation flag."""
        label_path = os.path.join(self.test_dir, "labels.txt")
        image_path = os.path.join(self.test_dir, "image.jpg")
        self._create_sample_label_file(label_path, format_type="detection")

        self._create_image_file(image_path, self.image_width, self.image_height)

        # With require_segmentation=True, detection annotations should be converted to polygons
        result = self.handler.read(label_path, image_path, self.classes,
                                   image_size=(self.image_width, self.image_height),
                                   require_segmentation=True)

        self.assertEqual(len(result["annotations"]), 2)
        ann = result["annotations"][0]
        self.assertIsNotNone(ann.get("segmentation"))
        self.assertTrue(ann.get("force_polygon", False))

    def test_read_nonexistent_file(self):
        """Test reading non-existent label file."""
        label_path = os.path.join(self.test_dir, "nonexistent.txt")
        image_path = os.path.join(self.test_dir, "image.jpg")

        with self.assertRaises(FileNotFoundError):
            self.handler.read(label_path, image_path, self.classes)

    def test_write_detection_format(self):
        """Test writing detection format label file."""
        # Create annotation data
        image_annotations = {
            "image_id": "test_image",
            "image_path": "/path/to/image.jpg",
            "width": self.image_width,
            "height": self.image_height,
            "annotations": [
                {
                    "category_id": 0,
                    "category_name": "person",
                    "bbox": [100, 150, 50, 80],
                    "segmentation": None
                },
                {
                    "category_id": 1,
                    "category_name": "car",
                    "bbox": [300, 200, 100, 60],
                    "segmentation": None
                }
            ]
        }

        output_path = os.path.join(self.test_dir, "output.txt")
        result = self.handler.write(image_annotations, output_path, self.classes)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_path))

        # Verify file content
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        self.assertEqual(len(lines), 2)
        # First line should be class 0 with normalized coordinates
        parts = lines[0].split()
        self.assertEqual(int(parts[0]), 0)
        self.assertEqual(len(parts), 5)  # class_id + 4 coordinates

    def test_write_segmentation_format(self):
        """Test writing segmentation format label file."""
        image_annotations = {
            "image_id": "test_image",
            "image_path": "/path/to/image.jpg",
            "width": self.image_width,
            "height": self.image_height,
            "annotations": [
                {
                    "category_id": 0,
                    "category_name": "person",
                    "bbox": [100, 150, 50, 80],
                    "segmentation": [[100, 150, 150, 150, 150, 230, 100, 230]]
                }
            ]
        }

        output_path = os.path.join(self.test_dir, "output.txt")
        result = self.handler.write(image_annotations, output_path, self.classes)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_path))

        with open(output_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        self.assertEqual(len(lines), 1)
        parts = lines[0].split()
        self.assertEqual(int(parts[0]), 0)
        # Should have 1 + 8 coordinates (4 points)
        self.assertEqual(len(parts), 9)

    def test_write_invalid_data(self):
        """Test writing invalid annotation data."""
        # Missing required field
        image_annotations = {
            "image_id": "test",
            # Missing annotations
        }

        output_path = os.path.join(self.test_dir, "output.txt")
        with self.assertRaises(ValueError):
            self.handler.write(image_annotations, output_path, self.classes)

        # Invalid class name
        image_annotations = {
            "image_id": "test",
            "annotations": [
                {
                    "category_name": "unknown_class",
                    "bbox": [100, 150, 50, 80]
                }
            ]
        }

        # Should not raise, but skip annotation
        result = self.handler.write(image_annotations, output_path, self.classes)
        self.assertTrue(result)  # File created, but empty

        # Check if file is empty or has no lines
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            self.assertEqual(len(lines), 0)

    def test_read_batch(self):
        """Test batch reading of label files."""
        # Create test directories
        labels_dir = os.path.join(self.test_dir, "labels")
        images_dir = os.path.join(self.test_dir, "images")
        os.makedirs(labels_dir)
        os.makedirs(images_dir)

        # Create sample label files
        for i in range(3):
            label_path = os.path.join(labels_dir, f"image{i}.txt")
            image_path = os.path.join(images_dir, f"image{i}.jpg")

            self._create_sample_label_file(label_path, format_type="detection")
            self._create_image_file(image_path, self.image_width, self.image_height)

        results = self.handler.read_batch(labels_dir, images_dir, self.classes_path)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(len(result["annotations"]), 2)

    def test_write_batch(self):
        """Test batch writing of label files."""
        # Create annotation data list
        images_annotations = []
        for i in range(3):
            images_annotations.append({
                "image_id": f"image{i}",
                "image_path": f"/path/to/image{i}.jpg",
                "width": self.image_width,
                "height": self.image_height,
                "annotations": [
                    {
                        "category_id": i % len(self.classes),
                        "category_name": self.classes[i % len(self.classes)],
                        "bbox": [100 + i*50, 150 + i*30, 50, 80]
                    }
                ]
            })

        output_dir = os.path.join(self.test_dir, "output_labels")
        result = self.handler.write_batch(images_annotations, output_dir, self.classes_path)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_dir))

        # Check that files were created
        for i in range(3):
            expected_file = os.path.join(output_dir, f"image{i}.txt")
            self.assertTrue(os.path.exists(expected_file))

    def test_handler_verbose_mode(self):
        """Test handler initialization with verbose mode."""
        handler = YoloHandler(verbose=True)
        self.assertTrue(handler.verbose)

        handler = YoloHandler(verbose=False)
        self.assertFalse(handler.verbose)

    def _create_image_file(self, image_path, width=640, height=480):
        """Create a valid image file for testing."""
        if HAS_PIL:
            # Create a simple RGB image
            img = Image.new('RGB', (width, height), color='white')
            img.save(image_path, 'JPEG')
        else:
            # Fallback: create empty file (will cause image size detection to fail)
            open(image_path, 'w').close()


if __name__ == "__main__":
    unittest.main()