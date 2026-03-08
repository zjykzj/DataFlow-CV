# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:40
@File    : test_yolo.py
@Author  : zj
@Description: Tests for YOLO visualization
"""

import os
import tempfile
import unittest
import numpy as np
import cv2
from pathlib import Path

from dataflow.visualize.yolo import YoloVisualizer


class TestYoloVisualizer(unittest.TestCase):
    """Test cases for YoloVisualizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = YoloVisualizer(verbose=False)
        self.temp_dir = tempfile.mkdtemp(prefix="test_yolo_vis_")
        self.image_dir = os.path.join(self.temp_dir, "images")
        self.label_dir = os.path.join(self.temp_dir, "labels")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

        # Create a simple class file
        self.class_file = os.path.join(self.temp_dir, "classes.names")
        with open(self.class_file, "w") as f:
            f.write("person\n")
            f.write("car\n")
            f.write("bicycle\n")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test visualizer initialization."""
        vis = YoloVisualizer()
        self.assertTrue(hasattr(vis, 'logger'))
        self.assertTrue(hasattr(vis, 'verbose'))

        vis = YoloVisualizer(verbose=False)
        self.assertFalse(vis.verbose)

        vis = YoloVisualizer(verbose=True)
        self.assertTrue(vis.verbose)

    def test_read_classes_file(self):
        """Test reading class names file."""
        classes = self.visualizer._read_classes_file(self.class_file)
        self.assertEqual(len(classes), 3)
        self.assertEqual(classes, ["person", "car", "bicycle"])

    def test_read_classes_file_invalid(self):
        """Test reading non-existent class file."""
        with self.assertRaises(ValueError):
            self.visualizer._read_classes_file("/non/existent/file.names")

    def test_get_color_for_class(self):
        """Test color generation for classes."""
        # Test with default colors
        color1 = self.visualizer.get_color_for_class(0)
        color2 = self.visualizer.get_color_for_class(1)
        self.assertEqual(len(color1), 3)
        self.assertEqual(len(color2), 3)
        self.assertNotEqual(color1, color2)  # Different classes should have different colors

        # Test with many classes (beyond default colors)
        color10 = self.visualizer.get_color_for_class(10, num_classes=20)
        self.assertEqual(len(color10), 3)

    def test_parse_yolo_annotations(self):
        """Test parsing YOLO annotation file."""
        # Create a test label file
        label_file = os.path.join(self.label_dir, "test.txt")
        with open(label_file, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")   # person at center
            f.write("1 0.3 0.3 0.1 0.1 0.95\n")  # car with confidence

        image_shape = (640, 480, 3)  # height, width, channels
        classes = ["person", "car", "bicycle"]

        annotations = self.visualizer._parse_yolo_annotations(
            label_file, classes, image_shape
        )

        self.assertEqual(len(annotations), 2)

        # Check first annotation
        ann1 = annotations[0]
        self.assertEqual(ann1["class_id"], 0)
        self.assertEqual(ann1["class_name"], "person")
        self.assertIsNone(ann1["confidence"])
        bbox = ann1["bbox"]
        self.assertEqual(len(bbox), 4)
        self.assertTrue(0 <= bbox[0] < bbox[2] <= 480)  # x coordinates within image width
        self.assertTrue(0 <= bbox[1] < bbox[3] <= 640)  # y coordinates within image height

        # Check second annotation (with confidence)
        ann2 = annotations[1]
        self.assertEqual(ann2["class_id"], 1)
        self.assertEqual(ann2["class_name"], "car")
        self.assertAlmostEqual(ann2["confidence"], 0.95)

    def test_parse_yolo_annotations_invalid(self):
        """Test parsing invalid YOLO annotations."""
        # Create invalid label file
        label_file = os.path.join(self.label_dir, "invalid.txt")
        with open(label_file, "w") as f:
            f.write("invalid line\n")
            f.write("0 1.5 0.5 0.2 0.2\n")  # x_center > 1
            f.write("10 0.5 0.5 0.2 0.2\n")  # class_id out of range
            f.write("\n")  # empty line

        image_shape = (640, 480, 3)
        classes = ["person", "car", "bicycle"]

        annotations = self.visualizer._parse_yolo_annotations(
            label_file, classes, image_shape
        )
        self.assertEqual(len(annotations), 0)  # All invalid lines should be skipped

    def test_draw_bounding_box(self):
        """Test drawing bounding box on image."""
        # Create a blank image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        original_image = image.copy()

        # Draw a bounding box
        bbox = (10, 10, 50, 50)  # x1, y1, x2, y2
        color = (255, 0, 0)  # Blue in BGR
        label = "test"

        result = self.visualizer.draw_bounding_box(image, bbox, color, label)

        # Image should be modified
        self.assertFalse(np.array_equal(result, original_image))
        self.assertEqual(result.shape, (100, 100, 3))

    def test_get_matching_label_file(self):
        """Test matching image file to label file."""
        image_file = "/path/to/images/test.jpg"
        label_files = [
            "/path/to/labels/test.txt",
            "/path/to/labels/other.txt",
            "/path/to/labels/another.txt"
        ]

        matched = self.visualizer._get_matching_label_file(image_file, label_files)
        self.assertEqual(matched, "/path/to/labels/test.txt")

        # Test no match
        image_file2 = "/path/to/images/nomatch.jpg"
        matched2 = self.visualizer._get_matching_label_file(image_file2, label_files)
        self.assertIsNone(matched2)

    def test_validate_paths(self):
        """Test path validation."""
        # Valid directory
        self.assertTrue(self.visualizer.validate_input_path(self.temp_dir, is_dir=True))

        # Invalid directory
        self.assertFalse(self.visualizer.validate_input_path("/non/existent/dir", is_dir=True))

        # Create directory with create=True
        new_dir = os.path.join(self.temp_dir, "new_subdir")
        self.assertTrue(self.visualizer.validate_input_path(new_dir, is_dir=True, create=True))
        self.assertTrue(os.path.exists(new_dir))

    def test_save_image(self):
        """Test saving image to file."""
        # Create a test image
        image = np.ones((50, 50, 3), dtype=np.uint8) * 255

        # Save to file
        output_path = os.path.join(self.temp_dir, "test_output.jpg")
        success = self.visualizer.save_image(image, output_path)

        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))

        # Verify the saved image can be read
        saved_image = cv2.imread(output_path)
        self.assertIsNotNone(saved_image)
        self.assertEqual(saved_image.shape, (50, 50, 3))

    def test_visualize_with_missing_data(self):
        """Test visualization with missing images or labels."""
        # Empty directories should raise error
        with self.assertRaises(ValueError):
            self.visualizer.visualize(self.image_dir, self.label_dir, self.class_file)

        # Create an image but no matching label
        test_image_path = os.path.join(self.image_dir, "test.jpg")
        cv2.imwrite(test_image_path, np.ones((100, 100, 3), dtype=np.uint8))

        # Should process but find no annotations
        result = self.visualizer.visualize(
            self.image_dir, self.label_dir, self.class_file, save_dir=self.temp_dir
        )
        self.assertEqual(result["images_processed"], 0)
        self.assertEqual(result["annotations_processed"], 0)


if __name__ == "__main__":
    unittest.main()