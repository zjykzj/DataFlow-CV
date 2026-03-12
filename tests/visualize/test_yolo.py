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

    def test_initialization_with_segmentation(self):
        """Test visualizer initialization with segmentation parameter."""
        vis = YoloVisualizer(segmentation=True)
        self.assertTrue(vis.segmentation)
        self.assertTrue(hasattr(vis, 'label_handler'))

        vis2 = YoloVisualizer(segmentation=False)
        self.assertFalse(vis2.segmentation)

    def test_basic_visualization_detection(self):
        """Test basic YOLO visualization with detection format."""
        # Create a test image
        image_path = os.path.join(self.image_dir, "test.jpg")
        image = np.ones((640, 480, 3), dtype=np.uint8) * 255
        cv2.imwrite(image_path, image)

        # Create a YOLO label file with detection format (bbox)
        label_path = os.path.join(self.label_dir, "test.txt")
        with open(label_path, "w") as f:
            # class_id x_center y_center width height
            f.write("0 0.5 0.5 0.2 0.2\n")   # person at center
            f.write("1 0.3 0.3 0.1 0.1\n")   # car

        # Visualize without saving (display mode)
        result = self.visualizer.visualize(
            self.image_dir, self.label_dir, self.class_file
        )

        # Check results
        self.assertIn("images_processed", result)
        self.assertIn("annotations_processed", result)
        self.assertIn("classes_found", result)
        self.assertEqual(result["images_processed"], 1)
        self.assertEqual(result["annotations_processed"], 2)
        self.assertEqual(sorted(result["classes_found"]), ["car", "person"])

    def test_basic_visualization_segmentation(self):
        """Test basic YOLO visualization with segmentation format."""
        # Create a test image
        image_path = os.path.join(self.image_dir, "test.jpg")
        image = np.ones((640, 480, 3), dtype=np.uint8) * 255
        cv2.imwrite(image_path, image)

        # Create a YOLO label file with segmentation format (polygon)
        label_path = os.path.join(self.label_dir, "test.txt")
        with open(label_path, "w") as f:
            # class_id x1 y1 x2 y2 x3 y3 (triangle)
            f.write("0 0.3 0.3 0.5 0.3 0.4 0.5\n")   # person polygon
            f.write("1 0.6 0.6 0.8 0.6 0.7 0.8\n")   # car polygon

        # Visualize without segmentation flag (auto detection)
        result = self.visualizer.visualize(
            self.image_dir, self.label_dir, self.class_file
        )

        # Check results
        self.assertEqual(result["images_processed"], 1)
        self.assertEqual(result["annotations_processed"], 2)

    def test_visualization_with_segmentation_flag(self):
        """Test visualization with segmentation flag enabled."""
        # Create a test image
        image_path = os.path.join(self.image_dir, "test.jpg")
        image = np.ones((640, 480, 3), dtype=np.uint8) * 255
        cv2.imwrite(image_path, image)

        # Create a YOLO label file with segmentation format only
        label_path = os.path.join(self.label_dir, "test.txt")
        with open(label_path, "w") as f:
            # class_id x1 y1 x2 y2 x3 y3 (triangle)
            f.write("0 0.3 0.3 0.5 0.3 0.4 0.5\n")   # person polygon

        # Create visualizer with segmentation flag
        seg_visualizer = YoloVisualizer(verbose=False, segmentation=True)

        # Should work with segmentation data
        result = seg_visualizer.visualize(
            self.image_dir, self.label_dir, self.class_file
        )
        self.assertEqual(result["images_processed"], 1)
        self.assertEqual(result["annotations_processed"], 1)

    def test_visualization_with_segmentation_flag_strict_fail(self):
        """Test strict segmentation mode failure when detection format is present."""
        # Create a test image
        image_path = os.path.join(self.image_dir, "test.jpg")
        image = np.ones((640, 480, 3), dtype=np.uint8) * 255
        cv2.imwrite(image_path, image)

        # Create a YOLO label file with detection format (bbox) only
        label_path = os.path.join(self.label_dir, "test.txt")
        with open(label_path, "w") as f:
            # class_id x_center y_center width height (detection format)
            f.write("0 0.5 0.5 0.2 0.2\n")   # person bbox

        # Create visualizer with segmentation flag (strict mode)
        seg_visualizer = YoloVisualizer(verbose=False, segmentation=True)

        # With the updated YOLO handler, detection annotations are converted to polygons
        # when segmentation=True, so visualization should succeed
        result = seg_visualizer.visualize(
            self.image_dir, self.label_dir, self.class_file
        )
        self.assertEqual(result["images_processed"], 1)
        self.assertEqual(result["annotations_processed"], 1)

    def test_visualization_with_save_dir(self):
        """Test visualization with save directory."""
        # Create a test image
        image_path = os.path.join(self.image_dir, "test.jpg")
        image = np.ones((640, 480, 3), dtype=np.uint8) * 255
        cv2.imwrite(image_path, image)

        # Create a YOLO label file
        label_path = os.path.join(self.label_dir, "test.txt")
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

        save_dir = os.path.join(self.temp_dir, "output")

        result = self.visualizer.visualize(
            self.image_dir, self.label_dir, self.class_file, save_dir=save_dir
        )

        self.assertEqual(result["images_processed"], 1)
        self.assertEqual(result["saved_images"], 1)
        self.assertTrue(os.path.exists(save_dir))

        # Check saved image exists
        saved_image_path = os.path.join(save_dir, "test.jpg")
        self.assertTrue(os.path.exists(saved_image_path))

    def test_visualization_mixed_format_auto_detection(self):
        """Test auto detection with mixed detection and segmentation formats."""
        # Create a test image
        image_path = os.path.join(self.image_dir, "test.jpg")
        image = np.ones((640, 480, 3), dtype=np.uint8) * 255
        cv2.imwrite(image_path, image)

        # Create a YOLO label file with mixed formats
        label_path = os.path.join(self.label_dir, "test.txt")
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")  # detection format
            f.write("1 0.3 0.3 0.5 0.3 0.4 0.5\n")  # segmentation format

        # Visualize without segmentation flag (auto detection)
        result = self.visualizer.visualize(
            self.image_dir, self.label_dir, self.class_file
        )

        # Both annotations should be processed
        self.assertEqual(result["annotations_processed"], 2)
        self.assertEqual(sorted(result["classes_found"]), ["car", "person"])

    def test_batch_visualize(self):
        """Test batch visualization method."""
        # Create test data for multiple datasets
        image_dirs = [self.image_dir, self.image_dir]
        label_dirs = [self.label_dir, self.label_dir]
        class_paths = [self.class_file, self.class_file]

        # Create test image and label
        image_path = os.path.join(self.image_dir, "test.jpg")
        image = np.ones((640, 480, 3), dtype=np.uint8) * 255
        cv2.imwrite(image_path, image)

        label_path = os.path.join(self.label_dir, "test.txt")
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

        results = self.visualizer.batch_visualize(
            image_dirs, label_dirs, class_paths
        )

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsNotNone(result)
            self.assertEqual(result["images_processed"], 1)


if __name__ == "__main__":
    unittest.main()