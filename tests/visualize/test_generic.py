# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/9 22:35
@File    : test_generic.py
@Author  : zj
@Description: Tests for GenericVisualizer base class
"""

import os
import tempfile
import unittest
import numpy as np
import cv2
from typing import List, Dict, Any, Optional

from dataflow.visualize.generic import GenericVisualizer


class TestGenericVisualizer(GenericVisualizer):
    """Test implementation of GenericVisualizer for unit testing."""

    def __init__(self, verbose: bool = None, segmentation: bool = False):
        super().__init__(verbose, segmentation)
        # Mock label handler
        self.label_handler = None

    def visualize(self, *args, **kwargs) -> Dict[str, Any]:
        """Mock implementation for testing."""
        return {"mock": True}

    def _load_annotations(self, label_source, image_dir):
        """Mock implementation for testing."""
        return []


class TestGenericVisualizerClass(unittest.TestCase):
    """Test cases for GenericVisualizer base class."""

    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = TestGenericVisualizer(verbose=False)
        self.temp_dir = tempfile.mkdtemp(prefix="test_generic_vis_")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test visualizer initialization."""
        vis = TestGenericVisualizer()
        self.assertTrue(hasattr(vis, 'logger'))
        self.assertTrue(hasattr(vis, 'verbose'))
        self.assertTrue(hasattr(vis, 'segmentation'))
        self.assertFalse(vis.segmentation)  # Default is False
        self.assertIsNone(vis.label_handler)

    def test_initialization_with_segmentation(self):
        """Test visualizer initialization with segmentation parameter."""
        vis = TestGenericVisualizer(segmentation=True)
        self.assertTrue(vis.segmentation)

        vis2 = TestGenericVisualizer(segmentation=False)
        self.assertFalse(vis2.segmentation)

    def test_validate_segmentation_format_valid(self):
        """Test segmentation format validation with valid data."""
        annotations = [
            {
                "category_id": 0,
                "category_name": "person",
                "bbox": [100, 100, 200, 150],
                "segmentation": [[100, 100, 300, 100, 300, 250, 100, 250]]
            },
            {
                "category_id": 1,
                "category_name": "car",
                "bbox": [300, 200, 100, 100],
                "segmentation": [[350, 250, 400, 250, 400, 300, 350, 300]]
            }
        ]

        # With segmentation=False, should always return True
        result = self.visualizer._validate_segmentation_format(annotations)
        self.assertTrue(result)

        # With segmentation=True, valid data should pass
        seg_visualizer = TestGenericVisualizer(segmentation=True)
        result = seg_visualizer._validate_segmentation_format(annotations)
        self.assertTrue(result)

    def test_validate_segmentation_format_invalid(self):
        """Test segmentation format validation with invalid data."""
        annotations = [
            {
                "category_id": 0,
                "category_name": "person",
                "bbox": [100, 100, 200, 150],
                "segmentation": []  # Empty segmentation
            }
        ]

        # With segmentation=False, should still return True
        result = self.visualizer._validate_segmentation_format(annotations)
        self.assertTrue(result)

        # With segmentation=True, should raise ValueError
        seg_visualizer = TestGenericVisualizer(segmentation=True)
        with self.assertRaises(ValueError) as context:
            seg_visualizer._validate_segmentation_format(annotations)

        error_msg = str(context.exception)
        self.assertIn("segmentation", error_msg.lower())
        self.assertIn("person", error_msg)

    def test_validate_segmentation_format_missing_field(self):
        """Test segmentation format validation with missing segmentation field."""
        annotations = [
            {
                "category_id": 0,
                "category_name": "person",
                "bbox": [100, 100, 200, 150]
                # No segmentation field
            }
        ]

        seg_visualizer = TestGenericVisualizer(segmentation=True)
        with self.assertRaises(ValueError) as context:
            seg_visualizer._validate_segmentation_format(annotations)

        error_msg = str(context.exception)
        self.assertIn("segmentation", error_msg.lower())

    def test_draw_annotations_segmentation_mode(self):
        """Test drawing annotations in segmentation mode."""
        # Create a test image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        original_image = image.copy()

        annotations = [
            {
                "category_id": 0,
                "category_name": "person",
                "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]]
            }
        ]
        classes = ["person", "car"]

        # With segmentation=False but annotation has segmentation, should draw polygon
        result = self.visualizer._draw_annotations(image, annotations, classes)
        self.assertFalse(np.array_equal(result, original_image))

        # With segmentation=True, should also draw polygon
        seg_visualizer = TestGenericVisualizer(segmentation=True)
        image2 = original_image.copy()
        result2 = seg_visualizer._draw_annotations(image2, annotations, classes)
        self.assertFalse(np.array_equal(result2, original_image))

    def test_draw_annotations_bbox_mode(self):
        """Test drawing annotations in bbox mode."""
        # Create a test image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        original_image = image.copy()

        annotations = [
            {
                "category_id": 0,
                "category_name": "person",
                "bbox": [10, 10, 40, 40]  # x_min, y_min, width, height
            }
        ]
        classes = ["person", "car"]

        result = self.visualizer._draw_annotations(image, annotations, classes)
        self.assertFalse(np.array_equal(result, original_image))

    def test_draw_annotations_mixed_modes(self):
        """Test drawing annotations with mixed bbox and segmentation."""
        # Create a test image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        original_image = image.copy()

        annotations = [
            {
                "category_id": 0,
                "category_name": "person",
                "bbox": [10, 10, 40, 40]
            },
            {
                "category_id": 1,
                "category_name": "car",
                "segmentation": [[60, 60, 80, 60, 80, 80, 60, 80]]
            }
        ]
        classes = ["person", "car"]

        result = self.visualizer._draw_annotations(image, annotations, classes)
        self.assertFalse(np.array_equal(result, original_image))

    def test_draw_annotations_no_valid_data(self):
        """Test drawing annotations with no valid bbox or segmentation."""
        # Create a test image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        original_image = image.copy()

        annotations = [
            {
                "category_id": 0,
                "category_name": "person"
                # No bbox or segmentation
            }
        ]
        classes = ["person", "car"]

        # Should log warning but not crash
        result = self.visualizer._draw_annotations(image, annotations, classes)
        # Image should be unchanged
        self.assertTrue(np.array_equal(result, original_image))

    def test_draw_segmentation_polygon(self):
        """Test drawing segmentation polygon."""
        # Create a test image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        original_image = image.copy()

        annotation = {
            "category_id": 0,
            "category_name": "person",
            "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]]
        }
        color = (255, 0, 0)
        label = "person"

        # This is a protected method, but we can test it since we're in the same module
        self.visualizer._draw_segmentation_polygon(image, annotation, color, label)
        self.assertFalse(np.array_equal(image, original_image))

    def test_draw_segmentation_polygon_invalid(self):
        """Test drawing invalid segmentation polygon."""
        # Create a test image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        original_image = image.copy()

        # Too few points (need at least 6 coordinates for 3 points)
        annotation = {
            "segmentation": [[10, 10, 30, 10]]  # Only 2 points
        }
        color = (255, 0, 0)
        label = "test"

        # Should log warning but not crash
        self.visualizer._draw_segmentation_polygon(image, annotation, color, label)
        # Image should be unchanged
        self.assertTrue(np.array_equal(image, original_image))

    def test_draw_bounding_box(self):
        """Test drawing bounding box."""
        # Create a test image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        original_image = image.copy()

        annotation = {
            "category_id": 0,
            "category_name": "person",
            "bbox": [10, 10, 40, 40]  # x_min, y_min, width, height
        }
        color = (255, 0, 0)
        label = "person"

        self.visualizer._draw_bounding_box(image, annotation, color, label)
        self.assertFalse(np.array_equal(image, original_image))

    def test_draw_bounding_box_invalid(self):
        """Test drawing invalid bounding box."""
        # Create a test image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        original_image = image.copy()

        # Invalid bbox format
        annotation = {
            "bbox": [10, 10]  # Not enough values
        }
        color = (255, 0, 0)
        label = "test"

        # Should log warning but not crash
        self.visualizer._draw_bounding_box(image, annotation, color, label)
        # Image should be unchanged
        self.assertTrue(np.array_equal(image, original_image))

    def test_process_image_annotations_with_save(self):
        """Test processing image annotations with save directory."""
        # Create test image file
        image_path = os.path.join(self.temp_dir, "test.jpg")
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite(image_path, image)

        image_data = {
            "image_id": "test",
            "image_path": image_path,
            "width": 100,
            "height": 100,
            "annotations": [
                {
                    "category_id": 0,
                    "category_name": "person",
                    "bbox": [10, 10, 40, 40]
                }
            ]
        }
        classes = ["person", "car"]

        save_dir = os.path.join(self.temp_dir, "output")

        result = self.visualizer._process_image_annotations(
            image_data, classes, save_dir
        )

        self.assertTrue(result["processed"])
        self.assertTrue(result["saved"])
        self.assertIn("output_path", result)
        self.assertEqual(result["annotations_count"], 1)

        # Check saved image exists
        self.assertTrue(os.path.exists(result["output_path"]))

    def test_process_image_annotations_without_save(self):
        """Test processing image annotations without save directory."""
        # Create test image file
        image_path = os.path.join(self.temp_dir, "test.jpg")
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite(image_path, image)

        image_data = {
            "image_id": "test",
            "image_path": image_path,
            "width": 100,
            "height": 100,
            "annotations": [
                {
                    "category_id": 0,
                    "category_name": "person",
                    "bbox": [10, 10, 40, 40]
                }
            ]
        }
        classes = ["person", "car"]

        result = self.visualizer._process_image_annotations(
            image_data, classes, save_dir=None
        )

        self.assertTrue(result["processed"])
        self.assertNotIn("saved", result)  # Not saved when save_dir is None
        self.assertEqual(result["annotations_count"], 1)

    def test_process_image_annotations_image_read_failed(self):
        """Test processing image annotations when image read fails."""
        image_data = {
            "image_id": "test",
            "image_path": "/non/existent/image.jpg",
            "width": 100,
            "height": 100,
            "annotations": []
        }
        classes = ["person", "car"]

        result = self.visualizer._process_image_annotations(
            image_data, classes, save_dir=None
        )

        self.assertFalse(result["processed"])
        self.assertEqual(result["reason"], "image_read_failed")

    def test_process_image_annotations_segmentation_validation_failed(self):
        """Test processing image annotations when segmentation validation fails."""
        # Create test image file
        image_path = os.path.join(self.temp_dir, "test.jpg")
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite(image_path, image)

        image_data = {
            "image_id": "test",
            "image_path": image_path,
            "width": 100,
            "height": 100,
            "annotations": [
                {
                    "category_id": 0,
                    "category_name": "person",
                    "bbox": [10, 10, 40, 40]
                    # No segmentation data
                }
            ]
        }
        classes = ["person", "car"]

        # Use visualizer with segmentation=True (strict mode)
        seg_visualizer = TestGenericVisualizer(segmentation=True)

        result = seg_visualizer._process_image_annotations(
            image_data, classes, save_dir=None
        )

        self.assertFalse(result["processed"])
        self.assertEqual(result["reason"], "segmentation_validation_failed")
        self.assertIn("error", result)

    def test_create_results_template(self):
        """Test creating results template."""
        template = self.visualizer._create_results_template()

        expected_fields = [
            "images_processed",
            "images_with_annotations",
            "annotations_processed",
            "saved_images",
            "classes_found",
            "errors"
        ]

        for field in expected_fields:
            self.assertIn(field, template)

        self.assertEqual(template["images_processed"], 0)
        self.assertEqual(template["images_with_annotations"], 0)
        self.assertEqual(template["annotations_processed"], 0)
        self.assertEqual(template["saved_images"], 0)
        self.assertEqual(template["classes_found"], set())
        self.assertEqual(template["errors"], [])

    def test_create_results_template_with_kwargs(self):
        """Test creating results template with additional keyword arguments."""
        template = self.visualizer._create_results_template(
            custom_field="custom_value",
            another_field=123
        )

        self.assertIn("custom_field", template)
        self.assertEqual(template["custom_field"], "custom_value")
        self.assertIn("another_field", template)
        self.assertEqual(template["another_field"], 123)

    def test_update_results_from_image(self):
        """Test updating aggregated results from single image processing result."""
        results = self.visualizer._create_results_template()

        # Test with successful image processing
        image_result = {
            "processed": True,
            "saved": True,
            "annotations_count": 2
        }
        annotations = [
            {"category_id": 0, "category_name": "person"},
            {"category_id": 1, "category_name": "car"}
        ]

        self.visualizer._update_results_from_image(results, image_result, annotations)

        self.assertEqual(results["images_processed"], 1)
        self.assertEqual(results["images_with_annotations"], 1)
        self.assertEqual(results["annotations_processed"], 2)
        self.assertEqual(results["saved_images"], 1)
        self.assertEqual(results["classes_found"], {0, 1})
        self.assertEqual(results["errors"], [])

    def test_update_results_from_image_not_processed(self):
        """Test updating results when image was not processed."""
        results = self.visualizer._create_results_template()

        image_result = {
            "processed": False,
            "reason": "image_read_failed",
            "error": "Failed to read image"
        }
        annotations = []

        self.visualizer._update_results_from_image(results, image_result, annotations)

        # No counts should be updated
        self.assertEqual(results["images_processed"], 0)
        self.assertEqual(results["images_with_annotations"], 0)
        self.assertEqual(results["annotations_processed"], 0)
        self.assertEqual(results["saved_images"], 0)
        self.assertEqual(results["classes_found"], set())
        # Error should be added
        self.assertEqual(len(results["errors"]), 1)

    def test_update_results_from_image_no_annotations(self):
        """Test updating results when image has no annotations."""
        results = self.visualizer._create_results_template()

        image_result = {
            "processed": True,
            "saved": True,
            "annotations_count": 0
        }
        annotations = []

        self.visualizer._update_results_from_image(results, image_result, annotations)

        self.assertEqual(results["images_processed"], 1)
        self.assertEqual(results["images_with_annotations"], 0)  # No annotations
        self.assertEqual(results["annotations_processed"], 0)
        self.assertEqual(results["saved_images"], 1)
        self.assertEqual(results["classes_found"], set())

    def test_update_results_from_image_stopped_by_user(self):
        """Test updating results when visualization was stopped by user."""
        results = self.visualizer._create_results_template()

        image_result = {
            "processed": True,
            "stopped": True,
            "annotations_count": 1
        }
        annotations = [{"category_id": 0, "category_name": "person"}]

        self.visualizer._update_results_from_image(results, image_result, annotations)

        self.assertEqual(results["images_processed"], 1)
        self.assertTrue(results.get("stopped_by_user"))


if __name__ == "__main__":
    unittest.main()