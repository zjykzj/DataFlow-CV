# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/9 22:30
@File    : test_labelme.py
@Author  : zj
@Description: Tests for LabelMe visualization
"""

import os
import tempfile
import json
import unittest
import numpy as np
import cv2
from pathlib import Path

from dataflow.visualize.labelme import LabelMeVisualizer


class TestLabelMeVisualizer(unittest.TestCase):
    """Test cases for LabelMeVisualizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = LabelMeVisualizer(verbose=False)
        self.temp_dir = tempfile.mkdtemp(prefix="test_labelme_vis_")
        self.image_dir = os.path.join(self.temp_dir, "images")
        self.label_dir = os.path.join(self.temp_dir, "labels")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

        # Create a test image
        self.image_path = os.path.join(self.image_dir, "test.jpg")
        image = np.ones((640, 480, 3), dtype=np.uint8) * 255
        cv2.imwrite(self.image_path, image)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_labelme_json(self, filename, shapes):
        """Helper to create LabelMe JSON file."""
        json_path = os.path.join(self.label_dir, filename)

        # Calculate relative path from label directory to image
        image_path_rel = os.path.relpath(self.image_path, self.label_dir)

        labelme_data = {
            "version": "5.3.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_path_rel,
            "imageData": None,
            "imageHeight": 640,
            "imageWidth": 480
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, indent=2, ensure_ascii=False)

        return json_path

    def test_initialization(self):
        """Test visualizer initialization."""
        vis = LabelMeVisualizer()
        self.assertTrue(hasattr(vis, 'logger'))
        self.assertTrue(hasattr(vis, 'verbose'))

        vis = LabelMeVisualizer(verbose=False)
        self.assertFalse(vis.verbose)

        vis = LabelMeVisualizer(verbose=True)
        self.assertTrue(vis.verbose)

    def test_initialization_with_segmentation(self):
        """Test visualizer initialization with segmentation parameter."""
        vis = LabelMeVisualizer(segmentation=True)
        self.assertTrue(vis.segmentation)
        self.assertTrue(hasattr(vis, 'label_handler'))

        vis2 = LabelMeVisualizer(segmentation=False)
        self.assertFalse(vis2.segmentation)

    def test_basic_visualization_rectangle(self):
        """Test basic LabelMe visualization with rectangle shapes."""
        # Create LabelMe JSON with rectangle shapes
        shapes = [
            {
                "label": "person",
                "points": [[100, 100], [200, 200]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            },
            {
                "label": "car",
                "points": [[300, 300], [350, 350]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
        ]

        self.create_labelme_json("test.json", shapes)

        # Visualize without saving (display mode)
        result = self.visualizer.visualize(
            self.image_dir, self.label_dir
        )

        # Check results
        self.assertIn("images_processed", result)
        self.assertIn("annotations_processed", result)
        self.assertIn("classes_found", result)
        self.assertEqual(result["images_processed"], 1)
        self.assertEqual(result["annotations_processed"], 2)
        self.assertEqual(sorted(result["classes_found"]), ["car", "person"])

    def test_basic_visualization_polygon(self):
        """Test basic LabelMe visualization with polygon shapes."""
        # Create LabelMe JSON with polygon shapes
        shapes = [
            {
                "label": "person",
                "points": [[100, 100], [200, 100], [150, 200]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            },
            {
                "label": "car",
                "points": [[300, 300], [350, 300], [350, 350], [300, 350]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
        ]

        self.create_labelme_json("test.json", shapes)

        # Visualize without segmentation flag (auto detection)
        result = self.visualizer.visualize(
            self.image_dir, self.label_dir
        )

        self.assertEqual(result["images_processed"], 1)
        self.assertEqual(result["annotations_processed"], 2)

    def test_visualization_with_segmentation_flag_polygon(self):
        """Test visualization with segmentation flag enabled (polygons only)."""
        # Create LabelMe JSON with polygon shapes only
        shapes = [
            {
                "label": "person",
                "points": [[100, 100], [200, 100], [150, 200]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
        ]

        self.create_labelme_json("test.json", shapes)

        # Create visualizer with segmentation flag
        seg_visualizer = LabelMeVisualizer(verbose=False, segmentation=True)

        # Should work with polygon data
        result = seg_visualizer.visualize(
            self.image_dir, self.label_dir
        )
        self.assertEqual(result["images_processed"], 1)
        self.assertEqual(result["annotations_processed"], 1)

    def test_visualization_with_segmentation_flag_strict_fail(self):
        """Test strict segmentation mode failure when rectangle shapes are present."""
        # Create LabelMe JSON with rectangle shapes only
        shapes = [
            {
                "label": "person",
                "points": [[100, 100], [200, 200]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
        ]

        self.create_labelme_json("test.json", shapes)

        # Create visualizer with segmentation flag (strict mode)
        seg_visualizer = LabelMeVisualizer(verbose=False, segmentation=True)

        # Should raise ValueError because require_segmentation=True but shapes are rectangles
        with self.assertRaises(ValueError):
            seg_visualizer.visualize(
                self.image_dir, self.label_dir
            )

    def test_visualization_mixed_shapes_auto_detection(self):
        """Test auto detection with mixed rectangle and polygon shapes."""
        # Create LabelMe JSON with mixed shapes
        shapes = [
            {
                "label": "person",
                "points": [[100, 100], [200, 200]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            },
            {
                "label": "car",
                "points": [[300, 300], [350, 300], [350, 350], [300, 350]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
        ]

        self.create_labelme_json("test.json", shapes)

        # Visualize without segmentation flag (auto detection)
        result = self.visualizer.visualize(
            self.image_dir, self.label_dir
        )

        # Both annotations should be processed
        self.assertEqual(result["annotations_processed"], 2)
        self.assertEqual(sorted(result["classes_found"]), ["car", "person"])

    def test_visualization_with_save_dir(self):
        """Test visualization with save directory."""
        shapes = [
            {
                "label": "person",
                "points": [[100, 100], [200, 200]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
        ]

        self.create_labelme_json("test.json", shapes)

        save_dir = os.path.join(self.temp_dir, "output")

        result = self.visualizer.visualize(
            self.image_dir, self.label_dir, save_dir=save_dir
        )

        self.assertEqual(result["images_processed"], 1)
        self.assertEqual(result["saved_images"], 1)
        self.assertTrue(os.path.exists(save_dir))

        # Check saved image exists
        saved_image_path = os.path.join(save_dir, "test.jpg")
        self.assertTrue(os.path.exists(saved_image_path))

    def test_visualization_multiple_json_files(self):
        """Test visualization with multiple JSON files in directory."""
        # Create first JSON file
        shapes1 = [
            {
                "label": "person",
                "points": [[100, 100], [200, 200]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
        ]
        self.create_labelme_json("image1.json", shapes1)

        # Create second image and JSON file
        image_path2 = os.path.join(self.image_dir, "test2.jpg")
        image2 = np.ones((640, 480, 3), dtype=np.uint8) * 128
        cv2.imwrite(image_path2, image2)

        # Need to update imagePath in second JSON
        shapes2 = [
            {
                "label": "car",
                "points": [[300, 300], [350, 350]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
        ]

        json_path2 = os.path.join(self.label_dir, "image2.json")
        # Calculate relative path from label directory to second image
        image_path2_rel = os.path.relpath(image_path2, self.label_dir)
        labelme_data2 = {
            "version": "5.3.1",
            "flags": {},
            "shapes": shapes2,
            "imagePath": image_path2_rel,
            "imageData": None,
            "imageHeight": 640,
            "imageWidth": 480
        }

        with open(json_path2, 'w', encoding='utf-8') as f:
            json.dump(labelme_data2, f, indent=2, ensure_ascii=False)

        # Visualize
        result = self.visualizer.visualize(
            self.image_dir, self.label_dir
        )

        # Both images should be processed
        self.assertEqual(result["images_processed"], 2)
        self.assertEqual(result["annotations_processed"], 2)

    def test_visualization_invalid_json(self):
        """Test visualization with invalid JSON file."""
        invalid_file = os.path.join(self.label_dir, "invalid.json")
        with open(invalid_file, "w") as f:
            f.write("{ invalid json")

        with self.assertRaises(ValueError):
            self.visualizer.visualize(self.image_dir, self.label_dir)

    def test_visualization_nonexistent_dirs(self):
        """Test visualization with non-existent directories."""
        with self.assertRaises(ValueError):
            self.visualizer.visualize("/non/existent/image_dir", self.label_dir)

        with self.assertRaises(ValueError):
            self.visualizer.visualize(self.image_dir, "/non/existent/label_dir")

    def test_batch_visualize(self):
        """Test batch visualization method."""
        # Create test data for multiple datasets
        image_dirs = [self.image_dir, self.image_dir]
        label_dirs = [self.label_dir, self.label_dir]

        # Create LabelMe JSON file
        shapes = [
            {
                "label": "person",
                "points": [[100, 100], [200, 200]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
        ]
        self.create_labelme_json("test.json", shapes)

        results = self.visualizer.batch_visualize(
            image_dirs, label_dirs
        )

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsNotNone(result)
            self.assertEqual(result["images_processed"], 1)

    def test_extract_classes_method(self):
        """Test _extract_classes method."""
        # Create test annotations list
        annotations_list = [
            {
                "image_id": "test1",
                "image_path": "/path/to/test1.jpg",
                "width": 480,
                "height": 640,
                "annotations": [
                    {"category_name": "person", "category_id": 0},
                    {"category_name": "car", "category_id": 1}
                ]
            },
            {
                "image_id": "test2",
                "image_path": "/path/to/test2.jpg",
                "width": 480,
                "height": 640,
                "annotations": [
                    {"category_name": "person", "category_id": 0},
                    {"category_name": "bicycle", "category_id": 2}
                ]
            }
        ]

        classes = self.visualizer._extract_classes(annotations_list)
        self.assertEqual(len(classes), 3)
        self.assertEqual(sorted(classes), ["bicycle", "car", "person"])

    def test_verbose_flag(self):
        """Test that verbose flag controls logging level."""
        import logging
        # Test with verbose=False
        vis_false = LabelMeVisualizer(verbose=False)
        self.assertFalse(vis_false.verbose)
        self.assertEqual(vis_false.logger.level, logging.WARNING)

        # Test with verbose=True
        vis_true = LabelMeVisualizer(verbose=True)
        self.assertTrue(vis_true.verbose)
        self.assertEqual(vis_true.logger.level, logging.INFO)

        # Test default (should use Config.VERBOSE which defaults to False)
        from dataflow.config import Config
        original = Config.VERBOSE
        try:
            Config.VERBOSE = True
            vis_default = LabelMeVisualizer()
            self.assertTrue(vis_default.verbose)
            Config.VERBOSE = False
            vis_default2 = LabelMeVisualizer()
            self.assertFalse(vis_default2.verbose)
        finally:
            Config.VERBOSE = original

if __name__ == "__main__":
    unittest.main()