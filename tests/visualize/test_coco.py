# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:40
@File    : test_coco.py
@Author  : zj
@Description: Tests for COCO visualization
"""

import os
import tempfile
import json
import unittest
import numpy as np
import cv2
from pathlib import Path

from dataflow.visualize.coco import CocoVisualizer


class TestCocoVisualizer(unittest.TestCase):
    """Test cases for CocoVisualizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = CocoVisualizer(verbose=False)
        self.temp_dir = tempfile.mkdtemp(prefix="test_coco_vis_")
        self.image_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        # Create a simple COCO annotation file
        self.annotation_file = os.path.join(self.temp_dir, "annotations.json")
        self.coco_data = {
            "info": {
                "year": 2026,
                "version": "1.0",
                "description": "Test dataset",
                "contributor": "Test",
                "url": "",
                "date_created": "2026-03-08"
            },
            "images": [
                {
                    "id": 1,
                    "file_name": "test1.jpg",
                    "width": 640,
                    "height": 480
                },
                {
                    "id": 2,
                    "file_name": "test2.jpg",
                    "width": 800,
                    "height": 600
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 100, 200, 150],  # x, y, width, height
                    "area": 30000,
                    "segmentation": [],
                    "iscrowd": 0
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [300, 200, 100, 100],
                    "area": 10000,
                    "segmentation": [[350, 250, 400, 250, 400, 300, 350, 300]],
                    "iscrowd": 0
                },
                {
                    "id": 3,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [150, 150, 100, 80],
                    "area": 8000,
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

        with open(self.annotation_file, "w") as f:
            json.dump(self.coco_data, f)

        # Create test images
        for img_info in self.coco_data["images"]:
            img_path = os.path.join(self.image_dir, img_info["file_name"])
            # Create a blank image with the specified dimensions
            img = np.ones((img_info["height"], img_info["width"], 3), dtype=np.uint8) * 255
            cv2.imwrite(img_path, img)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test visualizer initialization."""
        vis = CocoVisualizer()
        self.assertTrue(hasattr(vis, 'logger'))
        self.assertTrue(hasattr(vis, 'verbose'))

        vis = CocoVisualizer(verbose=False)
        self.assertFalse(vis.verbose)

        vis = CocoVisualizer(verbose=True)
        self.assertTrue(vis.verbose)

    def test_initialization_with_segmentation(self):
        """Test visualizer initialization with segmentation parameter."""
        vis = CocoVisualizer(segmentation=True)
        self.assertTrue(vis.segmentation)
        self.assertTrue(hasattr(vis, 'label_handler'))

        vis2 = CocoVisualizer(segmentation=False)
        self.assertFalse(vis2.segmentation)


    def test_validate_paths(self):
        """Test path validation."""
        # Valid directory
        self.assertTrue(self.visualizer.validate_input_path(self.temp_dir, is_dir=True))

        # Invalid directory
        self.assertFalse(self.visualizer.validate_input_path("/non/existent/dir", is_dir=True))

        # Valid file
        self.assertTrue(self.visualizer.validate_input_path(self.annotation_file, is_dir=False))

    def test_visualize_with_save(self):
        """Test visualization with save directory."""
        save_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(save_dir, exist_ok=True)

        result = self.visualizer.visualize(
            self.image_dir, self.annotation_file, save_dir=save_dir
        )

        self.assertIn("images_processed", result)
        self.assertIn("annotations_processed", result)
        self.assertIn("saved_images", result)
        self.assertEqual(result["save_dir"], save_dir)

        # Check that images were saved
        self.assertEqual(result["saved_images"], 2)  # Both images have annotations
        self.assertTrue(os.path.exists(os.path.join(save_dir, "test1.jpg")))
        self.assertTrue(os.path.exists(os.path.join(save_dir, "test2.jpg")))

    def test_visualize_with_missing_images(self):
        """Test visualization when some images are missing."""
        # Remove one image
        img_path = os.path.join(self.image_dir, "test2.jpg")
        os.remove(img_path)

        save_dir = os.path.join(self.temp_dir, "output")
        result = self.visualizer.visualize(
            self.image_dir, self.annotation_file, save_dir=save_dir
        )

        # Only one image should be processed
        self.assertEqual(result["images_processed"], 1)
        self.assertEqual(result["saved_images"], 1)

    def test_batch_visualize(self):
        """Test batch visualization."""
        # Create a second dataset
        temp_dir2 = tempfile.mkdtemp(prefix="test_coco_vis2_")
        image_dir2 = os.path.join(temp_dir2, "images")
        os.makedirs(image_dir2, exist_ok=True)

        annotation_file2 = os.path.join(temp_dir2, "annotations2.json")
        coco_data2 = {
            "info": self.coco_data["info"],
            "images": [{
                "id": 1,
                "file_name": "single.jpg",
                "width": 400,
                "height": 300
            }],
            "annotations": [{
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [50, 50, 100, 80],
                "area": 8000,
                "segmentation": [],
                "iscrowd": 0
            }],
            "categories": self.coco_data["categories"]
        }

        with open(annotation_file2, "w") as f:
            json.dump(coco_data2, f)

        # Create image
        img_path2 = os.path.join(image_dir2, "single.jpg")
        cv2.imwrite(img_path2, np.ones((300, 400, 3), dtype=np.uint8) * 255)

        # Run batch visualization
        image_dirs = [self.image_dir, image_dir2]
        annotation_files = [self.annotation_file, annotation_file2]
        save_dirs = [os.path.join(self.temp_dir, "out1"), os.path.join(temp_dir2, "out2")]

        results = self.visualizer.batch_visualize(image_dirs, annotation_files, save_dirs)

        self.assertEqual(len(results), 2)
        self.assertIsNotNone(results[0])
        self.assertIsNotNone(results[1])

        # Clean up second temp directory
        import shutil
        shutil.rmtree(temp_dir2)

    def test_visualization_with_segmentation_flag(self):
        """Test visualization with segmentation flag enabled."""
        # Create visualizer with segmentation flag
        seg_visualizer = CocoVisualizer(verbose=False, segmentation=True)

        # Our test data has mixed annotations: some with segmentation, some without
        # With segmentation flag, annotations without segmentation should be filtered out
        result = seg_visualizer.visualize(
            self.image_dir, self.annotation_file
        )

        # Check results: only annotation with segmentation (id=2) should be processed
        self.assertEqual(result["annotations_processed"], 1)
        self.assertEqual(result["categories_found"], ["car"])  # Only car category has segmentation

    def test_visualization_without_segmentation_flag(self):
        """Test visualization without segmentation flag (auto detection)."""
        # Without segmentation flag, all annotations should be processed
        result = self.visualizer.visualize(
            self.image_dir, self.annotation_file
        )

        # All 3 annotations should be processed
        self.assertEqual(result["annotations_processed"], 3)
        self.assertEqual(sorted(result["categories_found"]), ["car", "person"])

    def test_visualization_with_segmentation_only_data(self):
        """Test visualization when all annotations have segmentation."""
        # Create a COCO dataset where all annotations have segmentation
        seg_only_file = os.path.join(self.temp_dir, "seg_only.json")
        seg_only_data = {
            "info": self.coco_data["info"],
            "images": self.coco_data["images"][:1],  # Just first image
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 100, 200, 150],
                    "area": 30000,
                    "segmentation": [[100, 100, 300, 100, 300, 250, 100, 250]],
                    "iscrowd": 0
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [300, 200, 100, 100],
                    "area": 10000,
                    "segmentation": [[350, 250, 400, 250, 400, 300, 350, 300]],
                    "iscrowd": 0
                }
            ],
            "categories": self.coco_data["categories"]
        }

        with open(seg_only_file, "w") as f:
            json.dump(seg_only_data, f)

        # With segmentation flag, both annotations should be processed
        seg_visualizer = CocoVisualizer(verbose=False, segmentation=True)
        result = seg_visualizer.visualize(
            self.image_dir, seg_only_file
        )

        self.assertEqual(result["annotations_processed"], 2)
        self.assertEqual(sorted(result["categories_found"]), [1, 2])

    def test_visualization_invalid_json(self):
        """Test visualization with invalid JSON file."""
        invalid_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_file, "w") as f:
            f.write("{ invalid json")

        with self.assertRaises(ValueError):
            self.visualizer.visualize(self.image_dir, invalid_file)

    def test_visualization_nonexistent_json(self):
        """Test visualization with non-existent JSON file."""
        with self.assertRaises(ValueError):
            self.visualizer.visualize(self.image_dir, "/non/existent/file.json")

    def test_visualization_nonexistent_image_dir(self):
        """Test visualization with non-existent image directory."""
        with self.assertRaises(ValueError):
            self.visualizer.visualize("/non/existent/dir", self.annotation_file)


if __name__ == "__main__":
    unittest.main()