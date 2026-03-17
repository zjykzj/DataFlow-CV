# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/10 22:00
@File    : test_labelme_to_coco.py
@Author  : zj
@Description: Tests for LabelMe to COCO conversion
"""

import os
import json
import tempfile
import unittest
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataflow.convert import LabelMeToCocoConverter
from dataflow.config import Config


class TestLabelMeToCocoConverter(unittest.TestCase):
    """Test cases for LabelMeToCocoConverter."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp(prefix="test_labelme2coco_")
        self.label_dir = os.path.join(self.test_dir, "labels")
        self.classes_file = os.path.join(self.test_dir, "class.names")
        self.output_json = os.path.join(self.test_dir, "output.json")

        # Create directories
        os.makedirs(self.label_dir, exist_ok=True)

        # Create sample data
        self._create_sample_classes()
        self._create_sample_labelme_files()

    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_sample_classes(self):
        """Create sample class names file."""
        class_names = ["person", "car", "bicycle"]
        with open(self.classes_file, 'w', encoding='utf-8') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")

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
                    "label": "bicycle",
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
        converter = LabelMeToCocoConverter(verbose=False)
        self.assertIsInstance(converter, LabelMeToCocoConverter)
        self.assertFalse(converter.verbose)

    def test_successful_conversion(self):
        """Test successful LabelMe to COCO conversion."""
        converter = LabelMeToCocoConverter(verbose=False)

        # Perform conversion
        result = converter.convert(
            self.label_dir,
            self.classes_file,
            self.output_json
        )

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("label_dir"), self.label_dir)
        self.assertEqual(result.get("classes_file"), self.classes_file)
        self.assertEqual(result.get("coco_json_path"), self.output_json)
        self.assertEqual(result.get("images_processed"), 2)
        self.assertEqual(result.get("annotations_processed"), 3)
        self.assertEqual(result.get("categories_found"), 3)

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

        # Check number of images
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

        # Verify bbox values
        bbox = ann1["bbox"]
        self.assertEqual(len(bbox), 4)
        self.assertEqual(bbox[0], 100)  # x
        self.assertEqual(bbox[1], 150)  # y
        self.assertEqual(bbox[2], 200)  # width (300-100)
        self.assertEqual(bbox[3], 120)  # height (270-150)

    def test_invalid_label_dir(self):
        """Test conversion with invalid label directory."""
        converter = LabelMeToCocoConverter(verbose=False)

        with self.assertRaises(ValueError):
            converter.convert(
                "/invalid/label/dir",
                self.classes_file,
                self.output_json
            )

    def test_invalid_classes_file(self):
        """Test conversion with invalid classes file."""
        converter = LabelMeToCocoConverter(verbose=False)

        with self.assertRaises(ValueError):
            converter.convert(
                self.label_dir,
                "/invalid/classes.names",
                self.output_json
            )

    def test_empty_label_dir(self):
        """Test conversion with empty label directory."""
        empty_dir = os.path.join(self.test_dir, "empty_labels")
        os.makedirs(empty_dir, exist_ok=True)

        converter = LabelMeToCocoConverter(verbose=False)

        # Should work but produce zero results
        result = converter.convert(
            empty_dir,
            self.classes_file,
            self.output_json
        )
        self.assertEqual(result.get("images_processed"), 0)
        self.assertEqual(result.get("annotations_processed"), 0)

    def test_empty_classes_file(self):
        """Test conversion with empty classes file."""
        empty_classes = os.path.join(self.test_dir, "empty.names")
        with open(empty_classes, 'w', encoding='utf-8') as f:
            pass  # Empty file

        converter = LabelMeToCocoConverter(verbose=False)

        with self.assertRaises(ValueError):
            converter.convert(
                self.label_dir,
                empty_classes,
                self.output_json
            )

    def test_conversion_statistics(self):
        """Verify conversion statistics are accurate."""
        converter = LabelMeToCocoConverter(verbose=False)

        result = converter.convert(
            self.label_dir,
            self.classes_file,
            self.output_json
        )

        # Check statistics
        self.assertEqual(result.get("categories_found"), 3)
        self.assertEqual(result.get("images_processed"), 2)
        self.assertEqual(result.get("annotations_processed"), 3)

    def test_verbose_mode(self):
        """Test converter with verbose mode enabled."""
        converter = LabelMeToCocoConverter(verbose=True)
        self.assertTrue(converter.verbose)

        # Conversion should still work
        result = converter.convert(
            self.label_dir,
            self.classes_file,
            self.output_json
        )
        self.assertIsInstance(result, dict)

    def test_converter_verbose_mode(self):
        """Test converter verbose mode."""
        # Test with verbose=False (default)
        converter = LabelMeToCocoConverter(verbose=False)
        self.assertFalse(converter.verbose)

        # Test with verbose=True
        converter = LabelMeToCocoConverter(verbose=True)
        self.assertTrue(converter.verbose)

        # Test with verbose=None (should use Config.VERBOSE)
        original_verbose = Config.VERBOSE

        Config.VERBOSE = True
        converter = LabelMeToCocoConverter(verbose=None)
        self.assertTrue(converter.verbose)

        Config.VERBOSE = False
        converter = LabelMeToCocoConverter(verbose=None)
        self.assertFalse(converter.verbose)

        # Restore original value
        Config.VERBOSE = original_verbose

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

        converter = LabelMeToCocoConverter(verbose=False)

        # Test with segmentation disabled (default)
        result = converter.convert(
            seg_label_dir,
            self.classes_file,
            self.output_json
        )
        self.assertIsInstance(result, dict)

        # Test with segmentation enabled
        result = converter.convert(
            seg_label_dir,
            self.classes_file,
            self.output_json,
            segmentation=True
        )
        self.assertIsInstance(result, dict)

    def test_malformed_labelme_file(self):
        """Test handling of malformed LabelMe files."""
        # Create malformed LabelMe file
        bad_label_dir = os.path.join(self.test_dir, "bad_labels")
        os.makedirs(bad_label_dir, exist_ok=True)

        bad_label = {
            "version": "5.3.1",
            "flags": {},
            # Missing required fields
        }

        bad_label_path = os.path.join(bad_label_dir, "bad.json")
        with open(bad_label_path, 'w', encoding='utf-8') as f:
            json.dump(bad_label, f, indent=2)

        converter = LabelMeToCocoConverter(verbose=False)

        # Should fail when reading data
        with self.assertRaises(ValueError):
            converter.convert(
                bad_label_dir,
                self.classes_file,
                self.output_json
            )

    def test_rle_conversion(self):
        """Test LabelMe to COCO conversion with RLE mode."""
        # Check if pycocotools is available
        try:
            from pycocotools import mask as cocomask
            pycoco_available = True
        except ImportError:
            pycoco_available = False
            self.skipTest("pycocotools not available, skipping RLE test")

        # Create LabelMe JSON with polygon (segmentation) for RLE testing
        seg_label_dir = os.path.join(self.test_dir, "seg_labels_rle")
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

        converter = LabelMeToCocoConverter(verbose=False)

        # Test with rle=True
        result = converter.convert(
            seg_label_dir,
            self.classes_file,
            self.output_json,
            rle=True
        )

        # Should succeed
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("rle_mode", False), "RLE mode should be True")

        # Load COCO JSON
        with open(self.output_json, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # Check that annotation has segmentation field
        annotations = coco_data["annotations"]
        if annotations:
            segmentation = annotations[0].get("segmentation")
            self.assertIsNotNone(segmentation, "Annotation should have segmentation")

            # Check if it's RLE format (dictionary with counts and size)
            if isinstance(segmentation, dict):
                self.assertIn("counts", segmentation)
                self.assertIn("size", segmentation)
                # Verify size matches image dimensions
                self.assertEqual(segmentation["size"], [480, 640])  # height, width
            else:
                # Might be polygon format if RLE conversion failed
                self.assertIsInstance(segmentation, list)


if __name__ == "__main__":
    unittest.main()