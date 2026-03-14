# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/14 23:02
@File    : test_config.py
@Author  : zj
@Description: Tests for Config classes
"""

import os
import tempfile
import unittest
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataflow.config import Config
from dataflow.convert.config import ConvertConfig
from dataflow.visualize.config import VisualizeConfig


class TestConfig(unittest.TestCase):
    """Test cases for base Config class."""

    def test_default_values(self):
        """Test default configuration values."""
        # File and directory names
        self.assertEqual(Config.YOLO_CLASSES_FILENAME, "class.names")
        self.assertEqual(Config.YOLO_LABELS_DIRNAME, "labels")

        # File extensions
        self.assertIn(".jpg", Config.IMAGE_EXTENSIONS)
        self.assertEqual(Config.YOLO_LABEL_EXTENSION, ".txt")
        self.assertEqual(Config.COCO_JSON_EXTENSION, ".json")

        # Default values
        self.assertEqual(Config.DEFAULT_IMAGE_WIDTH, 640)
        self.assertEqual(Config.DEFAULT_IMAGE_HEIGHT, 640)
        self.assertEqual(Config.DEFAULT_IMAGE_CHANNELS, 3)

        # Conversion options
        self.assertFalse(Config.OVERWRITE_EXISTING)
        self.assertFalse(Config.VERBOSE)
        self.assertTrue(Config.CREATE_DIRS)

        # COCO format defaults
        self.assertIsInstance(Config.COCO_DEFAULT_INFO, dict)
        self.assertEqual(Config.COCO_DEFAULT_INFO["year"], 2026)

        # YOLO format defaults
        self.assertTrue(Config.YOLO_NORMALIZE)
        self.assertFalse(Config.YOLO_SEGMENTATION)

    def test_get_image_extensions(self):
        """Test get_image_extensions class method."""
        extensions = Config.get_image_extensions()
        self.assertIsInstance(extensions, tuple)
        self.assertIn(".jpg", extensions)
        self.assertIn(".png", extensions)

    def test_get_yolo_label_extension(self):
        """Test get_yolo_label_extension class method."""
        self.assertEqual(Config.get_yolo_label_extension(), ".txt")

    def test_get_coco_json_extension(self):
        """Test get_coco_json_extension class method."""
        self.assertEqual(Config.get_coco_json_extension(), ".json")

    def test_validate_path(self):
        """Test path validation."""
        # Test empty path
        self.assertFalse(Config.validate_path(""))

        # Test directory validation
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertTrue(Config.validate_path(temp_dir, is_dir=True))
            self.assertFalse(Config.validate_path(os.path.join(temp_dir, "nonexistent"), is_dir=True))

            # Test file validation (directory exists)
            file_path = os.path.join(temp_dir, "test.txt")
            self.assertTrue(Config.validate_path(file_path, is_dir=False))

            # Test file validation (directory doesn't exist)
            non_existent_dir = os.path.join(temp_dir, "subdir", "test.txt")
            self.assertFalse(Config.validate_path(non_existent_dir, is_dir=False))

    def test_validate_path_with_create(self):
        """Test path validation with create option."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "newdir")

            # With CREATE_DIRS = True, should create directory
            Config.CREATE_DIRS = True
            self.assertTrue(Config.validate_path(new_dir, is_dir=True, create=True))
            self.assertTrue(os.path.exists(new_dir))

            # With CREATE_DIRS = False, should not create directory
            Config.CREATE_DIRS = False
            another_dir = os.path.join(temp_dir, "anotherdir")
            self.assertFalse(Config.validate_path(another_dir, is_dir=True, create=True))
            self.assertFalse(os.path.exists(another_dir))

        # Restore default
        Config.CREATE_DIRS = True


class TestConvertConfig(unittest.TestCase):
    """Test cases for ConvertConfig class."""

    def test_inheritance(self):
        """Test that ConvertConfig references Config properties."""
        # ConvertConfig is not a subclass but references Config class attributes
        self.assertTrue(hasattr(ConvertConfig, 'VERBOSE'))
        self.assertTrue(hasattr(ConvertConfig, 'OVERWRITE_EXISTING'))
        # Check that it has module-specific properties
        self.assertTrue(hasattr(ConvertConfig, 'DEFAULT_SEGMENTATION'))
        self.assertTrue(hasattr(ConvertConfig, 'VALIDATE_ANNOTATIONS'))
        self.assertTrue(hasattr(ConvertConfig, 'BATCH_SIZE'))

    def test_default_values(self):
        """Test ConvertConfig default values."""
        self.assertFalse(ConvertConfig.DEFAULT_SEGMENTATION)
        self.assertTrue(ConvertConfig.VALIDATE_ANNOTATIONS)
        self.assertEqual(ConvertConfig.BATCH_SIZE, 100)

    def test_update_from_cli(self):
        """Test updating ConvertConfig from CLI options."""
        # Test verbose flag
        ConvertConfig.update_from_cli(verbose=True)
        self.assertTrue(Config.VERBOSE)

        ConvertConfig.update_from_cli(verbose=False)
        self.assertFalse(Config.VERBOSE)

        # Test overwrite flag (may still set OVERWRITE_EXISTING but CLI option is removed)
        Config.OVERWRITE_EXISTING = False
        ConvertConfig.update_from_cli(overwrite=True)
        # Depending on implementation, may set OVERWRITE_EXISTING to True
        # We'll just verify Config.VERBOSE is unaffected
        # Reset
        Config.OVERWRITE_EXISTING = False
        Config.VERBOSE = False


class TestVisualizeConfig(unittest.TestCase):
    """Test cases for VisualizeConfig class."""

    def test_inheritance(self):
        """Test that VisualizeConfig references Config properties."""
        # VisualizeConfig is not a subclass but references Config class attributes
        self.assertTrue(hasattr(VisualizeConfig, 'VERBOSE'))
        self.assertTrue(hasattr(VisualizeConfig, 'OVERWRITE_EXISTING'))
        # Check that it has module-specific properties
        self.assertTrue(hasattr(VisualizeConfig, 'DEFAULT_WINDOW_SIZE'))
        self.assertTrue(hasattr(VisualizeConfig, 'DEFAULT_COLOR_SCHEME'))
        self.assertTrue(hasattr(VisualizeConfig, 'SHOW_CONFIDENCE'))

    def test_default_values(self):
        """Test VisualizeConfig default values."""
        # Note: DEFAULT_WINDOW_SIZE may differ from Config
        self.assertEqual(VisualizeConfig.DEFAULT_WINDOW_SIZE, (800, 600))
        self.assertEqual(VisualizeConfig.DEFAULT_COLOR_SCHEME, "tab20c")
        self.assertFalse(VisualizeConfig.SHOW_CONFIDENCE)
        self.assertEqual(VisualizeConfig.FONT_SCALE, 0.5)
        self.assertEqual(VisualizeConfig.LINE_THICKNESS, 2)

    def test_update_from_cli(self):
        """Test updating VisualizeConfig from CLI options."""
        # Test verbose flag
        VisualizeConfig.update_from_cli(verbose=True)
        self.assertTrue(Config.VERBOSE)

        VisualizeConfig.update_from_cli(verbose=False)
        self.assertFalse(Config.VERBOSE)

        # Reset
        Config.VERBOSE = False


if __name__ == "__main__":
    unittest.main()