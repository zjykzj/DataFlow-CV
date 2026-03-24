"""
Unit tests for utils.py
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from dataflow.convert import utils
from dataflow.label.models import (BoundingBox, DatasetAnnotations,
                                   ImageAnnotation, ObjectAnnotation)


class TestCategoryFunctions:
    """Test category-related utility functions."""

    def test_extract_categories_from_annotations(self):
        """Test extracting categories from DatasetAnnotations."""
        annotations = DatasetAnnotations(categories={0: "cat", 1: "dog", 2: "bird"})
        categories = utils.extract_categories_from_annotations(annotations)
        assert categories == {0: "cat", 1: "dog", 2: "bird"}

    def test_generate_classes_file(self):
        """Test generating classes.txt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "classes.txt"
            categories = {0: "cat", 1: "dog", 2: "bird"}

            success = utils.generate_classes_file(categories, output_path)
            assert success is True
            assert output_path.exists()

            # Verify file content
            with open(output_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines()]
            assert lines == ["cat", "dog", "bird"]

    def test_generate_classes_file_invalid_path(self):
        """Test generating classes.txt with invalid path."""
        categories = {0: "cat"}
        # Try to write to non-existent parent directory
        output_path = Path("/nonexistent/directory/classes.txt")
        success = utils.generate_classes_file(categories, output_path)
        assert success is False

    def test_load_classes_file(self):
        """Test loading categories from classes.txt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class_file = Path(tmpdir) / "classes.txt"
            with open(class_file, "w", encoding="utf-8") as f:
                f.write("cat\n")
                f.write("dog\n")
                f.write("bird\n")
                f.write("\n")  # Empty line should be skipped
                f.write("car\n")

            categories = utils.load_classes_file(class_file)
            # Note: empty lines are skipped but still occupy an index
            # Line indices: 0="cat", 1="dog", 2="bird", 3="", 4="car"
            assert categories == {0: "cat", 1: "dog", 2: "bird", 4: "car"}

    def test_load_classes_file_nonexistent(self):
        """Test loading categories from non-existent file."""
        class_file = Path("/nonexistent/classes.txt")
        categories = utils.load_classes_file(class_file)
        assert categories == {}  # Should return empty dict

    def test_extract_categories_from_coco(self):
        """Test extracting categories from COCO data."""
        coco_data = {
            "categories": [
                {"id": 1, "name": "person", "supercategory": "human"},
                {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
                {"id": 3, "name": "car", "supercategory": "vehicle"},
            ]
        }
        categories = utils.extract_categories_from_coco(coco_data)
        assert categories == {1: "person", 2: "bicycle", 3: "car"}

    def test_extract_categories_from_coco_empty(self):
        """Test extracting categories from COCO data without categories."""
        coco_data = {}
        categories = utils.extract_categories_from_coco(coco_data)
        assert categories == {}

    def test_ensure_categories_in_annotations(self):
        """Test ensuring annotations have specific categories."""
        annotations = DatasetAnnotations(categories={0: "old_cat", 1: "old_dog"})
        new_categories = {0: "cat", 1: "dog", 2: "bird"}

        updated = utils.ensure_categories_in_annotations(annotations, new_categories)
        assert updated.categories == new_categories

    def test_ensure_categories_in_annotations_conflict(self):
        """Test ensuring categories with conflicts (should log warning)."""
        annotations = DatasetAnnotations(categories={0: "cat", 1: "dog"})
        new_categories = {0: "CAT", 1: "dog"}  # Conflict at ID 0

        # This should log a warning but update categories
        updated = utils.ensure_categories_in_annotations(annotations, new_categories)
        assert updated.categories == new_categories


class TestPathFunctions:
    """Test path-related utility functions."""

    def test_normalize_path_absolute(self):
        """Test normalizing absolute path."""
        base_dir = Path("/base")
        abs_path = "/absolute/path"
        result = utils.normalize_path(abs_path, base_dir)
        assert result == Path(abs_path).resolve()

    def test_normalize_path_relative(self):
        """Test normalizing relative path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            rel_path = "subdir/file.txt"
            result = utils.normalize_path(rel_path, base_dir)
            expected = (base_dir / rel_path).resolve()
            assert result == expected

    def test_validate_conversion_chain(self):
        """Test validating conversion chains."""
        allowed_chains = [("labelme", "yolo"), ("yolo", "coco"), ("coco", "labelme")]

        assert (
            utils.validate_conversion_chain("labelme", "yolo", allowed_chains) is True
        )
        assert utils.validate_conversion_chain("yolo", "coco", allowed_chains) is True
        assert (
            utils.validate_conversion_chain("labelme", "coco", allowed_chains) is False
        )
        assert (
            utils.validate_conversion_chain("unknown", "yolo", allowed_chains) is False
        )

    def test_create_conversion_chain(self):
        """Test creating conversion steps from format chain."""
        chain = ["labelme", "yolo", "coco", "labelme"]
        steps = utils.create_conversion_chain(chain)
        assert steps == [("labelme", "yolo"), ("yolo", "coco"), ("coco", "labelme")]

        # Test with minimal chain
        chain2 = ["a", "b"]
        steps2 = utils.create_conversion_chain(chain2)
        assert steps2 == [("a", "b")]

        # Test with single format (no conversion)
        chain3 = ["a"]
        steps3 = utils.create_conversion_chain(chain3)
        assert steps3 == []

    def test_resolve_image_paths(self):
        """Test resolving and updating image paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            target_dir = Path(tmpdir) / "target"
            source_dir.mkdir()
            target_dir.mkdir()

            # Create a subdirectory structure
            images_subdir = source_dir / "images"
            images_subdir.mkdir()

            annotations = DatasetAnnotations(
                images=[
                    ImageAnnotation(
                        image_id="img1",
                        image_path="images/img1.jpg",  # Relative path
                        width=100,
                        height=100,
                        objects=[],
                    ),
                    ImageAnnotation(
                        image_id="img2",
                        image_path=str(source_dir / "img2.jpg"),  # Absolute path
                        width=200,
                        height=200,
                        objects=[],
                    ),
                ]
            )

            updated = utils.resolve_image_paths(annotations, source_dir, target_dir)

            # First image should have relative path preserved
            assert updated.images[0].image_path == str(target_dir / "images/img1.jpg")
            # Second image (absolute) should just use filename
            assert updated.images[1].image_path == str(target_dir / "img2.jpg")


class TestImageDimensionFunction:
    """Test image dimension function."""

    def test_get_image_dimensions_from_handler_missing_libs(self, monkeypatch):
        """Test getting image dimensions when no imaging lib is available."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "cv2" or (name == "PIL" and "Image" in fromlist):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        handler = Mock()
        image_path = "/path/to/image.jpg"

        from dataflow.convert import utils

        with pytest.raises(ImportError) as exc_info:
            utils.get_image_dimensions_from_handler(handler, image_path)
        assert "Cannot determine image dimensions" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
