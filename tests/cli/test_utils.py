"""Tests for CLI utility functions."""

import tempfile
from pathlib import Path
import pytest
import click

from dataflow.cli.commands.utils import (
    validate_path_exists,
    validate_visualize_params,
    validate_convert_params,
)
from dataflow.cli.exceptions import InputError


def test_validate_path_exists():
    """Test validate_path_exists function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test existing path
        existing_path = Path(tmpdir) / "test.txt"
        existing_path.touch()

        result = validate_path_exists(existing_path, "测试路径")
        assert result == existing_path

        # Test non-existent path
        non_existent = Path(tmpdir) / "nonexistent.txt"
        with pytest.raises(InputError) as exc_info:
            validate_path_exists(non_existent, "测试路径")

        assert "does not exist" in str(exc_info.value)
        assert str(non_existent) in str(exc_info.value)


def test_validate_visualize_params():
    """Test validate_visualize_params function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test paths
        input_path = Path(tmpdir) / "input"
        input_path.mkdir()

        image_dir = Path(tmpdir) / "images"
        image_dir.mkdir()

        output_dir = Path(tmpdir) / "output"

        # Test with all paths
        validated_input, validated_image, validated_output = validate_visualize_params(
            input_path, image_dir, output_dir
        )

        assert validated_input == input_path
        assert validated_image == image_dir
        assert validated_output == output_dir
        assert validated_output.exists()  # Should be created

        # Test with None image_dir
        validated_input, validated_image, validated_output = validate_visualize_params(
            input_path, None, None
        )

        assert validated_input == input_path
        assert validated_image is None
        assert validated_output is None

        # Test with non-existent input path
        non_existent = Path(tmpdir) / "nonexistent"
        with pytest.raises(InputError):
            validate_visualize_params(non_existent, None, None)


def test_validate_convert_params():
    """Test validate_convert_params function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test paths
        input_path = Path(tmpdir) / "input"
        input_path.mkdir()

        # Test with file output path
        output_file = Path(tmpdir) / "output" / "annotations.json"

        image_dir = Path(tmpdir) / "images"
        image_dir.mkdir()

        class_file = Path(tmpdir) / "classes.txt"
        class_file.touch()

        # Test with all paths
        validated_input, validated_output, validated_image, validated_class = validate_convert_params(
            "yolo", "coco", input_path, output_file, image_dir, class_file
        )

        assert validated_input == input_path
        assert validated_output == output_file
        assert validated_image == image_dir
        assert validated_class == class_file
        assert output_file.parent.exists()  # Parent directory should be created

        # Test with directory output path
        output_dir = Path(tmpdir) / "output_dir"

        validated_input, validated_output, validated_image, validated_class = validate_convert_params(
            "coco", "yolo", input_path, output_dir, None, None
        )

        assert validated_input == input_path
        assert validated_output == output_dir
        assert validated_image is None
        assert validated_class is None
        assert output_dir.exists()  # Directory should be created

        # Test with non-existent input path
        non_existent = Path(tmpdir) / "nonexistent"
        with pytest.raises(InputError):
            validate_convert_params("yolo", "coco", non_existent, output_dir, None, None)

        # Test with non-existent image_dir when provided
        non_existent_image = Path(tmpdir) / "nonexistent_images"
        with pytest.raises(InputError):
            validate_convert_params("yolo", "coco", input_path, output_dir, non_existent_image, None)

        # Test with non-existent class_file when provided
        non_existent_class = Path(tmpdir) / "nonexistent_classes.txt"
        with pytest.raises(InputError):
            validate_convert_params("yolo", "coco", input_path, output_dir, None, non_existent_class)