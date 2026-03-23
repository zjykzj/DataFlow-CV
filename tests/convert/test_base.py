"""
Unit tests for base.py
"""

import pytest
import logging
from unittest.mock import Mock, MagicMock
from pathlib import Path
import tempfile

from dataflow.convert.base import BaseConverter, ConversionResult
from dataflow.label.base import AnnotationResult
from dataflow.label.models import DatasetAnnotations, ImageAnnotation, ObjectAnnotation, BoundingBox


class ConcreteConverter(BaseConverter):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, source_format="test_source", target_format="test_target", **kwargs):
        super().__init__(source_format, target_format, **kwargs)

    def convert(self, source_path: str, target_path: str, **kwargs) -> ConversionResult:
        # Simple implementation that returns success
        return ConversionResult(
            success=True,
            source_format=self.source_format,
            target_format=self.target_format,
            source_path=source_path,
            target_path=target_path,
            num_images_converted=1,
            num_objects_converted=2
        )

    def create_source_handler(self, source_path: str, kwargs: dict):
        return Mock()

    def create_target_handler(self, target_path: str, kwargs: dict):
        return Mock()


class TestConversionResult:
    """Test suite for ConversionResult class."""

    def test_init_defaults(self):
        """Test ConversionResult initialization with defaults."""
        result = ConversionResult(
            success=True,
            source_format="labelme",
            target_format="yolo",
            source_path="/path/source",
            target_path="/path/target"
        )
        assert result.success is True
        assert result.source_format == "labelme"
        assert result.target_format == "yolo"
        assert result.source_path == "/path/source"
        assert result.target_path == "/path/target"
        assert result.num_images_converted == 0
        assert result.num_objects_converted == 0
        assert result.warnings == []
        assert result.errors == []
        assert result.metadata == {}

    def test_init_with_values(self):
        """Test ConversionResult initialization with custom values."""
        warnings = ["warning1", "warning2"]
        errors = ["error1", "error2"]
        metadata = {"key": "value"}
        result = ConversionResult(
            success=False,
            source_format="yolo",
            target_format="coco",
            source_path="/src",
            target_path="/dst",
            num_images_converted=10,
            num_objects_converted=25,
            warnings=warnings,
            errors=errors,
            metadata=metadata
        )
        assert result.success is False
        assert result.source_format == "yolo"
        assert result.target_format == "coco"
        assert result.num_images_converted == 10
        assert result.num_objects_converted == 25
        assert result.warnings == warnings
        assert result.errors == errors
        assert result.metadata == metadata

    def test_add_warning(self):
        """Test adding warning messages."""
        result = ConversionResult(
            success=True,
            source_format="a",
            target_format="b",
            source_path="",
            target_path=""
        )
        result.add_warning("Test warning")
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning"
        assert result.success is True  # Warnings don't affect success

    def test_add_error(self):
        """Test adding error messages."""
        result = ConversionResult(
            success=True,
            source_format="a",
            target_format="b",
            source_path="",
            target_path=""
        )
        result.add_error("Test error")
        assert len(result.errors) == 1
        assert result.errors[0] == "Test error"
        assert result.success is False  # Errors set success to False

    def test_add_metadata(self):
        """Test adding metadata."""
        result = ConversionResult(
            success=True,
            source_format="a",
            target_format="b",
            source_path="",
            target_path=""
        )
        result.add_metadata("key1", "value1")
        result.add_metadata("key2", 123)
        assert result.metadata["key1"] == "value1"
        assert result.metadata["key2"] == 123

    def test_get_summary_success(self):
        """Test summary generation for successful conversion."""
        result = ConversionResult(
            success=True,
            source_format="labelme",
            target_format="yolo",
            source_path="/src",
            target_path="/dst",
            num_images_converted=5,
            num_objects_converted=12
        )
        summary = result.get_summary()
        assert "Successfully converted" in summary
        assert "5 images" in summary
        assert "12 objects" in summary
        assert "labelme" in summary
        assert "yolo" in summary

    def test_get_summary_failure(self):
        """Test summary generation for failed conversion."""
        result = ConversionResult(
            success=False,
            source_format="yolo",
            target_format="coco",
            source_path="/src",
            target_path="/dst"
        )
        result.add_error("Error 1")
        result.add_error("Error 2")
        summary = result.get_summary()
        assert "Conversion failed" in summary
        assert "2 errors" in summary


class TestBaseConverter:
    """Test suite for BaseConverter class."""

    def test_init(self):
        """Test BaseConverter initialization."""
        converter = ConcreteConverter(
            source_format="labelme",
            target_format="yolo",
            strict_mode=False,
            logger=logging.getLogger("test")
        )
        assert converter.source_format == "labelme"
        assert converter.target_format == "yolo"
        assert converter.strict_mode is False
        assert converter.logger.name == "test"
        assert converter.file_ops is not None

    def test_validate_inputs_source_not_exist(self):
        """Test validation when source path doesn't exist."""
        converter = ConcreteConverter()
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "nonexistent"
            target_path = Path(tmpdir) / "output"
            assert not converter.validate_inputs(str(source_path), str(target_path), {})

    def test_validate_inputs_valid(self):
        """Test validation with valid inputs."""
        converter = ConcreteConverter()
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.txt"
            source_path.touch()
            target_path = Path(tmpdir) / "output" / "result.txt"
            assert converter.validate_inputs(str(source_path), str(target_path), {})

    def test_create_conversion_result(self):
        """Test _create_conversion_result helper method."""
        converter = ConcreteConverter()

        # Test with minimal parameters
        result1 = converter._create_conversion_result(
            success=True,
            source_path="/src",
            target_path="/dst"
        )
        assert result1.success is True
        assert result1.source_path == "/src"
        assert result1.target_path == "/dst"
        assert result1.num_images_converted == 0
        assert result1.num_objects_converted == 0

        # Test with annotations
        annotations = DatasetAnnotations(
            images=[
                ImageAnnotation(
                    image_id="img1",
                    image_path="/img1.jpg",
                    width=100,
                    height=100,
                    objects=[
                        ObjectAnnotation(class_id=0, class_name="cat", bbox=BoundingBox(0.1, 0.1, 0.2, 0.2)),
                        ObjectAnnotation(class_id=1, class_name="dog", bbox=BoundingBox(0.3, 0.3, 0.2, 0.2)),
                    ]
                ),
                ImageAnnotation(
                    image_id="img2",
                    image_path="/img2.jpg",
                    width=200,
                    height=200,
                    objects=[
                        ObjectAnnotation(class_id=0, class_name="cat", bbox=BoundingBox(0.2, 0.2, 0.1, 0.1)),
                    ]
                )
            ],
            categories={0: "cat", 1: "dog"}
        )
        result2 = converter._create_conversion_result(
            success=True,
            source_path="/src",
            target_path="/dst",
            annotations=annotations
        )
        assert result2.num_images_converted == 2
        assert result2.num_objects_converted == 3

        # Test with write result errors
        write_result = AnnotationResult(success=False)
        write_result.add_error("Write failed")
        result3 = converter._create_conversion_result(
            success=False,
            source_path="/src",
            target_path="/dst",
            write_result=write_result
        )
        assert result3.success is False
        assert len(result3.errors) == 1
        assert "Write failed" in result3.errors[0]

    def test_log_methods(self):
        """Test logging methods."""
        converter = ConcreteConverter(strict_mode=True)

        # Test info logging (should not raise)
        converter._log_info("Test info")

        # Test warning logging (should not raise)
        converter._log_warning("Test warning")

        # Test error logging in strict mode (should raise)
        with pytest.raises(ValueError, match="Test error"):
            converter._log_error("Test error")

        # Test error logging in non-strict mode (should not raise)
        converter2 = ConcreteConverter(strict_mode=False)
        converter2._log_error("Test error 2")  # Should not raise

    def test_convert_annotations_default(self):
        """Test default convert_annotations implementation."""
        converter = ConcreteConverter()
        annotations = DatasetAnnotations()
        result = converter.convert_annotations(annotations, {})
        assert result == annotations  # Default implementation returns as-is


if __name__ == "__main__":
    pytest.main([__file__, "-v"])