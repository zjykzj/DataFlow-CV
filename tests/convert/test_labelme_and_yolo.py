"""
Unit tests for labelme_and_yolo.py
"""

import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from dataflow.convert.base import ConversionResult
from dataflow.convert.labelme_and_yolo import LabelMeAndYoloConverter
from dataflow.label.base import AnnotationResult
from dataflow.label.models import (BoundingBox, DatasetAnnotations,
                                   ImageAnnotation, ObjectAnnotation)


class TestLabelMeAndYoloConverter:
    """Test suite for LabelMeAndYoloConverter class."""

    def test_init_labelme_to_yolo(self):
        """Test initialization for LabelMe→YOLO direction."""
        converter = LabelMeAndYoloConverter(source_to_target=True)
        assert converter.source_format == "labelme"
        assert converter.target_format == "yolo"
        assert converter.source_to_target is True

    def test_init_yolo_to_labelme(self):
        """Test initialization for YOLO→LabelMe direction."""
        converter = LabelMeAndYoloConverter(source_to_target=False)
        assert converter.source_format == "yolo"
        assert converter.target_format == "labelme"
        assert converter.source_to_target is False

    def test_validate_inputs_labelme_to_yolo_missing_class_file(self):
        """Test validation for LabelMe→YOLO with missing class file."""
        converter = LabelMeAndYoloConverter(source_to_target=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            # Missing class_file parameter
            assert not converter.validate_inputs(str(source_path), str(target_path), {})

    def test_validate_inputs_labelme_to_yolo_invalid_class_file(self):
        """Test validation for LabelMe→YOLO with non-existent class file."""
        converter = LabelMeAndYoloConverter(source_to_target=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            kwargs = {"class_file": "/nonexistent/classes.txt"}
            assert not converter.validate_inputs(
                str(source_path), str(target_path), kwargs
            )

    def test_validate_inputs_yolo_to_labelme_missing_image_dir(self):
        """Test validation for YOLO→LabelMe with missing image_dir."""
        converter = LabelMeAndYoloConverter(source_to_target=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            # Create a dummy class file
            class_file = Path(tmpdir) / "classes.txt"
            class_file.write_text("cat\ndog\n")

            kwargs = {"class_file": str(class_file)}
            # Missing image_dir parameter
            assert not converter.validate_inputs(
                str(source_path), str(target_path), kwargs
            )

    def test_validate_inputs_yolo_to_labelme_invalid_image_dir(self):
        """Test validation for YOLO→LabelMe with non-existent image_dir."""
        converter = LabelMeAndYoloConverter(source_to_target=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            # Create a dummy class file
            class_file = Path(tmpdir) / "classes.txt"
            class_file.write_text("cat\ndog\n")

            kwargs = {"class_file": str(class_file), "image_dir": "/nonexistent/images"}
            assert not converter.validate_inputs(
                str(source_path), str(target_path), kwargs
            )

    def test_validate_inputs_valid_labelme_to_yolo(self):
        """Test validation for valid LabelMe→YOLO inputs."""
        converter = LabelMeAndYoloConverter(source_to_target=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            # Create a dummy class file
            class_file = Path(tmpdir) / "classes.txt"
            class_file.write_text("cat\ndog\n")

            kwargs = {"class_file": str(class_file)}
            assert converter.validate_inputs(str(source_path), str(target_path), kwargs)

    def test_validate_inputs_valid_yolo_to_labelme(self):
        """Test validation for valid YOLO→LabelMe inputs."""
        converter = LabelMeAndYoloConverter(source_to_target=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            # Create dummy files
            class_file = Path(tmpdir) / "classes.txt"
            class_file.write_text("cat\ndog\n")

            image_dir = Path(tmpdir) / "images"
            image_dir.mkdir()

            kwargs = {"class_file": str(class_file), "image_dir": str(image_dir)}
            assert converter.validate_inputs(str(source_path), str(target_path), kwargs)

    @patch("dataflow.convert.labelme_and_yolo.LabelMeAnnotationHandler")
    def test_create_source_handler_labelme_to_yolo(self, mock_handler_class):
        """Test creating source handler for LabelMe→YOLO."""
        converter = LabelMeAndYoloConverter(source_to_target=True)

        mock_handler = Mock()
        mock_handler_class.return_value = mock_handler

        source_path = "/path/to/source"
        kwargs = {"class_file": "/path/to/classes.txt"}

        handler = converter.create_source_handler(source_path, kwargs)

        # Verify handler was created with correct parameters
        mock_handler_class.assert_called_once_with(
            label_dir=source_path,
            class_file=kwargs["class_file"],
            logger=converter.logger,
        )
        assert handler == mock_handler

    @patch("dataflow.convert.labelme_and_yolo.YoloAnnotationHandler")
    def test_create_source_handler_yolo_to_labelme(self, mock_handler_class):
        """Test creating source handler for YOLO→LabelMe."""
        converter = LabelMeAndYoloConverter(source_to_target=False)

        mock_handler = Mock()
        mock_handler_class.return_value = mock_handler

        source_path = "/path/to/source"
        kwargs = {"class_file": "/path/to/classes.txt", "image_dir": "/path/to/images"}

        handler = converter.create_source_handler(source_path, kwargs)

        # Verify handler was created with correct parameters
        mock_handler_class.assert_called_once_with(
            label_dir=source_path,
            class_file=kwargs["class_file"],
            image_dir=kwargs["image_dir"],
            logger=converter.logger,
        )
        assert handler == mock_handler

    @patch("dataflow.convert.labelme_and_yolo.YoloAnnotationHandler")
    def test_create_target_handler_labelme_to_yolo(self, mock_handler_class):
        """Test creating target handler for LabelMe→YOLO."""
        converter = LabelMeAndYoloConverter(source_to_target=True)

        mock_handler = Mock()
        mock_handler_class.return_value = mock_handler

        target_path = "/path/to/target"
        kwargs = {"class_file": "/path/to/classes.txt"}

        with patch.object(Path, "mkdir") as mock_mkdir:
            handler = converter.create_target_handler(target_path, kwargs)

            # Verify handler was created
            assert handler == mock_handler

            # Verify directories were created
            assert mock_mkdir.call_count >= 3  # base, labels, images directories

    @patch("dataflow.convert.labelme_and_yolo.LabelMeAnnotationHandler")
    def test_create_target_handler_yolo_to_labelme(self, mock_handler_class):
        """Test creating target handler for YOLO→LabelMe."""
        converter = LabelMeAndYoloConverter(source_to_target=False)

        mock_handler = Mock()
        mock_handler_class.return_value = mock_handler

        target_path = "/path/to/target"
        kwargs = {"class_file": "/path/to/classes.txt"}

        with patch.object(Path, "mkdir") as mock_mkdir:
            handler = converter.create_target_handler(target_path, kwargs)

            # Verify handler was created
            mock_handler_class.assert_called_once_with(
                label_dir=target_path,
                class_file=kwargs["class_file"],
                logger=converter.logger,
            )
            assert handler == mock_handler

    def test_convert_annotations_default(self):
        """Test default convert_annotations implementation."""
        converter = LabelMeAndYoloConverter(source_to_target=True)

        # Create dummy annotations
        annotations = DatasetAnnotations(
            images=[
                ImageAnnotation(
                    image_id="test1",
                    image_path="/path/to/image.jpg",
                    width=100,
                    height=100,
                    objects=[],
                )
            ],
            categories={0: "cat", 1: "dog"},
        )

        result = converter.convert_annotations(annotations, {})
        assert result == annotations  # Should return as-is

    @patch("dataflow.convert.labelme_and_yolo.LabelMeAnnotationHandler")
    @patch("dataflow.convert.labelme_and_yolo.YoloAnnotationHandler")
    def test_convert_labelme_to_yolo_mocked(
        self, mock_yolo_handler_class, mock_labelme_handler_class
    ):
        """Test LabelMe→YOLO conversion with mocked handlers."""
        converter = LabelMeAndYoloConverter(source_to_target=True)

        # Mock handlers
        mock_source_handler = Mock()
        mock_target_handler = Mock()
        # Set label_dir attribute for YOLO handler
        mock_target_handler.label_dir = "mock_labels_dir"

        mock_labelme_handler_class.return_value = mock_source_handler
        mock_yolo_handler_class.return_value = mock_target_handler

        # Mock read result
        mock_annotations = Mock(spec=DatasetAnnotations)
        mock_annotations.images = []
        mock_annotations.categories = {}
        mock_read_result = Mock(spec=AnnotationResult)
        mock_read_result.success = True
        mock_read_result.data = mock_annotations
        mock_read_result.errors = []
        mock_source_handler.read.return_value = mock_read_result

        # Mock write result
        mock_write_result = Mock(spec=AnnotationResult)
        mock_write_result.success = True
        mock_write_result.errors = []
        mock_target_handler.write.return_value = mock_write_result

        # Run conversion
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            # Create dummy class file
            class_file = Path(tmpdir) / "classes.txt"
            class_file.write_text("cat\ndog\n")

            kwargs = {"class_file": str(class_file)}
            result = converter.convert(str(source_path), str(target_path), **kwargs)

        # Verify result
        assert result.success is True
        assert result.source_format == "labelme"
        assert result.target_format == "yolo"

        # Verify handlers were called
        mock_source_handler.read.assert_called_once()
        mock_target_handler.write.assert_called_once_with(
            mock_annotations, mock_target_handler.label_dir
        )

    @patch("dataflow.convert.labelme_and_yolo.YoloAnnotationHandler")
    @patch("dataflow.convert.labelme_and_yolo.LabelMeAnnotationHandler")
    def test_convert_yolo_to_labelme_mocked(
        self, mock_labelme_handler_class, mock_yolo_handler_class
    ):
        """Test YOLO→LabelMe conversion with mocked handlers."""
        converter = LabelMeAndYoloConverter(source_to_target=False)

        # Mock handlers
        mock_source_handler = Mock()
        mock_target_handler = Mock()

        mock_yolo_handler_class.return_value = mock_source_handler
        mock_labelme_handler_class.return_value = mock_target_handler

        # Mock read result
        mock_annotations = Mock(spec=DatasetAnnotations)
        mock_annotations.images = []
        mock_annotations.categories = {}
        mock_read_result = Mock(spec=AnnotationResult)
        mock_read_result.success = True
        mock_read_result.data = mock_annotations
        mock_read_result.errors = []
        mock_source_handler.read.return_value = mock_read_result

        # Mock write result
        mock_write_result = Mock(spec=AnnotationResult)
        mock_write_result.success = True
        mock_write_result.errors = []
        mock_target_handler.write.return_value = mock_write_result

        # Run conversion
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            # Create dummy files
            class_file = Path(tmpdir) / "classes.txt"
            class_file.write_text("cat\ndog\n")

            image_dir = Path(tmpdir) / "images"
            image_dir.mkdir()

            kwargs = {"class_file": str(class_file), "image_dir": str(image_dir)}
            result = converter.convert(str(source_path), str(target_path), **kwargs)

        # Verify result
        assert result.success is True
        assert result.source_format == "yolo"
        assert result.target_format == "labelme"

        # Verify handlers were called
        mock_source_handler.read.assert_called_once()
        mock_target_handler.write.assert_called_once_with(
            mock_annotations, str(target_path)
        )

    def test_converter_verbose_param(self):
        """Test converter verbose parameter."""
        # Test verbose=False (default)
        converter_no_verbose = LabelMeAndYoloConverter(
            source_to_target=True, verbose=False
        )
        assert converter_no_verbose.verbose is False
        assert converter_no_verbose.progress_logger is None

        # Test verbose=True
        converter_verbose = LabelMeAndYoloConverter(source_to_target=True, verbose=True)
        assert converter_verbose.verbose is True
        assert hasattr(converter_verbose, "progress_logger")
        assert converter_verbose.progress_logger is not None

        # Test verbose parameter in YOLO→LabelMe direction
        converter_reverse = LabelMeAndYoloConverter(
            source_to_target=False, verbose=True
        )
        assert converter_reverse.verbose is True
        assert converter_reverse.progress_logger is not None

    def test_conversion_result_verbose_log(self):
        """Test ConversionResult verbose logging."""
        from dataflow.convert.base import ConversionResult

        result = ConversionResult(
            success=True,
            source_format="labelme",
            target_format="yolo",
            source_path="/test/source",
            target_path="/test/target",
        )

        # Test add_verbose_log
        test_log = "Test verbose log entry"
        result.add_verbose_log(test_log)
        assert len(result.verbose_log) == 1
        assert test_log in result.verbose_log[0]

        # Test get_verbose_summary when no verbose logs
        summary_no_logs = result.get_verbose_summary()
        assert "Successfully converted" in summary_no_logs

        # Test get_verbose_summary with verbose logs
        result.add_verbose_log("Another log entry")
        summary_with_logs = result.get_verbose_summary()
        assert "Test verbose log entry" in summary_with_logs
        assert "Detailed processing log" in summary_with_logs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
