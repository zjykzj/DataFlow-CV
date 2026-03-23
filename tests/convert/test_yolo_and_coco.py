"""
Unit tests for yolo_and_coco.py
"""

import pytest
import logging
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from dataflow.convert.yolo_and_coco import YoloAndCocoConverter
from dataflow.convert.base import ConversionResult
from dataflow.label.base import AnnotationResult
from dataflow.label.models import DatasetAnnotations, ImageAnnotation, ObjectAnnotation, BoundingBox


class TestYoloAndCocoConverter:
    """Test suite for YoloAndCocoConverter class."""

    def test_init_yolo_to_coco(self):
        """Test initialization for YOLO→COCO direction."""
        converter = YoloAndCocoConverter(source_to_target=True)
        assert converter.source_format == "yolo"
        assert converter.target_format == "coco"
        assert converter.source_to_target is True

    def test_init_coco_to_yolo(self):
        """Test initialization for COCO→YOLO direction."""
        converter = YoloAndCocoConverter(source_to_target=False)
        assert converter.source_format == "coco"
        assert converter.target_format == "yolo"
        assert converter.source_to_target is False

    def test_validate_inputs_yolo_to_coco_missing_class_file(self):
        """Test validation for YOLO→COCO with missing class file."""
        converter = YoloAndCocoConverter(source_to_target=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            # Missing class_file parameter
            kwargs = {"image_dir": str(Path(tmpdir) / "images")}
            assert not converter.validate_inputs(str(source_path), str(target_path), kwargs)

    def test_validate_inputs_yolo_to_coco_invalid_class_file(self):
        """Test validation for YOLO→COCO with non-existent class file."""
        converter = YoloAndCocoConverter(source_to_target=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            kwargs = {
                "class_file": "/nonexistent/classes.txt",
                "image_dir": str(Path(tmpdir) / "images")
            }
            assert not converter.validate_inputs(str(source_path), str(target_path), kwargs)

    def test_validate_inputs_yolo_to_coco_missing_image_dir(self):
        """Test validation for YOLO→COCO with missing image_dir."""
        converter = YoloAndCocoConverter(source_to_target=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            # Create a dummy class file
            class_file = Path(tmpdir) / "classes.txt"
            class_file.write_text("cat\ndog\n")

            kwargs = {"class_file": str(class_file)}
            # Missing image_dir parameter
            assert not converter.validate_inputs(str(source_path), str(target_path), kwargs)

    def test_validate_inputs_yolo_to_coco_invalid_image_dir(self):
        """Test validation for YOLO→COCO with non-existent image_dir."""
        converter = YoloAndCocoConverter(source_to_target=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            # Create a dummy class file
            class_file = Path(tmpdir) / "classes.txt"
            class_file.write_text("cat\ndog\n")

            kwargs = {
                "class_file": str(class_file),
                "image_dir": "/nonexistent/images"
            }
            assert not converter.validate_inputs(str(source_path), str(target_path), kwargs)

    def test_validate_inputs_yolo_to_coco_valid(self):
        """Test validation for valid YOLO→COCO inputs."""
        converter = YoloAndCocoConverter(source_to_target=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            # Create dummy files
            class_file = Path(tmpdir) / "classes.txt"
            class_file.write_text("cat\ndog\n")

            image_dir = Path(tmpdir) / "images"
            image_dir.mkdir()

            kwargs = {
                "class_file": str(class_file),
                "image_dir": str(image_dir)
            }
            assert converter.validate_inputs(str(source_path), str(target_path), kwargs)

    def test_validate_inputs_coco_to_yolo_valid(self):
        """Test validation for valid COCO→YOLO inputs."""
        converter = YoloAndCocoConverter(source_to_target=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "coco.json"
            source_path.write_text("{}")  # Empty JSON
            target_path = Path(tmpdir) / "target"

            # class_file and image_dir are optional for COCO→YOLO
            kwargs = {}
            assert converter.validate_inputs(str(source_path), str(target_path), kwargs)

    @patch('dataflow.convert.yolo_and_coco.YoloAnnotationHandler')
    def test_create_source_handler_yolo_to_coco(self, mock_handler_class):
        """Test creating source handler for YOLO→COCO."""
        converter = YoloAndCocoConverter(source_to_target=True)

        mock_handler = Mock()
        mock_handler_class.return_value = mock_handler

        source_path = "/path/to/source"
        kwargs = {
            "class_file": "/path/to/classes.txt",
            "image_dir": "/path/to/images"
        }

        handler = converter.create_source_handler(source_path, kwargs)

        # Verify handler was created with correct parameters
        mock_handler_class.assert_called_once_with(
            label_dir=source_path,
            class_file=kwargs["class_file"],
            image_dir=kwargs["image_dir"],
            logger=converter.logger
        )
        assert handler == mock_handler

    @patch('dataflow.convert.yolo_and_coco.CocoAnnotationHandler')
    def test_create_source_handler_coco_to_yolo(self, mock_handler_class):
        """Test creating source handler for COCO→YOLO."""
        converter = YoloAndCocoConverter(source_to_target=False)

        mock_handler = Mock()
        mock_handler_class.return_value = mock_handler

        source_path = "/path/to/coco.json"
        kwargs = {}

        handler = converter.create_source_handler(source_path, kwargs)

        # Verify handler was created with correct parameters
        mock_handler_class.assert_called_once_with(
            annotation_file=source_path,
            logger=converter.logger
        )
        assert handler == mock_handler

    @patch('dataflow.convert.yolo_and_coco.CocoAnnotationHandler')
    def test_create_target_handler_yolo_to_coco(self, mock_handler_class):
        """Test creating target handler for YOLO→COCO."""
        converter = YoloAndCocoConverter(source_to_target=True)

        mock_handler = Mock()
        mock_handler_class.return_value = mock_handler

        target_path = "/path/to/coco.json"
        kwargs = {"do_rle": False}

        with patch.object(Path, 'mkdir') as mock_mkdir:
            handler = converter.create_target_handler(target_path, kwargs)

            # Verify handler was created
            mock_handler_class.assert_called_once_with(
                annotation_file=target_path,
                logger=converter.logger,
                do_rle=False
            )
            assert handler == mock_handler

            # Verify mkdir was called
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('dataflow.convert.yolo_and_coco.YoloAnnotationHandler')
    def test_create_target_handler_coco_to_yolo(self, mock_handler_class):
        """Test creating target handler for COCO→YOLO."""
        converter = YoloAndCocoConverter(source_to_target=False)

        mock_handler = Mock()
        mock_handler_class.return_value = mock_handler

        target_path = "/path/to/target"
        kwargs = {}

        with patch.object(Path, 'mkdir') as mock_mkdir:
            handler = converter.create_target_handler(target_path, kwargs)

            # Verify handler was created
            assert handler == mock_handler

            # Verify directories were created
            assert mock_mkdir.call_count >= 3  # base, labels, images directories

    def test_convert_annotations_default(self):
        """Test default convert_annotations implementation."""
        converter = YoloAndCocoConverter(source_to_target=True)

        # Create dummy annotations
        annotations = DatasetAnnotations(
            images=[
                ImageAnnotation(
                    image_id="test1",
                    image_path="/path/to/image.jpg",
                    width=100,
                    height=100,
                    objects=[]
                )
            ],
            categories={0: "cat", 1: "dog"}
        )

        result = converter.convert_annotations(annotations, {})
        assert result == annotations  # Should return as-is

    @patch('dataflow.convert.yolo_and_coco.YoloAnnotationHandler')
    @patch('dataflow.convert.yolo_and_coco.CocoAnnotationHandler')
    def test_convert_yolo_to_coco_mocked(self, mock_coco_handler_class, mock_yolo_handler_class):
        """Test YOLO→COCO conversion with mocked handlers."""
        converter = YoloAndCocoConverter(source_to_target=True)

        # Mock handlers
        mock_source_handler = Mock()
        mock_target_handler = Mock()

        mock_yolo_handler_class.return_value = mock_source_handler
        mock_coco_handler_class.return_value = mock_target_handler

        # Mock read result
        mock_annotations = Mock(spec=DatasetAnnotations)
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
            target_path = Path(tmpdir) / "coco.json"

            # Create dummy files
            class_file = Path(tmpdir) / "classes.txt"
            class_file.write_text("cat\ndog\n")

            image_dir = Path(tmpdir) / "images"
            image_dir.mkdir()

            kwargs = {
                "class_file": str(class_file),
                "image_dir": str(image_dir),
                "do_rle": False
            }
            result = converter.convert(str(source_path), str(target_path), **kwargs)

        # Verify result
        assert result.success is True
        assert result.source_format == "yolo"
        assert result.target_format == "coco"

        # Verify handlers were called
        mock_source_handler.read.assert_called_once()
        mock_target_handler.write.assert_called_once_with(mock_annotations, str(target_path))

    @patch('dataflow.convert.yolo_and_coco.CocoAnnotationHandler')
    @patch('dataflow.convert.yolo_and_coco.YoloAnnotationHandler')
    def test_convert_coco_to_yolo_mocked(self, mock_yolo_handler_class, mock_coco_handler_class):
        """Test COCO→YOLO conversion with mocked handlers."""
        converter = YoloAndCocoConverter(source_to_target=False)

        # Mock handlers
        mock_source_handler = Mock()
        mock_target_handler = Mock()
        # Set label_dir attribute for YOLO handler
        mock_target_handler.label_dir = "mock_labels_dir"

        mock_coco_handler_class.return_value = mock_source_handler
        mock_yolo_handler_class.return_value = mock_target_handler

        # Mock read result
        mock_annotations = Mock(spec=DatasetAnnotations)
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
            source_path = Path(tmpdir) / "coco.json"
            source_path.write_text("{}")  # Empty JSON
            target_path = Path(tmpdir) / "target"

            kwargs = {}
            result = converter.convert(str(source_path), str(target_path), **kwargs)

        # Verify result
        assert result.success is True
        assert result.source_format == "coco"
        assert result.target_format == "yolo"

        # Verify handlers were called
        mock_source_handler.read.assert_called_once()
        mock_target_handler.write.assert_called_once_with(mock_annotations, mock_target_handler.label_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])