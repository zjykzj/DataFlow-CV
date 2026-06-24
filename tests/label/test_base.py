"""
Unit tests for base.py
"""

import logging
from pathlib import Path
from typing import Iterator
from unittest.mock import Mock

import pytest

from dataflow.label.base import AnnotationResult, BaseAnnotationHandler
from dataflow.label.models import (BoundingBox, DatasetAnnotations,
                                   ImageAnnotation, ObjectAnnotation)


class ConcreteHandler(BaseAnnotationHandler):
    """Concrete implementation for testing abstract base class."""

    def read(self, *args, **kwargs) -> AnnotationResult:
        return AnnotationResult(success=True, data="test")

    def iter_images(self) -> Iterator[ImageAnnotation]:
        """Test implementation — yields nothing."""
        return iter([])

    def write(
        self, annotations: DatasetAnnotations, *args, **kwargs
    ) -> AnnotationResult:
        return AnnotationResult(success=True)

    def write_one(
        self, image_ann: ImageAnnotation, output_dir: Path
    ) -> AnnotationResult:
        return AnnotationResult(success=True)

    def validate(self, *args, **kwargs) -> bool:
        return True


class TestAnnotationResult:
    """Test suite for AnnotationResult class."""

    def test_init_defaults(self):
        """Test AnnotationResult initialization with defaults."""
        result = AnnotationResult(success=True)
        assert result.success is True
        assert result.data is None
        assert result.message == ""
        assert result.errors == []

    def test_init_with_values(self):
        """Test AnnotationResult initialization with custom values."""
        errors = ["error1", "error2"]
        result = AnnotationResult(
            success=False, data={"key": "value"}, message="Test message", errors=errors
        )
        assert result.success is False
        assert result.data == {"key": "value"}
        assert result.message == "Test message"
        assert result.errors == errors

    def test_add_error(self):
        """Test add_error method."""
        result = AnnotationResult(success=True)
        result.add_error("Something went wrong")
        assert result.errors == ["Something went wrong"]
        assert result.success is False

    def test_add_info(self):
        """Test add_info method."""
        result = AnnotationResult(success=True)
        result.add_info("Operation started")
        assert result.message == "Operation started"

        result.add_info("Operation completed")
        assert result.message == "Operation started; Operation completed"


class TestBaseAnnotationHandler:
    """Test suite for BaseAnnotationHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a concrete handler instance for testing."""
        return ConcreteHandler(strict_mode=True)

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return Mock(spec=logging.Logger)

    def test_init_defaults(self):
        """Test handler initialization with defaults."""
        handler = ConcreteHandler()
        assert handler.strict_mode is True
        assert isinstance(handler.logger, logging.Logger)
        assert handler.is_det is False
        assert handler.is_seg is False
        assert handler.is_rle is False

    def test_init_custom_logger(self, mock_logger):
        """Test handler initialization with custom logger."""
        handler = ConcreteHandler(logger=mock_logger)
        assert handler.logger is mock_logger

    def test_init_strict_mode_false(self):
        """Test handler initialization with strict_mode=False."""
        handler = ConcreteHandler(strict_mode=False)
        assert handler.strict_mode is False

    def test_log_info(self, handler, mock_logger):
        """Test _log_info method."""
        handler.logger = mock_logger
        handler._log_info("Test info message")
        mock_logger.info.assert_called_once_with("Test info message")

    def test_log_warning(self, handler, mock_logger):
        """Test _log_warning method."""
        handler.logger = mock_logger
        handler._log_warning("Test warning message")
        mock_logger.warning.assert_called_once_with("Test warning message")

    def test_log_error_strict_mode(self, handler, mock_logger):
        """Test _log_error method in strict mode raises ValueError."""
        handler.logger = mock_logger
        handler.strict_mode = True

        with pytest.raises(ValueError, match="Test error message"):
            handler._log_error("Test error message")
        mock_logger.error.assert_called_once_with("Test error message")

    def test_log_error_non_strict_mode(self, handler, mock_logger):
        """Test _log_error method in non-strict mode doesn't raise."""
        handler.logger = mock_logger
        handler.strict_mode = False

        # Should not raise
        handler._log_error("Test error message")
        mock_logger.error.assert_called_once_with("Test error message")

    def test_set_annotation_flags_detection(self, handler):
        """Test _set_annotation_flags for detection annotations."""
        dataset = DatasetAnnotations()
        image = ImageAnnotation(
            image_id="test.jpg",
            image_path="/path/test.jpg",
            width=100,
            height=100,
            objects=[
                ObjectAnnotation(
                    class_id=0,
                    class_name="cat",
                    bbox=BoundingBox(x=0.5, y=0.5, width=0.2, height=0.2),
                )
            ],
        )
        dataset.add_image(image)

        handler._set_annotation_flags(dataset)
        assert handler.is_det is True
        assert handler.is_seg is False

    def test_set_annotation_flags_segmentation(self, handler):
        """Test _set_annotation_flags for segmentation annotations."""
        from dataflow.label.models import Segmentation

        dataset = DatasetAnnotations()
        image = ImageAnnotation(
            image_id="test.jpg",
            image_path="/path/test.jpg",
            width=100,
            height=100,
            objects=[
                ObjectAnnotation(
                    class_id=0,
                    class_name="cat",
                    segmentation=Segmentation(
                        points=[(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)]
                    ),
                )
            ],
        )
        dataset.add_image(image)

        handler._set_annotation_flags(dataset)
        assert handler.is_det is False
        assert handler.is_seg is True

    def test_set_annotation_flags_mixed(self, handler):
        """Test _set_annotation_flags for mixed annotations."""
        from dataflow.label.models import Segmentation

        dataset = DatasetAnnotations()
        image = ImageAnnotation(
            image_id="test.jpg",
            image_path="/path/test.jpg",
            width=100,
            height=100,
            objects=[
                ObjectAnnotation(
                    class_id=0,
                    class_name="cat",
                    bbox=BoundingBox(x=0.5, y=0.5, width=0.2, height=0.2),
                    segmentation=Segmentation(
                        points=[(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)]
                    ),
                )
            ],
        )
        dataset.add_image(image)

        handler._set_annotation_flags(dataset)
        assert handler.is_det is True
        assert handler.is_seg is True

    def test_set_annotation_flags_empty(self, handler):
        """Test _set_annotation_flags for empty dataset."""
        dataset = DatasetAnnotations()
        handler._set_annotation_flags(dataset)
        assert handler.is_det is False
        assert handler.is_seg is False

    def test_validate_image_dimensions_valid(self, handler):
        """Test _validate_image_dimensions with valid dimensions."""
        assert handler._validate_image_dimensions(100, 100) is True

    def test_validate_image_dimensions_invalid(self, handler, mock_logger):
        """Test _validate_image_dimensions with invalid dimensions."""
        handler.logger = mock_logger
        handler.strict_mode = False
        assert handler._validate_image_dimensions(0, 100) is False
        mock_logger.error.assert_called_once()

    def test_validate_normalized_coordinate_valid(self, handler):
        """Test _validate_normalized_coordinate with valid values."""
        assert handler._validate_normalized_coordinate(0.5, "test") is True
        assert handler._validate_normalized_coordinate(0.0, "test") is True
        assert handler._validate_normalized_coordinate(1.0, "test") is True

    def test_validate_normalized_coordinate_invalid(self, handler, mock_logger):
        """Test _validate_normalized_coordinate with invalid values."""
        handler.logger = mock_logger
        handler.strict_mode = False
        assert handler._validate_normalized_coordinate(-0.1, "test") is False
        assert handler._validate_normalized_coordinate(1.1, "test") is False
        assert mock_logger.error.call_count == 2

    def test_validate_bbox_valid(self, handler):
        """Test _validate_bbox with valid bbox."""
        bbox = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.2)
        assert handler._validate_bbox(bbox) is True

    def test_validate_bbox_invalid(self, handler, mock_logger):
        """Test _validate_bbox with invalid bbox."""
        handler.logger = mock_logger
        handler.strict_mode = False
        bbox = BoundingBox(x=1.5, y=0.5, width=0.2, height=0.2)  # x out of range
        assert handler._validate_bbox(bbox) is False
        mock_logger.error.assert_called_once()

    def test_validate_bbox_none(self, handler):
        """Test _validate_bbox with None."""
        assert handler._validate_bbox(None) is True

    def test_validate_segmentation_points_valid(self, handler):
        """Test _validate_segmentation_points with valid points."""
        points = [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)]
        assert handler._validate_segmentation_points(points) is True

    def test_validate_segmentation_points_invalid(self, handler, mock_logger):
        """Test _validate_segmentation_points with invalid points."""
        handler.logger = mock_logger
        handler.strict_mode = False

        # Empty points
        assert handler._validate_segmentation_points([]) is False

        # Out of range
        points = [(1.5, 0.1), (0.2, 0.1), (0.2, 0.2)]
        assert handler._validate_segmentation_points(points) is False

        # Too few points
        points = [(0.1, 0.1), (0.2, 0.1)]
        assert handler._validate_segmentation_points(points) is False

    # ── _clamp_normalized_bbox ──────────────────────────────────────

    def test_clamp_normalized_bbox_no_change(self, handler):
        """Valid bbox passes through unchanged."""
        cx, cy, w, h = handler._clamp_normalized_bbox(0.5, 0.5, 0.2, 0.2)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)
        assert w == pytest.approx(0.2)
        assert h == pytest.approx(0.2)

    def test_clamp_normalized_bbox_right_overflow(self, handler):
        """Right edge overflow is clamped to 1.0."""
        # cx=0.979211, w=0.0415783 → right edge = 1.00000015
        cx, cy, w, h = handler._clamp_normalized_bbox(
            0.979211, 0.5, 0.0415783, 0.2
        )
        # Right edge clamped from 1.00000015 to 1.0
        assert cx == pytest.approx(0.979210925)
        assert w == pytest.approx(0.04157815)

    def test_clamp_normalized_bbox_left_overflow(self, handler):
        """Left edge underflow is clamped to 0.0."""
        # cx=0.02, w=0.05 → left = -0.005, clamped to 0
        # After clamp: left=0, right=0.045 → cx=0.0225, w=0.045
        cx, cy, w, h = handler._clamp_normalized_bbox(0.02, 0.5, 0.05, 0.2)
        assert cx == pytest.approx(0.0225)
        assert w == pytest.approx(0.045)
        assert cy == pytest.approx(0.5)
        assert h == pytest.approx(0.2)

    def test_clamp_normalized_bbox_top_overflow(self, handler):
        """Top edge underflow is clamped to 0.0."""
        # cy=0.01, h=0.04 → top = -0.01, clamped to 0
        # After clamp: top=0, bottom=0.03 → cy=0.015, h=0.03
        cx, cy, w, h = handler._clamp_normalized_bbox(0.5, 0.01, 0.2, 0.04)
        assert cy == pytest.approx(0.015)
        assert h == pytest.approx(0.03)

    def test_clamp_normalized_bbox_bottom_overflow(self, handler):
        """Bottom edge overflow is clamped to 1.0."""
        # cy=0.99, h=0.04 → bottom = 1.01, clamped to 1.0
        # After clamp: top=0.97, bottom=1.0 → cy=0.985, h=0.03
        cx, cy, w, h = handler._clamp_normalized_bbox(0.5, 0.99, 0.2, 0.04)
        assert cy == pytest.approx(0.985)
        assert h == pytest.approx(0.03)

    def test_clamp_normalized_bbox_both_overflow(self, handler):
        """Both edges overflow — both clamped to 1.0, w→0."""
        cx, cy, w, h = handler._clamp_normalized_bbox(1.5, 0.5, 0.2, 0.2)
        # left=1.4→1.0, right=1.6→1.0
        assert cx == pytest.approx(1.0)
        assert w == pytest.approx(0.0)

    def test_clamp_normalized_bbox_warning_logged(self, handler, mock_logger):
        """Warning is logged when clamping modifies values."""
        handler.logger = mock_logger
        handler._clamp_normalized_bbox(1.5, 0.5, 0.2, 0.2)
        mock_logger.warning.assert_called_once()
        assert "Clamped normalized bbox to [0, 1]" in (
            mock_logger.warning.call_args[0][0]
        )

    def test_clamp_normalized_bbox_no_warning_when_unchanged(
        self, handler, mock_logger
    ):
        """No warning logged when values are unchanged."""
        handler.logger = mock_logger
        handler._clamp_normalized_bbox(0.5, 0.5, 0.2, 0.2)
        mock_logger.warning.assert_not_called()

    def test_clamp_normalized_bbox_sub_threshold_silent(
        self, handler, mock_logger
    ):
        """Sub-threshold (<1e-6) edge overflow is silently clamped."""
        handler.logger = mock_logger
        # cx=0.5, w=1.0000001 → right edge = 1.00000005
        # → clamped to 1.0, change is ~5e-8 < 1e-6 → silent
        cx, cy, w, h = handler._clamp_normalized_bbox(
            0.5, 0.5, 1.0000001, 0.2
        )
        assert w == pytest.approx(1.0)
        mock_logger.warning.assert_not_called()

    def test_clamp_normalized_bbox_boundary_values(self, handler):
        """Values exactly at [0,1] boundary pass through unchanged."""
        # cx=0.0, w=0.0 → both edges at 0 (degenerate but not OOB)
        cx, cy, w, h = handler._clamp_normalized_bbox(0.0, 1.0, 0.0, 0.0)
        assert cx == pytest.approx(0.0)
        assert cy == pytest.approx(1.0)
        assert w == pytest.approx(0.0)
        assert h == pytest.approx(0.0)

    # ── _clamp_normalized_points ────────────────────────────────────

    def test_clamp_normalized_points_no_change(self, handler):
        """Valid points pass through unchanged."""
        points = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]
        result = handler._clamp_normalized_points(points)
        assert result == points

    def test_clamp_normalized_points_overflow(self, handler):
        """Points with sub-threshold FP imprecision get silently clamped."""
        points = [(1.0000001, -0.0000001), (0.5, 0.5)]
        result = handler._clamp_normalized_points(points)
        assert result == [(1.0, 0.0), (0.5, 0.5)]

    def test_clamp_normalized_points_warning(self, handler, mock_logger):
        """Warning is logged when point change exceeds 1e-6 threshold."""
        handler.logger = mock_logger
        # 1.00001 → change of 1e-5 > 1e-6
        handler._clamp_normalized_points([(1.00001, -0.00001)])
        mock_logger.warning.assert_called_once()
        assert "Clamped normalized polygon points" in (
            mock_logger.warning.call_args[0][0]
        )

    def test_clamp_normalized_points_sub_threshold_silent(
        self, handler, mock_logger
    ):
        """Sub-threshold (<5e-7) point overflow is silently clamped."""
        handler.logger = mock_logger
        # Change of 1e-7 < 5e-7 → silent
        handler._clamp_normalized_points([(1.0000001, -0.0000001)])
        mock_logger.warning.assert_not_called()

    def test_clamp_normalized_points_no_warning_when_unchanged(
        self, handler, mock_logger
    ):
        """No warning when points unchanged."""
        handler.logger = mock_logger
        handler._clamp_normalized_points([(0.1, 0.1), (0.5, 0.5)])
        mock_logger.warning.assert_not_called()

    def test_clamp_normalized_points_empty(self, handler):
        """Empty list returns empty list."""
        result = handler._clamp_normalized_points([])
        assert result == []

    def test_abstract_methods(self):
        """Test that abstract methods cannot be called on abstract base class."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            handler = BaseAnnotationHandler()
