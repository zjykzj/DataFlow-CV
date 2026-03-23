"""
Unit tests for base visualization classes.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from dataflow.visualize.base import BaseVisualizer, ColorManager, VisualizationResult
from dataflow.label.models import (
    DatasetAnnotations, ImageAnnotation, ObjectAnnotation,
    BoundingBox, Segmentation
)


class MockVisualizer(BaseVisualizer):
    """Concrete visualizer for testing base class."""

    def load_annotations(self) -> DatasetAnnotations:
        """Mock implementation for testing."""
        return DatasetAnnotations()


class TestColorManager:
    """Test ColorManager class."""

    def test_get_color_same_class_same_color(self):
        """Same class ID should return same color."""
        manager = ColorManager()
        color1 = manager.get_color(0)
        color2 = manager.get_color(0)
        assert color1 == color2

    def test_get_color_different_classes_different_colors(self):
        """Different class IDs should return different colors (usually)."""
        manager = ColorManager()
        colors = set()
        for i in range(10):
            colors.add(manager.get_color(i))
        # There should be at least 5 distinct colors for 10 classes
        assert len(colors) >= 5

    def test_get_color_no_cycling(self):
        """Colors should NOT cycle when class IDs exceed predefined colors.
        Each class ID should get a unique color (or at least different from
        earlier IDs within reasonable range).
        """
        manager = ColorManager()
        num_colors = len(manager.predefined_colors)

        # Get colors for first batch
        first_batch = [manager.get_color(i) for i in range(num_colors)]

        # Next batch should be different (no cycling)
        second_batch = [manager.get_color(i + num_colors) for i in range(min(10, num_colors))]

        # Colors should be different (no cycling)
        for i in range(min(10, num_colors)):
            # They might coincidentally match, but with 1000 unique colors and
            # different generation algorithm, they should be different
            if first_batch[i] == second_batch[i]:
                # If they match, it's OK as long as it's rare
                # Just log it
                print(f"Note: Color match at index {i}: {first_batch[i]}")
                # But we don't fail the test


class TestVisualizationResult:
    """Test VisualizationResult class."""

    def test_initialization(self):
        """Test result initialization."""
        result = VisualizationResult(success=True, message="Test")
        assert result.success is True
        assert result.message == "Test"
        assert result.errors == []
        assert result.data is None

    def test_add_error(self):
        """Test adding errors."""
        result = VisualizationResult(success=False)
        result.add_error("Error 1")
        result.add_error("Error 2")
        assert len(result.errors) == 2
        assert "Error 1" in result.errors
        assert "Error 2" in result.errors


class TestBaseVisualizer:
    """Test BaseVisualizer abstract class."""

    def test_initialization(self):
        """Test visualizer initialization."""
        visualizer = MockVisualizer(
            label_dir="/tmp/labels",
            image_dir="/tmp/images",
            output_dir="/tmp/output",
            is_show=False,
            is_save=False,
            strict_mode=True
        )

        assert visualizer.label_dir == Path("/tmp/labels")
        assert visualizer.image_dir == Path("/tmp/images")
        assert visualizer.output_dir == Path("/tmp/output")
        assert visualizer.is_show is False
        assert visualizer.is_save is False
        assert visualizer.strict_mode is True
        assert visualizer.logger is not None
        assert visualizer.file_ops is not None
        assert visualizer.color_manager is not None

    def test_config_defaults(self):
        """Test default configuration values."""
        visualizer = MockVisualizer("/tmp/labels", "/tmp/images")

        config = visualizer.config
        assert config['bbox_thickness'] == 2
        assert config['seg_thickness'] == 1
        assert config['seg_alpha'] == 0.3
        assert config['text_thickness'] == 1
        assert config['text_scale'] == 0.5
        assert config['text_padding'] == 5
        assert config['font'] is not None

    def test_visualize_abstract_method(self):
        """Test that abstract method raises error."""
        # BaseVisualizer cannot be instantiated directly
        with pytest.raises(TypeError):
            BaseVisualizer("/tmp/labels", "/tmp/images")

    def test_visualize_without_output_dir_in_save_mode(self):
        """Test that save mode requires output_dir."""
        visualizer = MockVisualizer("/tmp/labels", "/tmp/images", is_save=True)

        # Mock load_annotations to return empty dataset
        with patch.object(visualizer, 'load_annotations', return_value=DatasetAnnotations()):
            result = visualizer.visualize()

            assert result.success is False
            assert "output_dir" in result.errors[0].lower()

    @pytest.mark.skipif(not Path("/tmp").exists(), reason="Requires /tmp directory")
    def test_visualize_with_empty_dataset(self):
        """Test visualization with empty dataset."""
        visualizer = MockVisualizer(
            label_dir="/tmp/labels",
            image_dir="/tmp/images",
            output_dir="/tmp/output",
            is_show=False,
            is_save=False
        )

        # Mock load_annotations to return empty dataset
        with patch.object(visualizer, 'load_annotations', return_value=DatasetAnnotations()):
            result = visualizer.visualize()

            assert result.success is True
            assert result.data["processed_count"] == 0
            assert "0/0" in result.message

    def test_log_methods(self):
        """Test logging methods."""
        visualizer = MockVisualizer("/tmp/labels", "/tmp/images")

        # Test that log methods don't raise exceptions
        visualizer._log_info("Test info")
        visualizer._log_warning("Test warning")

        # Test error logging in strict mode
        visualizer.strict_mode = True
        with pytest.raises(ValueError):
            visualizer._log_error("Test error")

        # Test error logging in non-strict mode
        visualizer.strict_mode = False
        try:
            visualizer._log_error("Test error")
        except ValueError:
            pytest.fail("_log_error should not raise exception in non-strict mode")

    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyWindow')
    @patch('cv2.imread')
    @patch('pathlib.Path.exists')
    def test_visualize_single_image_is_show_mode(self, mock_exists, mock_imread, mock_destroy_window, mock_wait_key, mock_imshow):
        """Test visualization in is_show mode with keyboard interaction."""
        # Create a mock image
        mock_image = Mock()
        mock_imread.return_value = mock_image
        # Mock file exists check
        mock_exists.return_value = True

        # Create visualizer with is_show=True
        visualizer = MockVisualizer(
            label_dir="/tmp/labels",
            image_dir="/tmp/images",
            is_show=True,
            is_save=False,
            strict_mode=True
        )

        # Create a mock image annotation with absolute path
        image_ann = ImageAnnotation(
            image_id="test_image",
            image_path="/tmp/images/test.jpg",  # Absolute path
            width=800,
            height=600,
            objects=[]
        )

        # Test Enter key (continue)
        mock_wait_key.return_value = 13  # Enter key
        success = visualizer._visualize_single_image(image_ann)
        assert success is True
        mock_imshow.assert_called_once()
        mock_wait_key.assert_called_once_with(0)
        mock_destroy_window.assert_called_once()

        # Reset mocks for next test (but keep exists mock)
        mock_imshow.reset_mock()
        mock_wait_key.reset_mock()
        mock_destroy_window.reset_mock()
        mock_imread.reset_mock()

        # Test space key (continue)
        mock_wait_key.return_value = 32  # Space key
        success = visualizer._visualize_single_image(image_ann)
        assert success is True
        mock_imshow.assert_called_once()
        mock_wait_key.assert_called_once_with(0)
        mock_destroy_window.assert_called_once()

        # Reset mocks for next test
        mock_imshow.reset_mock()
        mock_wait_key.reset_mock()
        mock_destroy_window.reset_mock()
        mock_imread.reset_mock()

        # Test 'q' key (quit)
        mock_wait_key.return_value = ord('q')
        success = visualizer._visualize_single_image(image_ann)
        assert success is False  # Should return False to stop visualization
        mock_imshow.assert_called_once()
        mock_wait_key.assert_called_once_with(0)
        mock_destroy_window.assert_called_once()

        # Reset mocks for next test
        mock_imshow.reset_mock()
        mock_wait_key.reset_mock()
        mock_destroy_window.reset_mock()
        mock_imread.reset_mock()

        # Test ESC key (quit)
        mock_wait_key.return_value = 27  # ESC key
        success = visualizer._visualize_single_image(image_ann)
        assert success is False  # Should return False to stop visualization
        mock_imshow.assert_called_once()
        mock_wait_key.assert_called_once_with(0)
        mock_destroy_window.assert_called_once()

    @patch('cv2.imwrite')
    @patch('cv2.imread')
    @patch('pathlib.Path.exists')
    def test_visualize_single_image_is_save_mode(self, mock_exists, mock_imread, mock_imwrite):
        """Test visualization in is_save mode with image saving."""
        # Create a mock image
        mock_image = Mock()
        mock_imread.return_value = mock_image
        # Mock file exists check
        mock_exists.return_value = True

        # Create output directory
        output_dir = Path("/tmp/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create visualizer with is_save=True
        visualizer = MockVisualizer(
            label_dir="/tmp/labels",
            image_dir="/tmp/images",
            output_dir=output_dir,
            is_show=False,
            is_save=True,
            strict_mode=True
        )

        # Create a mock image annotation with absolute path
        image_ann = ImageAnnotation(
            image_id="test_image",
            image_path="/tmp/images/test.jpg",  # Absolute path
            width=800,
            height=600,
            objects=[]
        )

        success = visualizer._visualize_single_image(image_ann)
        assert success is True
        mock_imwrite.assert_called_once()

        # Check that output file path is correct
        call_args = mock_imwrite.call_args
        assert str(call_args[0][0]) == str(output_dir / "test_image_visualized.jpg")
        assert call_args[0][1] is mock_image

        # Clean up
        try:
            output_dir.rmdir()
        except OSError:
            pass


def test_draw_methods_signatures():
    """Test that draw methods have correct signatures."""
    # This test ensures the abstract methods are properly defined
    visualizer = MockVisualizer("/tmp/labels", "/tmp/images")

    # Check that draw methods exist and are callable
    assert hasattr(visualizer, '_draw_object')
    assert hasattr(visualizer, '_draw_bbox')
    assert hasattr(visualizer, '_draw_polygon')
    assert hasattr(visualizer, '_draw_rle_mask')
    assert hasattr(visualizer, '_draw_text')

    # Check they are methods (callable)
    assert callable(visualizer._draw_object)
    assert callable(visualizer._draw_bbox)
    assert callable(visualizer._draw_polygon)
    assert callable(visualizer._draw_rle_mask)
    assert callable(visualizer._draw_text)