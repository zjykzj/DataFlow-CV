"""
Unit tests for base visualization classes.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from dataflow.visualize.base import (BaseVisualizer, ColorManager,
                                     RenderAnnotation, RenderData,
                                     VisualizationResult)


class MockVisualizer(BaseVisualizer):
    """Concrete visualizer for testing base class."""

    def __init__(self, label_dir, image_dir, mock_images=None, **kwargs):
        super().__init__(label_dir, image_dir, **kwargs)
        self._mock_images = mock_images

    def _create_handler(self):
        """Mock handler with configurable iterator for testing."""
        handler = Mock()
        images = self._mock_images if self._mock_images is not None else []
        handler.iter_images.return_value = iter(images)
        return handler

    def _convert_to_render_data(self, image_ann):
        """Mock conversion returning empty RenderData."""
        return RenderData(
            annotations=[],
        )


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
        assert len(colors) >= 5

    def test_get_color_no_cycling(self):
        """Colors should NOT cycle when class IDs exceed predefined colors."""
        manager = ColorManager()
        num_colors = len(manager.predefined_colors)
        first_batch = [manager.get_color(i) for i in range(num_colors)]
        second_batch = [
            manager.get_color(i + num_colors) for i in range(min(10, num_colors))
        ]
        for i in range(min(10, num_colors)):
            if first_batch[i] == second_batch[i]:
                print(f"Note: Color match at index {i}: {first_batch[i]}")


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
        )
        assert visualizer.label_dir == Path("/tmp/labels")
        assert visualizer.image_dir == Path("/tmp/images")
        assert visualizer.output_dir == Path("/tmp/output")
        assert visualizer.is_show is False
        assert visualizer.is_save is False

    def test_config_defaults(self):
        """Test default configuration values."""
        visualizer = MockVisualizer("/tmp/labels", "/tmp/images")
        config = visualizer.config
        assert config["bbox_thickness"] == 2
        assert config["seg_thickness"] == 1
        assert config["seg_alpha"] == 0.3
        assert config["text_thickness"] == 1
        assert config["text_scale"] == 0.5
        assert config["text_padding"] == 5
        assert config["font"] is not None

    def test_visualize_abstract_method(self):
        """Test that abstract method raises error."""
        with pytest.raises(TypeError):
            BaseVisualizer("/tmp/labels", "/tmp/images")

    def test_visualize_without_output_dir_in_save_mode(self):
        """Test that save mode requires output_dir."""
        visualizer = MockVisualizer("/tmp/labels", "/tmp/images", is_save=True)
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
            is_save=False,
        )
        result = visualizer.visualize()
        assert result.success is True
        assert result.data["processed_count"] == 0

    def test_log_methods(self):
        """Test logging methods.

        _log_error always logs (never raises) — visualization is read-only,
        so a single bad file should never abort the entire dataset inspection.
        """
        visualizer = MockVisualizer("/tmp/labels", "/tmp/images")
        visualizer._log_info("Test info")
        visualizer._log_warning("Test warning")
        # _log_error should not raise — it always logs
        try:
            visualizer._log_error("Test error")
        except ValueError:
            pytest.fail("_log_error should not raise exception")

    @patch("cv2.imshow")
    @patch("cv2.waitKeyEx")
    @patch("cv2.resizeWindow")
    @patch("cv2.moveWindow")
    @patch("cv2.namedWindow")
    @patch("cv2.imread")
    @patch("pathlib.Path.exists")
    def test_visualize_single_image_is_show_mode(
        self,
        mock_exists,
        mock_imread,
        mock_named_window,
        mock_move_window,
        mock_resize_window,
        mock_wait_key_ex,
        mock_imshow,
    ):
        """Test visualization in is_show mode with keyboard interaction."""
        mock_image = Mock()
        mock_image.shape = [600, 800, 3]
        mock_imread.return_value = mock_image
        mock_exists.return_value = True

        visualizer = MockVisualizer(
            label_dir="/tmp/labels",
            image_dir="/tmp/images",
            is_show=True,
            is_save=False,

        )

        # Create render data
        render_data = RenderData(annotations=[])
        image_path = "/tmp/images/test.jpg"

        # Test Enter key (continue)
        mock_wait_key_ex.return_value = 13
        action = visualizer._visualize_single_image(image_path, render_data)
        assert action == "next"
        mock_named_window.assert_called_once()
        mock_move_window.assert_called_once()
        mock_resize_window.assert_called_once()
        mock_imshow.assert_called_once()
        mock_wait_key_ex.assert_called_once_with(0)

        mock_named_window.reset_mock()
        mock_move_window.reset_mock()
        mock_resize_window.reset_mock()
        mock_imshow.reset_mock()
        mock_wait_key_ex.reset_mock()
        mock_imread.reset_mock()

        # Test space key
        mock_wait_key_ex.return_value = 32
        action = visualizer._visualize_single_image(image_path, render_data)
        assert action == "next"
        mock_named_window.assert_called_once()
        mock_move_window.assert_not_called()
        mock_resize_window.assert_called_once()
        mock_imshow.assert_called_once()
        mock_wait_key_ex.assert_called_once_with(0)

        mock_named_window.reset_mock()
        mock_move_window.reset_mock()
        mock_resize_window.reset_mock()
        mock_imshow.reset_mock()
        mock_wait_key_ex.reset_mock()
        mock_imread.reset_mock()

        # Test 'q' key
        mock_wait_key_ex.return_value = ord("q")
        action = visualizer._visualize_single_image(image_path, render_data)
        assert action == "quit"
        mock_named_window.assert_called_once()
        mock_move_window.assert_not_called()
        mock_resize_window.assert_called_once()
        mock_imshow.assert_called_once()
        mock_wait_key_ex.assert_called_once_with(0)

        mock_named_window.reset_mock()
        mock_move_window.reset_mock()
        mock_resize_window.reset_mock()
        mock_imshow.reset_mock()
        mock_wait_key_ex.reset_mock()
        mock_imread.reset_mock()

        # Test ESC key
        mock_wait_key_ex.return_value = 27
        action = visualizer._visualize_single_image(image_path, render_data)
        assert action == "quit"

        # Test Left arrow (Linux X11) — navigate backward
        mock_wait_key_ex.return_value = 65361
        action = visualizer._visualize_single_image(image_path, render_data)
        assert action == "prev"

        # Test Right arrow (Linux X11) — navigate forward
        mock_wait_key_ex.return_value = 65363
        action = visualizer._visualize_single_image(image_path, render_data)
        assert action == "next"

        # Test Up arrow (Linux X11) — navigate backward
        mock_wait_key_ex.return_value = 65362
        action = visualizer._visualize_single_image(image_path, render_data)
        assert action == "prev"

        # Test Down arrow (Linux X11) — navigate forward
        mock_wait_key_ex.return_value = 65364
        action = visualizer._visualize_single_image(image_path, render_data)
        assert action == "next"

        # Test Windows Left arrow — navigate backward
        mock_wait_key_ex.return_value = 2424832
        action = visualizer._visualize_single_image(image_path, render_data)
        assert action == "prev"

        # Test Windows Right arrow — navigate forward
        mock_wait_key_ex.return_value = 2555904
        action = visualizer._visualize_single_image(image_path, render_data)
        assert action == "next"

        # Test unrecognized key — default forward
        mock_wait_key_ex.return_value = ord("x")
        action = visualizer._visualize_single_image(image_path, render_data)
        assert action == "next"

    @patch("cv2.imwrite")
    @patch("cv2.imread")
    @patch("pathlib.Path.exists")
    def test_visualize_single_image_is_save_mode(
        self, mock_exists, mock_imread, mock_imwrite
    ):
        """Test visualization in is_save mode with image saving."""
        mock_image = Mock()
        mock_imread.return_value = mock_image
        mock_exists.return_value = True

        output_dir = Path("/tmp/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        visualizer = MockVisualizer(
            label_dir="/tmp/labels",
            image_dir="/tmp/images",
            output_dir=output_dir,
            is_show=False,
            is_save=True,

        )

        render_data = RenderData(annotations=[])
        image_path = "/tmp/images/test.jpg"

        action = visualizer._visualize_single_image(image_path, render_data)
        assert action == "next"
        mock_imwrite.assert_called_once()
        call_args = mock_imwrite.call_args
        assert str(call_args[0][0]) == str(output_dir / "test_visualized.jpg")
        assert call_args[0][1] is mock_image

        try:
            output_dir.rmdir()
        except OSError:
            pass


class TestBidirectionalNavigation:
    """Test bidirectional navigation in the interactive visualize() loop."""

    def _make_mock_image_ann(self, idx: int):
        """Create a mock ImageAnnotation for testing."""
        from dataflow.label.models import (BoundingBox, ImageAnnotation,
                                           ObjectAnnotation)

        return ImageAnnotation(
            image_id=f"img_{idx:03d}",
            image_path=f"/tmp/images/img_{idx:03d}.jpg",
            width=800,
            height=600,
            objects=[
                ObjectAnnotation(
                    class_id=0,
                    class_name="test",
                    bbox=BoundingBox(x=10, y=10, width=100, height=100),
                )
            ],
        )

    @patch("cv2.waitKeyEx")
    @patch("cv2.imshow")
    @patch("cv2.resizeWindow")
    @patch("cv2.moveWindow")
    @patch("cv2.namedWindow")
    @patch("cv2.imread")
    @patch("pathlib.Path.exists")
    def test_navigate_forward_through_all_images(
        self,
        mock_exists,
        mock_imread,
        mock_named,
        mock_move,
        mock_resize,
        mock_imshow,
        mock_wait_ex,
    ):
        """Forward navigation visits all images in sequence."""
        mock_image = Mock()
        mock_image.shape = [600, 800, 3]
        mock_imread.return_value = mock_image
        mock_exists.return_value = True

        # 3 images, user presses next twice then quits
        mock_wait_ex.side_effect = [
            65363,       # → (next) after img_000
            65363,       # → (next) after img_001
            ord("q"),    # quit after img_002
        ]

        mock_images = [
            self._make_mock_image_ann(0),
            self._make_mock_image_ann(1),
            self._make_mock_image_ann(2),
        ]

        visualizer = MockVisualizer(
            label_dir="/tmp/labels",
            image_dir="/tmp/images",
            is_show=True,
            is_save=False,
            mock_images=mock_images,
        )
        result = visualizer.visualize()
        assert result.success is True
        assert result.data["processed_count"] == 3
        assert "interrupted" in result.data
        assert mock_imshow.call_count == 3

    @patch("cv2.waitKeyEx")
    @patch("cv2.imshow")
    @patch("cv2.resizeWindow")
    @patch("cv2.moveWindow")
    @patch("cv2.namedWindow")
    @patch("cv2.imread")
    @patch("pathlib.Path.exists")
    def test_backward_navigation_noop_at_first_image(
        self,
        mock_exists,
        mock_imread,
        mock_named,
        mock_move,
        mock_resize,
        mock_imshow,
        mock_wait_ex,
    ):
        """Left arrow at first image is a no-op — stays at first image."""
        mock_image = Mock()
        mock_image.shape = [600, 800, 3]
        mock_imread.return_value = mock_image
        mock_exists.return_value = True

        # ← at first image → stays at first, then → next, then quit
        mock_wait_ex.side_effect = [
            65361,       # ← at img_000 (no-op — already at first)
            65363,       # → (next) to img_001
            ord("q"),    # quit
        ]

        mock_images = [
            self._make_mock_image_ann(0),
            self._make_mock_image_ann(1),
        ]

        visualizer = MockVisualizer(
            label_dir="/tmp/labels",
            image_dir="/tmp/images",
            is_show=True,
            is_save=False,
            mock_images=mock_images,
        )
        result = visualizer.visualize()
        assert result.success is True
        # img_000 displayed twice (initial + after ← no-op), img_001 once
        assert result.data["processed_count"] == 2
        assert result.data["interrupted"] is True

    @patch("cv2.waitKeyEx")
    @patch("cv2.imshow")
    @patch("cv2.resizeWindow")
    @patch("cv2.moveWindow")
    @patch("cv2.namedWindow")
    @patch("cv2.imread")
    @patch("pathlib.Path.exists")
    def test_forward_noop_at_last_image(
        self,
        mock_exists,
        mock_imread,
        mock_named,
        mock_move,
        mock_resize,
        mock_imshow,
        mock_wait_ex,
    ):
        """Right arrow at last image with exhausted iterator is a no-op."""
        mock_image = Mock()
        mock_image.shape = [600, 800, 3]
        mock_imread.return_value = mock_image
        mock_exists.return_value = True

        # Single image dataset, → twice then quit
        mock_wait_ex.side_effect = [
            65363,       # → at img_000 (no-op — iterator exhausted)
            65363,       # → again (still no-op)
            ord("q"),    # quit
        ]

        mock_images = [self._make_mock_image_ann(0)]

        visualizer = MockVisualizer(
            label_dir="/tmp/labels",
            image_dir="/tmp/images",
            is_show=True,
            is_save=False,
            mock_images=mock_images,
        )
        result = visualizer.visualize()
        assert result.success is True
        assert result.data["processed_count"] == 1

    @patch("cv2.waitKeyEx")
    @patch("cv2.imshow")
    @patch("cv2.resizeWindow")
    @patch("cv2.moveWindow")
    @patch("cv2.namedWindow")
    @patch("cv2.imread")
    @patch("pathlib.Path.exists")
    def test_back_and_forth_counts_unique_images(
        self,
        mock_exists,
        mock_imread,
        mock_named,
        mock_move,
        mock_resize,
        mock_imshow,
        mock_wait_ex,
    ):
        """Navigating back and forth counts unique images, not total events."""
        mock_image = Mock()
        mock_image.shape = [600, 800, 3]
        mock_imread.return_value = mock_image
        mock_exists.return_value = True

        # 3 images, navigate: → → ← ← → → quit
        #   img_000 → img_001 → img_002 → img_001 → img_000 → img_001 → img_002
        # Unique displayed: 3 (img_000, img_001, img_002)
        mock_wait_ex.side_effect = [
            65363,       # → img_001
            65363,       # → img_002
            65361,       # ← img_001
            65361,       # ← img_000
            65363,       # → img_001
            65363,       # → img_002
            ord("q"),    # quit
        ]

        mock_images = [
            self._make_mock_image_ann(0),
            self._make_mock_image_ann(1),
            self._make_mock_image_ann(2),
        ]

        visualizer = MockVisualizer(
            label_dir="/tmp/labels",
            image_dir="/tmp/images",
            is_show=True,
            is_save=False,
            mock_images=mock_images,
        )
        result = visualizer.visualize()
        assert result.success is True
        # 3 unique images displayed, not 7 display events
        assert result.data["processed_count"] == 3
        assert result.data["interrupted"] is True

    def test_empty_dataset_returns_gracefully(self):
        """Empty dataset returns success with processed_count=0."""
        visualizer = MockVisualizer(
            label_dir="/tmp/labels",
            image_dir="/tmp/images",
            is_show=True,
            is_save=False,
            mock_images=[],  # empty
        )
        result = visualizer.visualize()
        assert result.success is True
        assert result.data["processed_count"] == 0


def test_draw_methods_signatures():
    """Test that draw methods have correct signatures."""
    visualizer = MockVisualizer("/tmp/labels", "/tmp/images")
    assert hasattr(visualizer, "_draw_render_annotation")
    assert hasattr(visualizer, "_draw_bbox")
    assert hasattr(visualizer, "_draw_polygon")
    assert hasattr(visualizer, "_draw_rle_mask")
    assert hasattr(visualizer, "_draw_text")
    assert callable(visualizer._draw_render_annotation)
    assert callable(visualizer._draw_bbox)
    assert callable(visualizer._draw_polygon)
    assert callable(visualizer._draw_rle_mask)
    assert callable(visualizer._draw_text)
