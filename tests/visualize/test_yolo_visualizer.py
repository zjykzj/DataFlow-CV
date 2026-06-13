"""
Unit tests for YOLOVisualizer.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from dataflow.util.logging import LogConfig
from dataflow.visualize import YOLOVisualizer

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_DET = PROJECT_ROOT / "assets" / "test_data" / "det" / "yolo"
TEST_DATA_SEG = PROJECT_ROOT / "assets" / "test_data" / "seg" / "yolo"


def _load_all(visualizer):
    """Helper: load all annotations using the streaming API."""
    render_data_map = {}
    handler = visualizer._create_handler()
    for image_ann in handler.iter_images():
        render_data = visualizer._convert_to_render_data(image_ann)
        render_data_map[image_ann.image_path] = render_data
    return render_data_map


class TestYOLOVisualizer:
    """Test YOLOVisualizer class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test output."""
        temp_dir = tempfile.mkdtemp(prefix="test_yolo_visualizer_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test visualizer initialization."""
        visualizer = YOLOVisualizer(
            label_dir=TEST_DATA_DET / "labels",
            image_dir=TEST_DATA_DET / "images",
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=False,

        )
        assert visualizer.label_dir == TEST_DATA_DET / "labels"
        assert visualizer.image_dir == TEST_DATA_DET / "images"
        assert visualizer.class_file == TEST_DATA_DET / "classes.txt"
        assert visualizer.is_show is False
        assert visualizer.is_save is False
        # Handler is created lazily via _create_handler()
        handler = visualizer._create_handler()
        assert handler is not None

    def test_load_annotations_detection(self):
        """Test loading detection annotations (streaming)."""
        visualizer = YOLOVisualizer(
            label_dir=TEST_DATA_DET / "labels",
            image_dir=TEST_DATA_DET / "images",
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=False,
        )

        render_data_map = _load_all(visualizer)
        assert render_data_map is not None
        assert isinstance(render_data_map, dict)
        assert len(render_data_map) > 0

        # Check that render annotations have bboxes
        for image_path, render_data in render_data_map.items():
            assert render_data.image_width > 0
            assert render_data.image_height > 0
            for render_ann in render_data.annotations:
                assert render_ann.bbox is not None or render_ann.polygon is not None
                assert render_ann.class_id >= 0
                assert render_ann.class_name != ""

    def test_load_annotations_segmentation(self):
        """Test loading segmentation annotations (streaming)."""
        if not TEST_DATA_SEG.exists():
            pytest.skip(f"Segmentation test data not found: {TEST_DATA_SEG}")

        visualizer = YOLOVisualizer(
            label_dir=TEST_DATA_SEG / "labels",
            image_dir=TEST_DATA_SEG / "images",
            class_file=TEST_DATA_SEG / "classes.txt",
            is_show=False,
            is_save=False,
        )

        render_data_map = _load_all(visualizer)
        assert render_data_map is not None
        assert len(render_data_map) > 0

        # Check that some annotations have polygon data
        has_polygon = False
        for render_data in render_data_map.values():
            for render_ann in render_data.annotations:
                if render_ann.polygon is not None:
                    has_polygon = True
                    assert len(render_ann.polygon) > 0
        assert has_polygon, "No polygon annotations found"

    def test_visualize_detection(self, temp_dir):
        """Test visualization of detection data (no display, no save)."""
        visualizer = YOLOVisualizer(
            label_dir=TEST_DATA_DET / "labels",
            image_dir=TEST_DATA_DET / "images",
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=False,

        )

        result = visualizer.visualize()
        assert result.success is True
        image_count = sum(
            1 for _ in visualizer._create_handler().iter_images()
        )
        assert result.data["processed_count"] == image_count
        assert "Visualization completed:" in result.message

    def test_visualize_with_save(self, temp_dir):
        """Test visualization with save mode."""
        visualizer = YOLOVisualizer(
            label_dir=TEST_DATA_DET / "labels",
            image_dir=TEST_DATA_DET / "images",
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=True,
            output_dir=temp_dir,

        )

        result = visualizer.visualize()
        assert result.success is True

        output_files = list(temp_dir.glob("*_visualized.jpg"))
        image_count = sum(
            1 for _ in visualizer._create_handler().iter_images()
        )
        assert len(output_files) == image_count

    def test_visualize_with_invalid_paths(self):
        """Test visualization with invalid paths."""
        with pytest.raises(ValueError):
            visualizer = YOLOVisualizer(
                label_dir="/invalid/path",
                image_dir="/invalid/path",
                class_file="/invalid/path",
                is_show=False,
                is_save=False,
    
            )
            handler = visualizer._create_handler()
            list(handler.iter_images())

    def test_visualize_with_missing_class_file(self):
        """Test visualization with missing class file."""
        with pytest.raises(ValueError):
            visualizer = YOLOVisualizer(
                label_dir=TEST_DATA_DET / "labels",
                image_dir=TEST_DATA_DET / "images",
                class_file="/nonexistent/classes.txt",
                is_show=False,
                is_save=False,
    
            )
            handler = visualizer._create_handler()
            list(handler.iter_images())

    def test_verbose_parameter(self):
        """Test verbose parameter functionality."""
        visualizer_no_verbose = YOLOVisualizer(
            label_dir=TEST_DATA_DET / "labels",
            image_dir=TEST_DATA_DET / "images",
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=False,
            log_config=LogConfig(name="test", verbose=False),
        )
        assert visualizer_no_verbose._log_manager.log_path is None
        assert visualizer_no_verbose._log_manager is not None

        visualizer_verbose = YOLOVisualizer(
            label_dir=TEST_DATA_DET / "labels",
            image_dir=TEST_DATA_DET / "images",
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=False,
            log_config=LogConfig(name="test", verbose=True),
        )
        assert visualizer_verbose._log_manager.log_path is not None
        assert visualizer_verbose._log_manager is not None

    def test_visualize_with_verbose(self, temp_dir):
        """Test visualization with verbose mode."""
        visualizer = YOLOVisualizer(
            label_dir=TEST_DATA_DET / "labels",
            image_dir=TEST_DATA_DET / "images",
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=True,
            output_dir=temp_dir,
            log_config=LogConfig(name="test", verbose=True),
        )

        result = visualizer.visualize()
        assert result.success is True

        output_files = list(temp_dir.glob("*_visualized.jpg"))
        image_count = sum(
            1 for _ in visualizer._create_handler().iter_images()
        )
        assert len(output_files) == image_count
