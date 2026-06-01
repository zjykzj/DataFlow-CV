"""
Unit tests for YOLOVisualizer.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from dataflow.visualize import YOLOVisualizer

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_DET = PROJECT_ROOT / "assets" / "test_data" / "det" / "yolo"
TEST_DATA_SEG = PROJECT_ROOT / "assets" / "test_data" / "seg" / "yolo"


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
            strict_mode=True,
        )
        assert visualizer.label_dir == TEST_DATA_DET / "labels"
        assert visualizer.image_dir == TEST_DATA_DET / "images"
        assert visualizer.class_file == TEST_DATA_DET / "classes.txt"
        assert visualizer.is_show is False
        assert visualizer.is_save is False
        assert visualizer.strict_mode is True
        assert visualizer.handler is not None

    def test_load_annotations_detection(self):
        """Test loading detection annotations."""
        visualizer = YOLOVisualizer(
            label_dir=TEST_DATA_DET / "labels",
            image_dir=TEST_DATA_DET / "images",
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=False,
        )

        render_data_map = visualizer.load_annotations()
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
        """Test loading segmentation annotations."""
        if not TEST_DATA_SEG.exists():
            pytest.skip(f"Segmentation test data not found: {TEST_DATA_SEG}")

        visualizer = YOLOVisualizer(
            label_dir=TEST_DATA_SEG / "labels",
            image_dir=TEST_DATA_SEG / "images",
            class_file=TEST_DATA_SEG / "classes.txt",
            is_show=False,
            is_save=False,
        )

        render_data_map = visualizer.load_annotations()
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
            strict_mode=True,
        )

        result = visualizer.visualize()
        assert result.success is True
        render_data_map = visualizer.load_annotations()
        assert result.data["processed_count"] == len(render_data_map)
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
            strict_mode=True,
        )

        result = visualizer.visualize()
        assert result.success is True

        output_files = list(temp_dir.glob("*_visualized.jpg"))
        render_data_map = visualizer.load_annotations()
        assert len(output_files) == len(render_data_map)

    def test_visualize_with_invalid_paths(self):
        """Test visualization with invalid paths."""
        with pytest.raises(ValueError):
            visualizer = YOLOVisualizer(
                label_dir="/invalid/path",
                image_dir="/invalid/path",
                class_file="/invalid/path",
                is_show=False,
                is_save=False,
                strict_mode=True,
            )
            visualizer.load_annotations()

    def test_visualize_with_missing_class_file(self):
        """Test visualization with missing class file."""
        with pytest.raises(ValueError):
            visualizer = YOLOVisualizer(
                label_dir=TEST_DATA_DET / "labels",
                image_dir=TEST_DATA_DET / "images",
                class_file="/nonexistent/classes.txt",
                is_show=False,
                is_save=False,
                strict_mode=True,
            )
            visualizer.load_annotations()

    def test_verbose_parameter(self):
        """Test verbose parameter functionality."""
        visualizer_no_verbose = YOLOVisualizer(
            label_dir=TEST_DATA_DET / "labels",
            image_dir=TEST_DATA_DET / "images",
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=False,
            verbose=False,
        )
        assert visualizer_no_verbose.verbose is False
        assert visualizer_no_verbose.progress_logger is None

        visualizer_verbose = YOLOVisualizer(
            label_dir=TEST_DATA_DET / "labels",
            image_dir=TEST_DATA_DET / "images",
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=False,
            verbose=True,
        )
        assert visualizer_verbose.verbose is True
        assert visualizer_verbose.progress_logger is not None

    def test_visualize_with_verbose(self, temp_dir):
        """Test visualization with verbose mode."""
        visualizer = YOLOVisualizer(
            label_dir=TEST_DATA_DET / "labels",
            image_dir=TEST_DATA_DET / "images",
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=True,
            output_dir=temp_dir,
            verbose=True,
        )

        result = visualizer.visualize()
        assert result.success is True

        output_files = list(temp_dir.glob("*_visualized.jpg"))
        render_data_map = visualizer.load_annotations()
        assert len(output_files) == len(render_data_map)
