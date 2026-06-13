"""
Unit tests for LabelMeVisualizer.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from dataflow.visualize import LabelMeVisualizer

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_DET = PROJECT_ROOT / "assets" / "test_data" / "det" / "labelme"
TEST_DATA_SEG = PROJECT_ROOT / "assets" / "test_data" / "seg" / "labelme"


def _load_all(visualizer):
    """Helper: load all annotations using the streaming API."""
    render_data_map = {}
    handler = visualizer._create_handler()
    for image_ann in handler.iter_images():
        render_data = visualizer._convert_to_render_data(image_ann)
        render_data_map[image_ann.image_path] = render_data
    return render_data_map


class TestLabelMeVisualizer:
    """Test LabelMeVisualizer class."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp(prefix="test_labelme_visualizer_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_initialization(self):
        visualizer = LabelMeVisualizer(
            label_dir=TEST_DATA_DET,
            image_dir=TEST_DATA_DET,
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=False,

        )
        assert visualizer.label_dir == TEST_DATA_DET
        assert visualizer.image_dir == TEST_DATA_DET
        assert visualizer.class_file == TEST_DATA_DET / "classes.txt"
        assert visualizer.is_show is False
        assert visualizer.is_save is False
        # Handler is created lazily via _create_handler()
        handler = visualizer._create_handler()
        assert handler is not None

    def test_load_annotations_detection(self):
        visualizer = LabelMeVisualizer(
            label_dir=TEST_DATA_DET,
            image_dir=TEST_DATA_DET,
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=False,
        )

        render_data_map = _load_all(visualizer)
        assert render_data_map is not None
        assert isinstance(render_data_map, dict)
        assert len(render_data_map) > 0

        for image_path, render_data in render_data_map.items():
            assert render_data.image_width > 0
            assert render_data.image_height > 0
            for render_ann in render_data.annotations:
                assert render_ann.bbox is not None or render_ann.polygon is not None
                assert render_ann.class_id >= 0
                assert render_ann.class_name != ""

    def test_load_annotations_segmentation(self):
        if not TEST_DATA_SEG.exists():
            pytest.skip(f"Segmentation test data not found: {TEST_DATA_SEG}")

        visualizer = LabelMeVisualizer(
            label_dir=TEST_DATA_SEG,
            image_dir=TEST_DATA_SEG,
            class_file=TEST_DATA_SEG / "classes.txt",
            is_show=False,
            is_save=False,
        )

        render_data_map = _load_all(visualizer)
        assert render_data_map is not None
        assert len(render_data_map) > 0

        has_segmentation = False
        for render_data in render_data_map.values():
            for render_ann in render_data.annotations:
                if render_ann.polygon is not None:
                    has_segmentation = True
                    assert len(render_ann.polygon) > 0
        assert has_segmentation, "No segmentation annotations found"

    def test_visualize_detection(self, temp_dir):
        visualizer = LabelMeVisualizer(
            label_dir=TEST_DATA_DET,
            image_dir=TEST_DATA_DET,
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
        visualizer = LabelMeVisualizer(
            label_dir=TEST_DATA_DET,
            image_dir=TEST_DATA_DET,
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=True,
            output_dir=temp_dir,

        )

        result = visualizer.visualize()
        assert result.success is True

        image_count = sum(
            1 for _ in visualizer._create_handler().iter_images()
        )
        output_files = list(temp_dir.glob("*_visualized.jpg"))
        assert len(output_files) == image_count

    def test_visualize_with_invalid_paths(self):
        with pytest.raises(ValueError):
            visualizer = LabelMeVisualizer(
                label_dir="/invalid/path",
                image_dir="/invalid/path",
                is_show=False,
                is_save=False,
    
            )
            handler = visualizer._create_handler()
            list(handler.iter_images())

    def test_visualize_without_class_file(self):
        visualizer = LabelMeVisualizer(
            label_dir=TEST_DATA_DET,
            image_dir=TEST_DATA_DET,
            class_file=None,
            is_show=False,
            is_save=False,

        )

        render_data_map = _load_all(visualizer)
        assert render_data_map is not None
        assert len(render_data_map) > 0

    def test_verbose_parameter(self):
        visualizer_no_verbose = LabelMeVisualizer(
            label_dir=TEST_DATA_DET,
            image_dir=TEST_DATA_DET,
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=False,
            verbose=False,
        )
        assert visualizer_no_verbose.verbose is False
        assert visualizer_no_verbose.progress_logger is None

        visualizer_verbose = LabelMeVisualizer(
            label_dir=TEST_DATA_DET,
            image_dir=TEST_DATA_DET,
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=False,
            verbose=True,
        )
        assert visualizer_verbose.verbose is True
        assert visualizer_verbose.progress_logger is not None

    def test_visualize_with_verbose(self, temp_dir):
        visualizer = LabelMeVisualizer(
            label_dir=TEST_DATA_DET,
            image_dir=TEST_DATA_DET,
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=True,
            output_dir=temp_dir,
            verbose=True,
        )

        result = visualizer.visualize()
        assert result.success is True

        image_count = sum(
            1 for _ in visualizer._create_handler().iter_images()
        )
        output_files = list(temp_dir.glob("*_visualized.jpg"))
        assert len(output_files) == image_count
