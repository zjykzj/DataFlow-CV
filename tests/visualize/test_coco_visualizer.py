"""
Unit tests for COCOVisualizer.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from dataflow.visualize import COCOVisualizer

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_DET = PROJECT_ROOT / "assets" / "test_data" / "det" / "coco"
TEST_DATA_SEG = PROJECT_ROOT / "assets" / "test_data" / "seg" / "coco"


class TestCOCOVisualizer:
    """Test COCOVisualizer class."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp(prefix="test_coco_visualizer_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_initialization(self):
        annotation_file = TEST_DATA_DET / "annotations.json"
        if not annotation_file.exists():
            pytest.skip(f"Annotation file not found: {annotation_file}")

        visualizer = COCOVisualizer(
            annotation_file=annotation_file,
            image_dir=TEST_DATA_DET / "images",
            is_show=False,
            is_save=False,
            strict_mode=True,
        )
        assert visualizer.annotation_file == annotation_file
        assert visualizer.image_dir == TEST_DATA_DET / "images"
        assert visualizer.is_show is False
        assert visualizer.is_save is False
        assert visualizer.strict_mode is True
        assert visualizer.handler is not None

    def test_load_annotations_detection(self):
        annotation_file = TEST_DATA_DET / "annotations.json"
        if not annotation_file.exists():
            pytest.skip(f"Annotation file not found: {annotation_file}")

        visualizer = COCOVisualizer(
            annotation_file=annotation_file,
            image_dir=TEST_DATA_DET / "images",
            is_show=False,
            is_save=False,
        )

        render_data_map = visualizer.load_annotations()
        assert render_data_map is not None
        assert isinstance(render_data_map, dict)
        assert len(render_data_map) > 0

        has_bbox = False
        for image_path, render_data in render_data_map.items():
            assert render_data.image_width > 0
            assert render_data.image_height > 0
            for render_ann in render_data.annotations:
                if render_ann.bbox is not None:
                    has_bbox = True
                assert render_ann.class_id >= 0
                assert render_ann.class_name != ""
        assert has_bbox, "No bounding boxes found"

    def test_load_annotations_segmentation(self):
        annotation_file = TEST_DATA_SEG / "annotations.json"
        if not annotation_file.exists():
            pytest.skip(f"Segmentation annotation file not found: {annotation_file}")

        visualizer = COCOVisualizer(
            annotation_file=annotation_file,
            image_dir=TEST_DATA_SEG / "images",
            is_show=False,
            is_save=False,
        )

        render_data_map = visualizer.load_annotations()
        assert render_data_map is not None
        assert len(render_data_map) > 0

        has_seg = False
        for render_data in render_data_map.values():
            for render_ann in render_data.annotations:
                if render_ann.polygon is not None or render_ann.rle is not None:
                    has_seg = True
        assert has_seg, "No segmentation annotations found"

    def test_visualize_detection(self, temp_dir):
        annotation_file = TEST_DATA_DET / "annotations.json"
        if not annotation_file.exists():
            pytest.skip(f"Annotation file not found: {annotation_file}")

        visualizer = COCOVisualizer(
            annotation_file=annotation_file,
            image_dir=TEST_DATA_DET / "images",
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
        annotation_file = TEST_DATA_DET / "annotations.json"
        if not annotation_file.exists():
            pytest.skip(f"Annotation file not found: {annotation_file}")

        visualizer = COCOVisualizer(
            annotation_file=annotation_file,
            image_dir=TEST_DATA_DET / "images",
            is_show=False,
            is_save=True,
            output_dir=temp_dir,
            strict_mode=True,
        )

        result = visualizer.visualize()
        assert result.success is True

        render_data_map = visualizer.load_annotations()
        output_files = list(temp_dir.glob("*_visualized.jpg"))
        assert len(output_files) == len(render_data_map)

    def test_visualize_with_invalid_paths(self):
        with pytest.raises(ValueError):
            visualizer = COCOVisualizer(
                annotation_file="/invalid/annotations.json",
                image_dir="/invalid/images",
                is_show=False,
                is_save=False,
                strict_mode=True,
            )
            visualizer.load_annotations()

    def test_visualize_with_invalid_json(self, temp_dir):
        invalid_json = temp_dir / "invalid.json"
        invalid_json.write_text("{ invalid json }")

        visualizer = COCOVisualizer(
            annotation_file=invalid_json,
            image_dir=temp_dir,
            is_show=False,
            is_save=False,
            strict_mode=False,
        )

        with pytest.raises(ValueError):
            visualizer.load_annotations()

    def test_verbose_parameter(self):
        annotation_file = TEST_DATA_DET / "annotations.json"
        if not annotation_file.exists():
            pytest.skip(f"Annotation file not found: {annotation_file}")

        visualizer_no_verbose = COCOVisualizer(
            annotation_file=annotation_file,
            image_dir=TEST_DATA_DET / "images",
            is_show=False,
            is_save=False,
            verbose=False,
        )
        assert visualizer_no_verbose.verbose is False
        assert visualizer_no_verbose.progress_logger is None

        visualizer_verbose = COCOVisualizer(
            annotation_file=annotation_file,
            image_dir=TEST_DATA_DET / "images",
            is_show=False,
            is_save=False,
            verbose=True,
        )
        assert visualizer_verbose.verbose is True
        assert visualizer_verbose.progress_logger is not None

    def test_visualize_with_verbose(self, temp_dir):
        annotation_file = TEST_DATA_DET / "annotations.json"
        if not annotation_file.exists():
            pytest.skip(f"Annotation file not found: {annotation_file}")

        visualizer = COCOVisualizer(
            annotation_file=annotation_file,
            image_dir=TEST_DATA_DET / "images",
            is_show=False,
            is_save=True,
            output_dir=temp_dir,
            verbose=True,
        )

        result = visualizer.visualize()
        assert result.success is True

        render_data_map = visualizer.load_annotations()
        output_files = list(temp_dir.glob("*_visualized.jpg"))
        assert len(output_files) == len(render_data_map)
