"""
Unit tests for COCOVisualizer.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from dataflow.visualize import COCOVisualizer

# Test data paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_DET = PROJECT_ROOT / "assets" / "test_data" / "det" / "coco"
TEST_DATA_SEG = PROJECT_ROOT / "assets" / "test_data" / "seg" / "coco"


class TestCOCOVisualizer:
    """Test COCOVisualizer class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test output."""
        temp_dir = tempfile.mkdtemp(prefix="test_coco_visualizer_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test visualizer initialization."""
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
        """Test loading detection annotations."""
        annotation_file = TEST_DATA_DET / "annotations.json"
        if not annotation_file.exists():
            pytest.skip(f"Annotation file not found: {annotation_file}")

        visualizer = COCOVisualizer(
            annotation_file=annotation_file,
            image_dir=TEST_DATA_DET / "images",
            is_show=False,
            is_save=False,
        )

        annotations = visualizer.load_annotations()
        assert annotations is not None
        assert len(annotations.images) > 0
        assert annotations.num_images > 0
        assert annotations.num_objects > 0

        # Check that objects have bounding boxes
        has_bbox = False
        for image_ann in annotations.images:
            for obj in image_ann.objects:
                if obj.bbox is not None:
                    has_bbox = True
                assert obj.class_id >= 0
                assert obj.class_name != ""
        assert has_bbox, "No bounding boxes found"

    def test_load_annotations_segmentation(self):
        """Test loading segmentation annotations."""
        annotation_file = TEST_DATA_SEG / "annotations.json"
        if not annotation_file.exists():
            pytest.skip(f"Segmentation annotation file not found: {annotation_file}")

        visualizer = COCOVisualizer(
            annotation_file=annotation_file,
            image_dir=TEST_DATA_SEG / "images",
            is_show=False,
            is_save=False,
        )

        annotations = visualizer.load_annotations()
        assert annotations is not None
        assert len(annotations.images) > 0

        # Check that objects have segmentation
        has_segmentation = False
        for image_ann in annotations.images:
            for obj in image_ann.objects:
                if obj.segmentation is not None:
                    has_segmentation = True
                    if obj.segmentation.has_rle():
                        assert obj.segmentation.rle is not None
                    else:
                        assert len(obj.segmentation.points) > 0
        assert has_segmentation, "No segmentation annotations found"

    def test_visualize_detection(self, temp_dir):
        """Test visualization of detection data (no display, no save)."""
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
        assert (
            result.data["processed_count"] == visualizer.load_annotations().num_images
        )
        assert "Successfully visualized" in result.message

    def test_visualize_with_save(self, temp_dir):
        """Test visualization with save mode."""
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

        # Check that output files were created
        annotations = visualizer.load_annotations()
        output_files = list(temp_dir.glob("*_visualized.jpg"))
        assert len(output_files) == annotations.num_images

    def test_visualize_with_invalid_paths(self):
        """Test visualization with invalid paths."""
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
        """Test visualization with invalid JSON file."""
        invalid_json = temp_dir / "invalid.json"
        invalid_json.write_text("{ invalid json }")

        visualizer = COCOVisualizer(
            annotation_file=invalid_json,
            image_dir=temp_dir,
            is_show=False,
            is_save=False,
            strict_mode=False,  # Non-strict to avoid immediate crash
        )

        # Should raise error when loading annotations
        with pytest.raises(ValueError):
            visualizer.load_annotations()

    def test_verbose_parameter(self):
        """Test verbose parameter functionality."""
        annotation_file = TEST_DATA_DET / "annotations.json"
        if not annotation_file.exists():
            pytest.skip(f"Annotation file not found: {annotation_file}")

        # Test verbose=False (default)
        visualizer_no_verbose = COCOVisualizer(
            annotation_file=annotation_file,
            image_dir=TEST_DATA_DET / "images",
            is_show=False,
            is_save=False,
            verbose=False,
        )
        assert visualizer_no_verbose.verbose is False
        assert visualizer_no_verbose.progress_logger is None

        # Test verbose=True
        visualizer_verbose = COCOVisualizer(
            annotation_file=annotation_file,
            image_dir=TEST_DATA_DET / "images",
            is_show=False,
            is_save=False,
            verbose=True,
        )
        assert visualizer_verbose.verbose is True
        assert hasattr(visualizer_verbose, "progress_logger")
        assert visualizer_verbose.progress_logger is not None

    def test_visualize_with_verbose(self, temp_dir):
        """Test visualization with verbose mode."""
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

        # Check that output files were created
        annotations = visualizer.load_annotations()
        output_files = list(temp_dir.glob("*_visualized.jpg"))
        assert len(output_files) == annotations.num_images
