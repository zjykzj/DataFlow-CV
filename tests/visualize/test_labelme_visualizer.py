"""
Unit tests for LabelMeVisualizer.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from dataflow.visualize import LabelMeVisualizer

# Test data paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_DET = PROJECT_ROOT / "assets" / "test_data" / "det" / "labelme"
TEST_DATA_SEG = PROJECT_ROOT / "assets" / "test_data" / "seg" / "labelme"


class TestLabelMeVisualizer:
    """Test LabelMeVisualizer class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test output."""
        temp_dir = tempfile.mkdtemp(prefix="test_labelme_visualizer_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test visualizer initialization."""
        visualizer = LabelMeVisualizer(
            label_dir=TEST_DATA_DET,
            image_dir=TEST_DATA_DET,
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=False,
            strict_mode=True
        )

        assert visualizer.label_dir == TEST_DATA_DET
        assert visualizer.image_dir == TEST_DATA_DET
        assert visualizer.class_file == TEST_DATA_DET / "classes.txt"
        assert visualizer.is_show is False
        assert visualizer.is_save is False
        assert visualizer.strict_mode is True
        assert visualizer.handler is not None

    def test_load_annotations_detection(self):
        """Test loading detection annotations."""
        visualizer = LabelMeVisualizer(
            label_dir=TEST_DATA_DET,
            image_dir=TEST_DATA_DET,
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=False
        )

        annotations = visualizer.load_annotations()
        assert annotations is not None
        assert len(annotations.images) > 0
        assert annotations.num_images > 0
        assert annotations.num_objects > 0

        # Check that objects have bounding boxes
        for image_ann in annotations.images:
            for obj in image_ann.objects:
                assert obj.bbox is not None
                assert obj.class_id >= 0
                assert obj.class_name != ""

    def test_load_annotations_segmentation(self):
        """Test loading segmentation annotations."""
        if not TEST_DATA_SEG.exists():
            pytest.skip(f"Segmentation test data not found: {TEST_DATA_SEG}")

        visualizer = LabelMeVisualizer(
            label_dir=TEST_DATA_SEG,
            image_dir=TEST_DATA_SEG,
            class_file=TEST_DATA_SEG / "classes.txt",
            is_show=False,
            is_save=False
        )

        annotations = visualizer.load_annotations()
        assert annotations is not None
        assert len(annotations.images) > 0

        # Check that objects have segmentation polygons
        has_segmentation = False
        for image_ann in annotations.images:
            for obj in image_ann.objects:
                if obj.segmentation is not None:
                    has_segmentation = True
                    assert len(obj.segmentation.points) > 0
        assert has_segmentation, "No segmentation annotations found"

    def test_visualize_detection(self, temp_dir):
        """Test visualization of detection data (no display, no save)."""
        visualizer = LabelMeVisualizer(
            label_dir=TEST_DATA_DET,
            image_dir=TEST_DATA_DET,
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=False,
            strict_mode=True
        )

        result = visualizer.visualize()
        assert result.success is True
        assert result.data["processed_count"] == visualizer.load_annotations().num_images
        assert "Successfully visualized" in result.message

    def test_visualize_with_save(self, temp_dir):
        """Test visualization with save mode."""
        visualizer = LabelMeVisualizer(
            label_dir=TEST_DATA_DET,
            image_dir=TEST_DATA_DET,
            class_file=TEST_DATA_DET / "classes.txt",
            is_show=False,
            is_save=True,
            output_dir=temp_dir,
            strict_mode=True
        )

        result = visualizer.visualize()
        assert result.success is True

        # Check that output files were created
        output_files = list(temp_dir.glob("*_visualized.jpg"))
        assert len(output_files) == visualizer.load_annotations().num_images

    def test_visualize_with_invalid_paths(self):
        """Test visualization with invalid paths."""
        with pytest.raises(ValueError):
            visualizer = LabelMeVisualizer(
                label_dir="/invalid/path",
                image_dir="/invalid/path",
                is_show=False,
                is_save=False,
                strict_mode=True
            )
            visualizer.load_annotations()

    def test_visualize_without_class_file(self):
        """Test visualization without class file."""
        visualizer = LabelMeVisualizer(
            label_dir=TEST_DATA_DET,
            image_dir=TEST_DATA_DET,
            class_file=None,  # No class file
            is_show=False,
            is_save=False,
            strict_mode=False  # Non-strict to avoid errors
        )

        # Should still work (categories extracted from annotations)
        annotations = visualizer.load_annotations()
        assert annotations is not None
        assert len(annotations.categories) > 0