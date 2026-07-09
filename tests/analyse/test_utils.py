"""Tests for analyse utility functions."""

import tempfile
from pathlib import Path

import pytest

from dataflow.analyse.utils import (
    create_handler,
    detect_format,
    load_class_names,
)


# ---------------------------------------------------------------------------
# Test data paths
# ---------------------------------------------------------------------------

TEST_DATA = Path(__file__).parent.parent.parent / "assets" / "test_data"


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------


class TestDetectFormat:
    """Tests for auto-detection of annotation format."""

    def test_detect_coco_file(self):
        """Single .json file is detected as COCO."""
        fmt = detect_format(TEST_DATA / "det" / "coco" / "annotations.json")
        assert fmt == "coco"

    def test_detect_yolo_dir(self):
        """Directory with .txt files is detected as YOLO."""
        fmt = detect_format(TEST_DATA / "det" / "yolo" / "labels")
        assert fmt == "yolo"

    def test_detect_labelme_dir(self):
        """Directory with LabelMe .json files is detected as labelme."""
        fmt = detect_format(TEST_DATA / "det" / "labelme")
        assert fmt == "labelme"

    def test_nonexistent_path(self):
        """Non-existent path raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            detect_format(Path("/nonexistent/path"))

    def test_empty_dir(self):
        """Empty directory raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No annotation files found"):
                detect_format(Path(tmpdir))

    def test_non_json_single_file(self):
        """Single non-JSON file raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            f.write(b"hello")
            f.flush()
            with pytest.raises(ValueError, match=".json"):
                detect_format(Path(f.name))


# ---------------------------------------------------------------------------
# load_class_names
# ---------------------------------------------------------------------------


class TestLoadClassNames:
    """Tests for classes.txt parsing."""

    def test_standard_file(self):
        """Standard classes.txt with one name per line."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("cat\ndog\nperson\n")
            f.flush()
            result = load_class_names(Path(f.name))
        assert result == {0: "cat", 1: "dog", 2: "person"}

    def test_with_blank_lines(self):
        """Blank lines are skipped."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("cat\n\n\ndog\n\nperson\n")
            f.flush()
            result = load_class_names(Path(f.name))
        assert result == {0: "cat", 1: "dog", 2: "person"}

    def test_with_comments(self):
        """Lines starting with # are skipped."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("# header comment\ncat\ndog\n# inline\nperson\n")
            f.flush()
            result = load_class_names(Path(f.name))
        assert result == {0: "cat", 1: "dog", 2: "person"}

    def test_file_not_found(self):
        """Non-existent class file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Class file not found"):
            load_class_names(Path("/nonexistent/classes.txt"))

    def test_empty_file(self):
        """Empty file (no valid class names) raises ValueError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("\n\n")  # only blank lines
            f.flush()
            with pytest.raises(ValueError, match="No class names found"):
                load_class_names(Path(f.name))


# ---------------------------------------------------------------------------
# create_handler
# ---------------------------------------------------------------------------


class TestCreateHandler:
    """Tests for handler factory."""

    def test_creates_yolo_handler(self):
        """Handler created for YOLO format."""
        yolo_labels = TEST_DATA / "det" / "yolo" / "labels"
        class_file = TEST_DATA / "det" / "yolo" / "classes.txt"
        handler = create_handler(
            yolo_labels, "yolo", class_file=class_file
        )
        from dataflow.label.yolo_handler import YoloAnnotationHandler

        assert isinstance(handler, YoloAnnotationHandler)
        assert handler.strict_mode is False

    def test_creates_coco_handler(self):
        """Handler created for COCO format."""
        coco_file = TEST_DATA / "det" / "coco" / "annotations.json"
        handler = create_handler(coco_file, "coco")
        from dataflow.label.coco_handler import CocoAnnotationHandler

        assert isinstance(handler, CocoAnnotationHandler)
        assert handler.strict_mode is False

    def test_creates_labelme_handler(self):
        """Handler created for LabelMe format."""
        labelme_dir = TEST_DATA / "det" / "labelme"
        handler = create_handler(labelme_dir, "labelme")
        from dataflow.label.labelme_handler import LabelMeAnnotationHandler

        assert isinstance(handler, LabelMeAnnotationHandler)
        assert handler.strict_mode is False

    def test_yolo_without_class_file_raises(self):
        """YOLO format requires class_file."""
        yolo_labels = TEST_DATA / "det" / "yolo" / "labels"
        with pytest.raises(ValueError, match="class_file is required"):
            create_handler(yolo_labels, "yolo")

    def test_unknown_format_raises(self):
        """Unknown format string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown format"):
            create_handler(Path("."), "invalid")
