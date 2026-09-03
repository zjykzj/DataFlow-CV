"""Tests for analyse utility functions."""

import tempfile
from pathlib import Path

import pytest

from dataflow.analyse.utils import (
    _auto_generate_class_file,
    _scan_yolo_class_ids,
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("cat\ndog\nperson\n")
            f.flush()
            result = load_class_names(Path(f.name))
        assert result == {0: "cat", 1: "dog", 2: "person"}

    def test_with_blank_lines(self):
        """Blank lines are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("cat\n\n\ndog\n\nperson\n")
            f.flush()
            result = load_class_names(Path(f.name))
        assert result == {0: "cat", 1: "dog", 2: "person"}

    def test_with_comments(self):
        """Lines starting with # are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
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
        handler = create_handler(yolo_labels, "yolo", class_file=class_file)
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

    def test_yolo_without_class_file_auto_generates(self):
        """YOLO without class_file auto-generates class names from label IDs."""
        yolo_labels = TEST_DATA / "det" / "yolo" / "labels"
        handler = create_handler(yolo_labels, "yolo")
        from dataflow.label.yolo_handler import YoloAnnotationHandler

        assert isinstance(handler, YoloAnnotationHandler)
        assert handler.strict_mode is False
        # Verify the handler can read the dataset
        result = handler.read()
        assert result.success

    def test_unknown_format_raises(self):
        """Unknown format string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown format"):
            create_handler(Path("."), "invalid")


# ---------------------------------------------------------------------------
# Float-tolerant class ID parsing
# ---------------------------------------------------------------------------


class TestAutoGenerateClassFile:
    """Tests for _auto_generate_class_file with float-formatted class IDs."""

    def test_integer_class_ids(self, tmp_path):
        """Plain integer class IDs work."""
        (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (tmp_path / "b.txt").write_text("2 0.3 0.3 0.2 0.2\n")
        result = _auto_generate_class_file(tmp_path)
        assert result.exists()
        names = result.read_text().strip().split("\n")
        assert names[0] == "class_0"
        assert names[2] == "class_2"

    def test_float_formatted_class_ids(self, tmp_path):
        """Float-formatted class IDs like '5.000000' are parsed correctly."""
        (tmp_path / "a.txt").write_text("5.000000 0.5 0.5 0.1 0.1\n")
        (tmp_path / "b.txt").write_text("3.000000 0.3 0.3 0.2 0.2\n")
        result = _auto_generate_class_file(tmp_path)
        assert result.exists()
        names = result.read_text().strip().split("\n")
        assert names[3] == "class_3"
        assert names[5] == "class_5"

    def test_mixed_format_class_ids(self, tmp_path):
        """Mix of integer and float-formatted class IDs."""
        (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (tmp_path / "b.txt").write_text("2.000000 0.3 0.3 0.2 0.2\n")
        (tmp_path / "c.txt").write_text("1 0.4 0.4 0.15 0.15\n")
        result = _auto_generate_class_file(tmp_path)
        names = result.read_text().strip().split("\n")
        assert names[0] == "class_0"
        assert names[1] == "class_1"
        assert names[2] == "class_2"

    def test_recursive_float_class_ids(self, tmp_path):
        """Recursive scan finds float class IDs in subdirectories."""
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (tmp_path / "sub" / "b.txt").write_text("2.000000 0.3 0.3 0.2 0.2\n")
        result = _auto_generate_class_file(tmp_path, recursive=True)
        names = result.read_text().strip().split("\n")
        assert names[0] == "class_0"
        assert names[2] == "class_2"

    def test_no_valid_class_ids_errors(self, tmp_path):
        """Empty or invalid directory raises ValueError."""
        (tmp_path / "a.txt").write_text("0.5 0.5 0.5 0.1 0.1\n")  # non-integer float
        (tmp_path / "b.txt").write_text("abc 0.3 0.3 0.2 0.2\n")  # non-numeric
        with pytest.raises(ValueError, match="No valid class IDs found"):
            _auto_generate_class_file(tmp_path)

    def test_skips_classes_txt(self, tmp_path):
        """classes.txt in the label directory is skipped."""
        (tmp_path / "classes.txt").write_text("ignored\n")
        (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        result = _auto_generate_class_file(tmp_path)
        names = result.read_text().strip().split("\n")
        assert names[0] == "class_0"


class TestScanYoloClassIds:
    """Tests for _scan_yolo_class_ids with float-formatted class IDs."""

    def test_float_class_ids(self, tmp_path):
        """Float-formatted class IDs are found."""
        (tmp_path / "a.txt").write_text("5.000000 0.5 0.5 0.1 0.1\n")
        (tmp_path / "b.txt").write_text("3.000000 0.3 0.3 0.2 0.2\n")
        ids = _scan_yolo_class_ids(tmp_path)
        assert ids == {3, 5}

    def test_mixed_format_ids(self, tmp_path):
        """Mix of integer and float class IDs."""
        (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (tmp_path / "b.txt").write_text("2.000000 0.3 0.3 0.2 0.2\n")
        ids = _scan_yolo_class_ids(tmp_path)
        assert ids == {0, 2}

    def test_recursive_scan(self, tmp_path):
        """Recursive scan finds IDs in subdirectories."""
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (tmp_path / "sub" / "b.txt").write_text("2.000000 0.3 0.3 0.2 0.2\n")
        ids = _scan_yolo_class_ids(tmp_path, recursive=True)
        assert ids == {0, 2}
