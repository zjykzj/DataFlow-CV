"""
Unit tests for log_templates.py — output layout summary.
"""

import pytest

from dataflow.convert.base import ConversionResult
from dataflow.convert.log_templates import (
    format_convert_header,
    format_convert_result,
    format_output_layout,
)


class TestFormatOutputLayout:
    """Tests for format_output_layout — scan of the generated target."""

    def test_directory_layout(self, tmp_path):
        """Directories show recursive file counts, files are listed."""
        (tmp_path / "labels").mkdir()
        (tmp_path / "labels" / "a.txt").write_text("", encoding="utf-8")
        (tmp_path / "labels" / "b.txt").write_text("", encoding="utf-8")
        (tmp_path / "images").mkdir()
        (tmp_path / "classes.txt").write_text("", encoding="utf-8")

        layout = format_output_layout(str(tmp_path))

        assert "Output layout:" in layout
        assert "labels/  2 files" in layout
        assert "classes.txt" in layout

    def test_empty_images_hint(self, tmp_path):
        """Empty images/ dir gets the 'images are not copied' hint."""
        (tmp_path / "labels").mkdir()
        (tmp_path / "images").mkdir()

        layout = format_output_layout(str(tmp_path))

        assert "images/  empty — images are not copied; place your image files here" in layout

    def test_other_empty_dir_generic(self, tmp_path):
        """Non-images empty dirs just show 'empty'."""
        (tmp_path / "labels").mkdir()

        layout = format_output_layout(str(tmp_path))

        assert "labels/  empty" in layout

    def test_single_file_target(self, tmp_path):
        """Single-file target (COCO JSON) shows the file name."""
        out = tmp_path / "coco.json"
        out.write_text("{}", encoding="utf-8")

        layout = format_output_layout(str(out))

        assert "coco.json  (single file)" in layout

    def test_missing_path_returns_empty(self, tmp_path):
        """Non-existent target path → empty string (skipped silently)."""
        assert format_output_layout(str(tmp_path / "nope")) == ""


class TestFormatConvertResultLayout:
    """Tests for the layout summary appended to the result block."""

    def _result(self, target_path):
        return ConversionResult(
            success=True,
            source_format="coco",
            target_format="yolo",
            source_path="/tmp/src.json",
            target_path=target_path,
            num_images_converted=1,
            num_objects_converted=1,
        )

    def test_result_includes_layout(self, tmp_path):
        """Result block is followed by the output layout summary."""
        (tmp_path / "labels").mkdir()
        (tmp_path / "labels" / "a.txt").write_text("", encoding="utf-8")

        output = format_convert_result(self._result(str(tmp_path)))

        assert "Output layout:" in output
        assert "labels/  1 files" in output
        assert output.index("Output layout:") > output.index("Status:")

    def test_result_without_target_skips_layout(self):
        """Empty target_path → no layout section."""
        output = format_convert_result(self._result(""))
        assert "Output layout:" not in output

    def test_header_unaffected(self):
        """Header formatting remains intact."""
        header = format_convert_header(
            source_format="coco",
            target_format="yolo",
            source_path="/tmp/a.json",
            target_path="/tmp/out",
        )
        assert "Convert: COCO → YOLO" in header
        assert "/tmp/a.json" in header


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
