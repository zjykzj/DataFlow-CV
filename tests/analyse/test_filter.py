"""Tests for FilterAnalyser."""

from pathlib import Path

import pytest

from dataflow.analyse import FilterAnalyser, FilterResult


TEST_DATA = Path(__file__).parent.parent.parent / "assets" / "test_data"


class TestFilterAnalyser:
    """Tests for category-based annotation filtering."""

    # ------------------------------------------------------------------
    # YOLO
    # ------------------------------------------------------------------

    def test_filter_yolo_subset(self, tmp_path):
        """Filter YOLO data keeping only person and zebra."""
        new_classes = tmp_path / "new_classes.txt"
        new_classes.write_text("person\nzebra\n")

        analyser = FilterAnalyser()
        result = analyser.analyse(
            TEST_DATA / "det" / "yolo" / "labels",
            original_class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            new_class_file=new_classes,
            output_dir=tmp_path / "output",
        )

        assert result.success
        fr = result.data
        assert isinstance(fr, FilterResult)
        assert fr.total_files == 2
        assert fr.total_files_with_annotations == 2
        assert fr.total_annotations_before == 6
        assert fr.total_annotations_after == 5  # elephant removed
        assert fr.format == "yolo"

        # Verify class mapping
        assert len(fr.kept_categories) == 2
        assert fr.kept_categories[0].new_id == 0
        assert fr.kept_categories[0].name == "person"
        assert fr.kept_categories[1].new_id == 1
        assert fr.kept_categories[1].name == "zebra"
        assert fr.kept_categories[1].old_id == 22

        assert len(fr.missing_categories) == 0

        # Verify output files exist
        out_dir = tmp_path / "output"
        assert (out_dir / "classes.txt").read_text().strip() == "person\nzebra"
        assert (out_dir / "image1.txt").exists()
        assert (out_dir / "image2.txt").exists()

    def test_filter_yolo_reorder(self, tmp_path):
        """Filter YOLO with reordered classes (zebra first, then person)."""
        new_classes = tmp_path / "new_classes.txt"
        new_classes.write_text("zebra\nperson\n")

        analyser = FilterAnalyser()
        result = analyser.analyse(
            TEST_DATA / "det" / "yolo" / "labels",
            original_class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            new_class_file=new_classes,
            output_dir=tmp_path / "output",
        )

        assert result.success
        fr = result.data
        assert fr.kept_categories[0].new_id == 0
        assert fr.kept_categories[0].name == "zebra"
        assert fr.kept_categories[1].new_id == 1
        assert fr.kept_categories[1].name == "person"

    def test_filter_yolo_empty_result(self, tmp_path):
        """Filter with no matching categories produces empty annotations."""
        new_classes = tmp_path / "new_classes.txt"
        new_classes.write_text("cat\ndog\n")

        analyser = FilterAnalyser()
        result = analyser.analyse(
            TEST_DATA / "det" / "yolo" / "labels",
            original_class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            new_class_file=new_classes,
            output_dir=tmp_path / "output",
        )

        assert result.success
        fr = result.data
        assert fr.total_annotations_after == 0
        assert fr.total_files_with_annotations == 0
        assert len(result.warnings) > 0

    # ------------------------------------------------------------------
    # COCO
    # ------------------------------------------------------------------

    def test_filter_coco_subset(self, tmp_path):
        """Filter COCO data keeping person, elephant, zebra."""
        new_classes = tmp_path / "new_classes.txt"
        new_classes.write_text("person\nelephant\nzebra\n")

        analyser = FilterAnalyser()
        result = analyser.analyse(
            TEST_DATA / "det" / "coco" / "annotations.json",
            original_class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            new_class_file=new_classes,
            output_dir=tmp_path / "output",
        )

        assert result.success
        fr = result.data
        assert fr.total_files == 2
        assert fr.total_files_with_annotations == 2
        assert fr.total_annotations_before == 6
        assert fr.total_annotations_after == 6  # all three present
        assert fr.format == "coco"

        # Verify class remapping: COCO 1-indexed → new 0-indexed
        assert fr.kept_categories[0].old_id == 1   # person in COCO
        assert fr.kept_categories[1].old_id == 21  # elephant in COCO
        assert fr.kept_categories[2].old_id == 23  # zebra in COCO
        assert fr.kept_categories[0].new_id == 0
        assert fr.kept_categories[1].new_id == 1
        assert fr.kept_categories[2].new_id == 2

        # Verify output file
        out_dir = tmp_path / "output"
        import json
        with open(out_dir / "annotations.json") as f:
            data = json.load(f)
        assert len(data["categories"]) == 3
        assert data["categories"][0]["id"] == 0
        assert data["categories"][0]["name"] == "person"
        assert data["categories"][1]["id"] == 1
        assert data["categories"][1]["name"] == "elephant"
        assert data["categories"][2]["id"] == 2
        assert data["categories"][2]["name"] == "zebra"

    # ------------------------------------------------------------------
    # LabelMe
    # ------------------------------------------------------------------

    def test_filter_labelme(self, tmp_path):
        """Filter LabelMe data keeping only person and zebra."""
        new_classes = tmp_path / "new_classes.txt"
        new_classes.write_text("person\nzebra\n")

        analyser = FilterAnalyser()
        result = analyser.analyse(
            TEST_DATA / "det" / "labelme",
            original_class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            new_class_file=new_classes,
            output_dir=tmp_path / "output",
        )

        assert result.success
        fr = result.data
        assert fr.total_files == 2
        assert fr.total_files_with_annotations == 2
        assert fr.format == "labelme"

        # Verify output
        out_dir = tmp_path / "output"
        assert (out_dir / "classes.txt").exists()
        assert len(list(out_dir.glob("*.json"))) == 2

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_missing_original_class_file(self, tmp_path):
        """Missing original class file produces an error."""
        new_classes = tmp_path / "new_classes.txt"
        new_classes.write_text("person\n")

        analyser = FilterAnalyser()
        result = analyser.analyse(
            TEST_DATA / "det" / "yolo" / "labels",
            original_class_file=Path("/nonexistent/classes.txt"),
            new_class_file=new_classes,
            output_dir=tmp_path / "output",
        )
        assert not result.success
        assert len(result.errors) > 0

    def test_missing_new_class_file(self, tmp_path):
        """Missing new class file produces an error."""
        analyser = FilterAnalyser()
        result = analyser.analyse(
            TEST_DATA / "det" / "yolo" / "labels",
            original_class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            new_class_file=Path("/nonexistent/new_classes.txt"),
            output_dir=Path("/tmp/output"),
        )
        assert not result.success
        assert len(result.errors) > 0

    def test_no_matching_categories(self, tmp_path):
        """New class file with no overlap produces an error."""
        new_classes = tmp_path / "new_classes.txt"
        new_classes.write_text("cat\ndog\n")

        analyser = FilterAnalyser()
        result = analyser.analyse(
            TEST_DATA / "det" / "yolo" / "labels",
            original_class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            new_class_file=new_classes,
            output_dir=tmp_path / "output",
        )
        assert result.success  # not an error — just empty result
        assert result.data.total_annotations_after == 0

    def test_nonexistent_label_path(self, tmp_path):
        """Non-existent label path produces an error."""
        new_classes = tmp_path / "new_classes.txt"
        new_classes.write_text("person\n")

        analyser = FilterAnalyser()
        result = analyser.analyse(
            Path("/nonexistent/path"),
            original_class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            new_class_file=new_classes,
            output_dir=tmp_path / "output",
        )
        assert not result.success

    def test_missing_categories_in_new_file(self, tmp_path):
        """New class file with a non-existent class name emits a warning."""
        new_classes = tmp_path / "new_classes.txt"
        new_classes.write_text("person\nunicorn\n")  # unicorn doesn't exist

        analyser = FilterAnalyser()
        result = analyser.analyse(
            TEST_DATA / "det" / "yolo" / "labels",
            original_class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            new_class_file=new_classes,
            output_dir=tmp_path / "output",
        )

        assert result.success
        assert "unicorn" in result.data.missing_categories
        assert len(result.warnings) > 0
        # Only person is kept
        assert result.data.total_annotations_after == 2

    def test_empty_new_class_file(self, tmp_path):
        """Empty new class file produces an error."""
        new_classes = tmp_path / "new_classes.txt"
        new_classes.write_text("")

        analyser = FilterAnalyser()
        result = analyser.analyse(
            TEST_DATA / "det" / "yolo" / "labels",
            original_class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            new_class_file=new_classes,
            output_dir=tmp_path / "output",
        )
        assert not result.success
