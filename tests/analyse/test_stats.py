"""Tests for StatsAnalyser."""

from pathlib import Path

import pytest

from dataflow.analyse import StatsAnalyser


TEST_DATA = Path(__file__).parent.parent.parent / "assets" / "test_data"


class TestStatsAnalyser:
    """Tests for dataset statistics computation."""

    def test_stats_yolo(self):
        """Stats on YOLO test data returns correct counts."""
        analyser = StatsAnalyser()
        result = analyser.analyse(
            TEST_DATA / "det" / "yolo" / "labels",
            class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
        )
        assert result.success
        assert result.data is not None

        stats = result.data
        assert stats.total_files == 2
        assert stats.total_annotations == 6
        assert stats.format == "yolo"
        assert "person" in stats.per_class
        assert stats.per_class["person"] == 2
        assert stats.per_class["zebra"] == 3
        assert stats.per_class["elephant"] == 1

    def test_stats_coco(self):
        """Stats on COCO test data returns correct counts."""
        analyser = StatsAnalyser()
        result = analyser.analyse(
            TEST_DATA / "det" / "coco" / "annotations.json"
        )
        assert result.success
        assert result.data is not None

        stats = result.data
        assert stats.total_files == 2
        assert stats.total_annotations == 6
        assert stats.format == "coco"

    def test_stats_labelme(self):
        """Stats on LabelMe test data returns correct counts."""
        analyser = StatsAnalyser()
        result = analyser.analyse(TEST_DATA / "det" / "labelme")
        assert result.success
        assert result.data is not None

        stats = result.data
        assert stats.total_files == 2
        assert stats.total_annotations > 0
        assert stats.format == "labelme"

    def test_class_file_ordering(self):
        """Per-class output respects class_file ordering."""
        analyser = StatsAnalyser()
        # Use COCO where classes.txt controls ordering
        result = analyser.analyse(
            TEST_DATA / "det" / "coco" / "annotations.json",
            class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
        )
        assert result.success
        stats = result.data
        # Classes present in the dataset: person, elephant, zebra
        # In COCO class order: person(0) → elephant(20) → zebra(22)
        names = list(stats.per_class.keys())
        assert names.index("person") < names.index("elephant")
        assert names.index("elephant") < names.index("zebra")

    def test_missing_class_file(self):
        """Missing class_file produces an error."""
        analyser = StatsAnalyser()
        result = analyser.analyse(
            TEST_DATA / "det" / "yolo" / "labels",
            class_file=Path("/nonexistent/classes.txt"),
        )
        assert not result.success
        assert len(result.errors) > 0

    def test_nonexistent_label_path(self):
        """Non-existent label path produces an error."""
        analyser = StatsAnalyser()
        result = analyser.analyse(Path("/nonexistent/path"))
        assert not result.success
        assert len(result.errors) > 0
