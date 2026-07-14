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

    # ------------------------------------------------------------------
    # Multi-path
    # ------------------------------------------------------------------

    def test_multi_path_yolo(self):
        """Stats on two YOLO dirs merges correctly."""
        analyser = StatsAnalyser()
        path1 = TEST_DATA / "det" / "yolo" / "labels"
        path2 = TEST_DATA / "det" / "yolo" / "labels"  # same data, doubled
        result = analyser.analyse(
            [path1, path2],
            class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
        )
        assert result.success
        stats = result.data
        # Each path has 2 files, 6 annotations → doubled
        assert stats.total_files == 4
        assert stats.total_annotations == 12
        assert stats.per_class["person"] == 4  # 2×2
        assert stats.per_class["zebra"] == 6   # 3×2
        assert stats.per_class["elephant"] == 2  # 1×2
        assert len(stats.source_paths) == 2

    def test_multi_path_mixed_format_errors(self):
        """YOLO + COCO in multi-path produces an error."""
        analyser = StatsAnalyser()
        result = analyser.analyse([
            TEST_DATA / "det" / "yolo" / "labels",
            TEST_DATA / "det" / "coco" / "annotations.json",
        ])
        assert not result.success
        assert "same format" in result.errors[0].lower()

    def test_multi_path_single_path_backward_compat(self):
        """Single Path (not list) still works."""
        analyser = StatsAnalyser()
        result = analyser.analyse(
            TEST_DATA / "det" / "yolo" / "labels",
            class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
        )
        assert result.success
        stats = result.data
        assert stats.total_files == 2
        assert len(stats.source_paths) == 1

    # ------------------------------------------------------------------
    # Strict class validation
    # ------------------------------------------------------------------

    def test_strict_validation_unknown_class_errors(self, tmp_path):
        """Data class not in class_file → error."""
        # Create a class_file with only "person" (missing "zebra", "elephant")
        subset_classes = tmp_path / "subset.txt"
        subset_classes.write_text("person\n")

        analyser = StatsAnalyser()
        result = analyser.analyse(
            TEST_DATA / "det" / "yolo" / "labels",
            class_file=subset_classes,
        )
        assert not result.success
        assert "not found in class file" in result.errors[0]
        # YOLO pre-scan reports invalid class IDs (20=elephant, 22=zebra)
        assert "20" in result.errors[0] and "22" in result.errors[0]

    def test_strict_validation_all_known_passes(self):
        """All data classes present in class_file → success."""
        analyser = StatsAnalyser()
        result = analyser.analyse(
            TEST_DATA / "det" / "yolo" / "labels",
            class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
        )
        assert result.success
        # class_file has: person, ..., elephant(20), zebra(22) — all present
        stats = result.data
        assert "person" in stats.per_class

    # ------------------------------------------------------------------
    # Recursive traversal
    # ------------------------------------------------------------------

    def test_recursive_yolo(self, tmp_path):
        """Recursive stats on nested YOLO dirs."""
        # Create nested structure: root/sub_a/labels/*.txt, root/sub_b/labels/*.txt
        root = tmp_path / "nested"
        sub_a = root / "sub_a"
        sub_b = root / "sub_b"
        sub_a.mkdir(parents=True)
        sub_b.mkdir(parents=True)

        # Copy test YOLO files into both subdirs
        import shutil
        src = TEST_DATA / "det" / "yolo" / "labels"
        for f in src.glob("*.txt"):
            shutil.copy2(str(f), str(sub_a / f.name))
            shutil.copy2(str(f), str(sub_b / f.name))

        analyser = StatsAnalyser()
        result = analyser.analyse(
            root,
            recursive=True,
        )
        assert result.success
        stats = result.data
        # Each subdir has 2 files → 4 total.
        # Without class_file, _auto_generate_class_file scans the
        # original path with rglob (recursive=True), and the handler
        # also uses rglob to find label files natively.
        assert stats.total_files == 4
        assert stats.total_annotations == 12  # 6×2

    def test_recursive_yolo_with_class_file(self, tmp_path):
        """Recursive YOLO stats with class_file."""
        root = tmp_path / "nested2"
        sub_a = root / "sub_a"
        sub_a.mkdir(parents=True)

        import shutil
        src = TEST_DATA / "det" / "yolo" / "labels"
        for f in src.glob("*.txt"):
            shutil.copy2(str(f), str(sub_a / f.name))

        analyser = StatsAnalyser()
        result = analyser.analyse(
            root,
            class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            recursive=True,
        )
        assert result.success
        stats = result.data
        assert stats.total_files == 2

    def test_recursive_no_files_errors(self, tmp_path):
        """Recursive on dir with no label files → error."""
        empty = tmp_path / "empty_nested"
        empty.mkdir(parents=True)
        (empty / "subdir").mkdir()

        analyser = StatsAnalyser()
        result = analyser.analyse(empty, recursive=True)
        assert not result.success
        assert "No files found" in result.errors[0]
