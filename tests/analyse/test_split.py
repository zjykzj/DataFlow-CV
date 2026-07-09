"""Tests for SplitAnalyser."""

import tempfile
from pathlib import Path

import pytest

from dataflow.analyse import SplitAnalyser


TEST_DATA = Path(__file__).parent.parent.parent / "assets" / "test_data"


class TestSplitAnalyser:
    """Tests for dataset train/val splitting."""

    def test_split_yolo(self):
        """YOLO split produces train and val directories."""
        analyser = SplitAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                TEST_DATA / "det" / "yolo" / "labels",
                output_dir,
                ratio=0.5,  # 50/50 so both splits get data with 2 images
                seed=42,
                class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            )
            assert result.success
            assert result.data is not None

            split = result.data
            assert split.train_count == 1
            assert split.val_count == 1
            assert split.format == "yolo"
            assert split.seed == 42

            # Check output files exist
            train_dir = output_dir / "train"
            val_dir = output_dir / "val"
            assert train_dir.is_dir()
            assert val_dir.is_dir()
            assert len(list(train_dir.glob("*.txt"))) > 0
            assert len(list(val_dir.glob("*.txt"))) > 0

    def test_split_coco(self):
        """COCO split produces train.json and val.json."""
        analyser = SplitAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                TEST_DATA / "det" / "coco" / "annotations.json",
                output_dir,
                ratio=0.5,
                seed=42,
            )
            assert result.success
            assert result.data is not None

            split = result.data
            assert split.train_count == 1
            assert split.val_count == 1
            assert split.format == "coco"

            # Check output JSON files exist
            train_json = output_dir / "train.json"
            val_json = output_dir / "val.json"
            assert train_json.is_file()
            assert val_json.is_file()

    def test_split_labelme(self):
        """LabelMe split produces train and val directories."""
        analyser = SplitAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                TEST_DATA / "det" / "labelme",
                output_dir,
                ratio=0.5,
                seed=42,
            )
            assert result.success
            assert result.data is not None

            split = result.data
            assert split.train_count == 1
            assert split.val_count == 1
            assert split.format == "labelme"

    def test_seed_reproducibility(self):
        """Same seed produces identical splits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out1 = Path(tmpdir) / "out1"
            out2 = Path(tmpdir) / "out2"
            split = SplitAnalyser()

            r1 = split.analyse(
                TEST_DATA / "det" / "yolo" / "labels",
                out1,
                ratio=0.5,
                seed=42,
                class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            )
            r2 = split.analyse(
                TEST_DATA / "det" / "yolo" / "labels",
                out2,
                ratio=0.5,
                seed=42,
                class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            )
            assert r1.data.train_count == r2.data.train_count
            assert r1.data.val_count == r2.data.val_count

    def test_invalid_ratio(self):
        """Ratio out of range produces an error."""
        analyser = SplitAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                TEST_DATA / "det" / "yolo" / "labels",
                output_dir,
                ratio=1.5,
                class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            )
            assert not result.success
            assert any("Ratio" in e for e in result.errors)

    def test_nonexistent_label_path(self):
        """Non-existent label path produces an error."""
        analyser = SplitAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                Path("/nonexistent/path"),
                output_dir,
            )
            assert not result.success
