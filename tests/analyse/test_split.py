"""Tests for SplitAnalyser."""

import tempfile
from pathlib import Path

import pytest

from dataflow.analyse import SplitAnalyser


TEST_DATA = Path(__file__).parent.parent.parent / "assets" / "test_data"


class TestSplitAnalyser:
    """Tests for dataset train/val splitting."""

    def test_split_yolo_labels_only(self):
        """YOLO labels-only split produces train/val with .txt files."""
        analyser = SplitAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                ratio=0.5,  # 50/50 so both splits get data with 2 images
                seed=42,
                label_dir=TEST_DATA / "det" / "yolo" / "labels",
                class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            )
            assert result.success
            assert result.data is not None

            split = result.data
            assert split.train_count == 1
            assert split.val_count == 1
            assert split.format == "yolo"
            assert split.mode == "labels"
            assert split.seed == 42

            # Check output files exist
            train_dir = output_dir / "train"
            val_dir = output_dir / "val"
            assert train_dir.is_dir()
            assert val_dir.is_dir()
            assert len(list(train_dir.glob("*.txt"))) > 0
            assert len(list(val_dir.glob("*.txt"))) > 0

    def test_split_labelme_labels_only(self):
        """LabelMe labels-only split produces train/val with .json files."""
        analyser = SplitAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                ratio=0.5,
                seed=42,
                label_dir=TEST_DATA / "det" / "labelme",
            )
            assert result.success
            assert result.data is not None

            split = result.data
            assert split.train_count == 1
            assert split.val_count == 1
            assert split.format == "labelme"
            assert split.mode == "labels"

    def test_split_images_only(self):
        """Images-only split copies image files to train/val."""
        analyser = SplitAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                ratio=0.5,
                seed=42,
                image_dir=TEST_DATA / "det" / "yolo" / "images",
            )
            assert result.success
            assert result.data is not None

            split = result.data
            assert split.train_count == 1
            assert split.val_count == 1
            assert split.format == ""  # images-only has no format
            assert split.mode == "images"

            # Check output image files exist
            train_dir = output_dir / "train"
            val_dir = output_dir / "val"
            assert train_dir.is_dir()
            assert val_dir.is_dir()

    def test_split_both(self):
        """Both mode copies labels and images with labels/images subdirs."""
        analyser = SplitAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                ratio=0.5,
                seed=42,
                label_dir=TEST_DATA / "det" / "yolo" / "labels",
                image_dir=TEST_DATA / "det" / "yolo" / "images",
            )
            assert result.success
            assert result.data is not None

            split = result.data
            assert split.mode == "both"
            assert split.format == "yolo"

            # Check both labels and images subdirs exist
            train_labels = output_dir / "train" / "labels"
            train_images = output_dir / "train" / "images"
            val_labels = output_dir / "val" / "labels"
            val_images = output_dir / "val" / "images"
            assert train_labels.is_dir()
            assert train_images.is_dir()
            assert val_labels.is_dir()
            assert val_images.is_dir()

    def test_split_move(self):
        """Move mode relocates source files instead of copying."""
        analyser = SplitAnalyser()
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create temp label files to move (don't touch test assets)
            src_dir = Path(tmpdir) / "src_labels"
            src_dir.mkdir()
            (src_dir / "a.txt").write_text("0 0.5 0.5 0.1 0.1")
            (src_dir / "b.txt").write_text("0 0.5 0.5 0.1 0.1")

            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                ratio=0.5,
                seed=42,
                label_dir=src_dir,
                move=True,
            )
            assert result.success
            assert result.data is not None
            assert result.data.move is True

            # Source files should be gone
            remaining = list(src_dir.glob("*.txt"))
            assert len(remaining) == 0

            # Output files should exist
            train_files = list((output_dir / "train").glob("*.txt"))
            val_files = list((output_dir / "val").glob("*.txt"))
            assert len(train_files) == 1
            assert len(val_files) == 1

    def test_seed_reproducibility(self):
        """Same seed produces identical splits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out1 = Path(tmpdir) / "out1"
            out2 = Path(tmpdir) / "out2"
            analyser = SplitAnalyser()

            r1 = analyser.analyse(
                output_dir=out1,
                ratio=0.5,
                seed=42,
                label_dir=TEST_DATA / "det" / "yolo" / "labels",
                class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            )
            r2 = analyser.analyse(
                output_dir=out2,
                ratio=0.5,
                seed=42,
                label_dir=TEST_DATA / "det" / "yolo" / "labels",
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
                output_dir=output_dir,
                ratio=1.5,
                label_dir=TEST_DATA / "det" / "yolo" / "labels",
                class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            )
            assert not result.success
            assert any("Ratio" in e for e in result.errors)

    def test_no_input_source(self):
        """Missing both label_dir and image_dir produces an error."""
        analyser = SplitAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(output_dir=output_dir)
            assert not result.success
            assert any("label_dir" in e or "image_dir" in e
                      for e in result.errors)

    def test_nonexistent_label_path(self):
        """Non-existent label path produces an error."""
        analyser = SplitAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                label_dir=Path("/nonexistent/path"),
            )
            assert not result.success

    def test_coco_rejected(self):
        """COCO format produces a clear error (only YOLO/LabelMe supported)."""
        analyser = SplitAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                label_dir=TEST_DATA / "det" / "coco" / "annotations.json",
            )
            assert not result.success
            assert any("coco" in e.lower() for e in result.errors)
