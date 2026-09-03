"""Tests for SampleAnalyser."""

import tempfile
from pathlib import Path

import pytest

from dataflow.analyse import SampleAnalyser

TEST_DATA = Path(__file__).parent.parent.parent / "assets" / "test_data"


class TestSampleAnalyser:
    """Tests for dataset file sampling."""

    # ------------------------------------------------------------------
    # Labels-only mode
    # ------------------------------------------------------------------

    def test_sample_yolo_labels_random(self):
        """YOLO labels-only random sampling collects N .txt files."""
        analyser = SampleAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                count=1,
                seed=42,
                shuffle=True,
                label_dir=TEST_DATA / "det" / "yolo" / "labels",
                class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            )
            assert result.success
            assert result.data is not None

            sr = result.data
            assert sr.sampled_count == 1
            assert sr.total_count == 2
            assert sr.format == "yolo"
            assert sr.mode == "labels"
            assert sr.shuffle is True
            assert sr.seed == 42

            # Check output
            txt_files = [f for f in output_dir.glob("*.txt") if f.name != "classes.txt"]
            assert len(txt_files) == 1

            # Class file should be copied
            assert (output_dir / "classes.txt").exists()

    def test_sample_yolo_labels_sequential(self):
        """YOLO labels-only sequential sampling takes first N files."""
        analyser = SampleAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                count=1,
                shuffle=False,
                label_dir=TEST_DATA / "det" / "yolo" / "labels",
            )
            assert result.success

            sr = result.data
            assert sr.sampled_count == 1
            assert sr.shuffle is False
            assert sr.mode == "labels"

            txt_files = list(output_dir.glob("*.txt"))
            assert len(txt_files) == 1

    def test_sample_labelme_labels(self):
        """LabelMe labels-only sampling collects N .json files."""
        analyser = SampleAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                count=1,
                seed=42,
                label_dir=TEST_DATA / "det" / "labelme",
            )
            assert result.success

            sr = result.data
            assert sr.sampled_count == 1
            assert sr.format == "labelme"
            assert sr.mode == "labels"

            json_files = list(output_dir.glob("*.json"))
            assert len(json_files) == 1

    # ------------------------------------------------------------------
    # Images-only mode
    # ------------------------------------------------------------------

    def test_sample_images_only(self):
        """Images-only sampling collects N image files."""
        analyser = SampleAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                count=1,
                seed=42,
                image_dir=TEST_DATA / "det" / "yolo" / "images",
            )
            assert result.success

            sr = result.data
            assert sr.sampled_count == 1
            assert sr.total_count == 2
            assert sr.format == ""
            assert sr.mode == "images"

            img_files = list(output_dir.glob("*.jpg"))
            assert len(img_files) == 1

    # ------------------------------------------------------------------
    # Both mode
    # ------------------------------------------------------------------

    def test_sample_both(self):
        """Both mode samples labels and matches images by stem."""
        analyser = SampleAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                count=1,
                seed=42,
                label_dir=TEST_DATA / "det" / "yolo" / "labels",
                image_dir=TEST_DATA / "det" / "yolo" / "images",
                class_file=TEST_DATA / "det" / "yolo" / "classes.txt",
            )
            assert result.success

            sr = result.data
            assert sr.sampled_count == 1
            assert sr.total_count == 2
            assert sr.format == "yolo"
            assert sr.mode == "both"

            # Check output structure
            label_subdir = output_dir / "labels"
            image_subdir = output_dir / "images"
            assert label_subdir.is_dir()
            assert image_subdir.is_dir()

            txt_files = list(label_subdir.glob("*.txt"))
            img_files = list(image_subdir.glob("*.jpg"))
            assert len(txt_files) == 1
            assert len(img_files) == 1

            # Files should share the same stem
            label_stem = txt_files[0].stem
            image_stem = img_files[0].stem
            assert label_stem == image_stem

            # Class file should be copied to output_dir
            assert (output_dir / "classes.txt").exists()

    # ------------------------------------------------------------------
    # Shuffle reproducibility
    # ------------------------------------------------------------------

    def test_shuffle_deterministic(self):
        """Same seed should produce the same sample."""

        def sample_once():
            analyser = SampleAnalyser()
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir) / "output"
                result = analyser.analyse(
                    output_dir=output_dir,
                    count=1,
                    seed=42,
                    shuffle=True,
                    label_dir=TEST_DATA / "det" / "yolo" / "labels",
                )
                assert result.success
                return {f.name for f in output_dir.glob("*.txt")}

        files1 = sample_once()
        files2 = sample_once()
        assert files1 == files2

    def test_shuffle_different_seeds(self):
        """Different seeds may produce different samples."""
        analyser = SampleAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_a = Path(tmpdir) / "a"
            out_b = Path(tmpdir) / "b"
            r1 = analyser.analyse(
                output_dir=out_a,
                count=1,
                seed=42,
                shuffle=True,
                label_dir=TEST_DATA / "det" / "yolo" / "labels",
            )
            r2 = analyser.analyse(
                output_dir=out_b,
                count=1,
                seed=99,
                shuffle=True,
                label_dir=TEST_DATA / "det" / "yolo" / "labels",
            )
            assert r1.success
            assert r2.success
            # With only 2 files, seeds can produce same or different.
            # We just verify both succeeded and collected 1 file each.

    # ------------------------------------------------------------------
    # Move mode
    # ------------------------------------------------------------------

    def test_sample_move(self):
        """Move mode relocates files instead of copying."""
        analyser = SampleAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy test data to a temp location so we can safely move it
            import shutil

            src_labels = Path(tmpdir) / "src_labels"
            shutil.copytree(
                str(TEST_DATA / "det" / "yolo" / "labels"),
                str(src_labels),
            )
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                count=2,
                shuffle=False,
                label_dir=src_labels,
                move=True,
            )
            assert result.success

            sr = result.data
            assert sr.sampled_count == 2
            assert sr.move is True

            # Output should have the moved files
            txt_files = list(output_dir.glob("*.txt"))
            assert len(txt_files) == 2

            # Source should be empty
            src_files = list(src_labels.glob("*.txt"))
            assert len(src_files) == 0

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_count_exceeds_available(self):
        """Request count > available files → warning, collects all."""
        analyser = SampleAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                count=100,  # More than available
                shuffle=False,
                label_dir=TEST_DATA / "det" / "yolo" / "labels",
            )
            assert result.success
            assert len(result.warnings) >= 1
            w0 = result.warnings[0].lower()
            assert "100" in w0 or "only" in w0

            sr = result.data
            assert sr.sampled_count == 2  # all 2 available
            assert sr.total_count == 2

            txt_files = list(output_dir.glob("*.txt"))
            assert len(txt_files) == 2

    def test_empty_input(self):
        """Empty directory produces error (format detection fails)."""
        analyser = SampleAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / "empty"
            empty_dir.mkdir()
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                count=5,
                label_dir=empty_dir,
            )
            assert not result.success
            assert "annotation files" in result.errors[0].lower()

    def test_no_input_dir(self):
        """Neither label_dir nor image_dir → error."""
        analyser = SampleAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                count=5,
            )
            assert not result.success
            assert "at least one of" in result.errors[0].lower()

    def test_count_zero_or_negative(self):
        """Count < 1 → error."""
        analyser = SampleAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            result = analyser.analyse(
                output_dir=output_dir,
                count=0,
                label_dir=TEST_DATA / "det" / "yolo" / "labels",
            )
            assert not result.success
            assert "count" in result.errors[0].lower()

    def test_coco_not_supported(self):
        """COCO format → error (not supported)."""
        analyser = SampleAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            # Point to a COCO JSON file
            coco_file = TEST_DATA / "det" / "coco" / "annotations.json"
            if not coco_file.exists():
                pytest.skip("COCO test data not found")
            result = analyser.analyse(
                output_dir=output_dir,
                count=1,
                label_dir=coco_file,
            )
            assert not result.success
            err = result.errors[0].lower() if result.errors else ""
            assert "coco" in err

    # ------------------------------------------------------------------
    # Sequential (deterministic order)
    # ------------------------------------------------------------------

    def test_sequential_same_order(self):
        """Sequential (shuffle=False) always takes the same files."""
        analyser = SampleAnalyser()
        with tempfile.TemporaryDirectory() as tmpdir:
            out1 = Path(tmpdir) / "a"
            out2 = Path(tmpdir) / "b"
            r1 = analyser.analyse(
                output_dir=out1,
                count=1,
                shuffle=False,
                label_dir=TEST_DATA / "det" / "yolo" / "labels",
            )
            r2 = analyser.analyse(
                output_dir=out2,
                count=1,
                shuffle=False,
                label_dir=TEST_DATA / "det" / "yolo" / "labels",
            )
            assert r1.success
            assert r2.success

            files1 = {f.name for f in out1.glob("*.txt")}
            files2 = {f.name for f in out2.glob("*.txt")}
            # Sequential always takes first N by sorted name — identical result
            assert files1 == files2
