"""
Regression tests on realistic COCO fixtures.

Uses a small subset of real COCO val2017 data (multi-category scenes with
small objects and border-truncated boxes, zero-padded file names) to guard
behaviors that synthetic fixtures cannot cover — most importantly the
per-file output naming rule (COCO → YOLO / LabelMe outputs must share the
stem with the image file).
"""

import json
from pathlib import Path

import pytest

from dataflow.convert.coco_and_labelme import CocoAndLabelMeConverter
from dataflow.convert.yolo_and_coco import YoloAndCocoConverter

REALISTIC = Path(__file__).parent.parent.parent / "assets" / "test_data" / "realistic"


@pytest.fixture
def realistic_src():
    """Path to the realistic COCO annotation JSON."""
    return REALISTIC / "annotations.json"


def _image_stems():
    return sorted(p.stem for p in (REALISTIC / "images").glob("*.jpg"))


class TestCocoToYoloRealistic:
    """COCO → YOLO on realistic multi-category data."""

    def test_output_names_share_image_stems(self, realistic_src, tmp_path):
        """Zero-padded file names are preserved in label file names."""
        out = tmp_path / "out"
        result = YoloAndCocoConverter(source_to_target=False).convert(str(realistic_src), str(out))

        assert result.success is True, f"Conversion failed: {result.errors}"
        label_files = sorted(p.name for p in (out / "labels").glob("*.txt"))
        assert label_files == [stem + ".txt" for stem in _image_stems()]

    def test_coordinates_within_unit_interval(self, realistic_src, tmp_path):
        """All converted coordinates stay in [0, 1] (border boxes clamp)."""
        out = tmp_path / "out"
        result = YoloAndCocoConverter(source_to_target=False).convert(str(realistic_src), str(out))
        assert result.success is True

        classes = (out / "classes.txt").read_text(encoding="utf-8").splitlines()
        assert len(classes) > 1  # multi-category scene
        for label_file in (out / "labels").glob("*.txt"):
            for line in label_file.read_text(encoding="utf-8").splitlines():
                tokens = line.split()
                assert int(tokens[0]) < len(classes)
                assert all(0.0 <= float(t) <= 1.0 for t in tokens[1:])

    def test_roundtrip_preserves_counts(self, realistic_src, tmp_path):
        """COCO → YOLO → COCO roundtrip keeps image and object counts."""
        yolo_out = tmp_path / "yolo"
        to_yolo = YoloAndCocoConverter(source_to_target=False)
        r1 = to_yolo.convert(str(realistic_src), str(yolo_out))
        assert r1.success is True, f"COCO→YOLO failed: {r1.errors}"

        expected_objects = sum(
            len(p.read_text(encoding="utf-8").splitlines())
            for p in (yolo_out / "labels").glob("*.txt")
        )

        coco_out = tmp_path / "roundtrip.json"
        to_coco = YoloAndCocoConverter(source_to_target=True, strict_mode=False)
        r2 = to_coco.convert(
            str(yolo_out / "labels"),
            str(coco_out),
            class_file=str(yolo_out / "classes.txt"),
            image_dir=str(REALISTIC / "images"),
        )
        assert r2.success is True, f"YOLO→COCO failed: {r2.errors}"

        data = json.loads(Path(coco_out).read_text(encoding="utf-8"))
        assert len(data["images"]) == len(_image_stems())
        assert len(data["annotations"]) == expected_objects


class TestCocoToLabelMeRealistic:
    """COCO → LabelMe on realistic multi-category data."""

    def test_output_names_share_image_stems(self, realistic_src, tmp_path):
        """LabelMe JSON files are named by the image file stem."""
        out = tmp_path / "out"
        result = CocoAndLabelMeConverter(source_to_target=True).convert(
            str(realistic_src), str(out)
        )

        assert result.success is True, f"Conversion failed: {result.errors}"
        json_files = sorted(p.name for p in out.glob("*.json"))
        assert json_files == [stem + ".json" for stem in _image_stems()]
