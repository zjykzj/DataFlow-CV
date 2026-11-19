"""
Unit tests for pose (keypoint) conversion (P3).

Covers the v3.4.0 contracts (spec_conversion.md §3.7): the six conversion
directions in pose mode, keypoint metadata propagation, k consistency,
visibility degradation (O2), non-pose-object skipping (O3), prediction
mode, and the --save-kpts-config sidecar.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from dataflow.convert import CocoAndLabelMeConverter, LabelMeAndYoloConverter, YoloAndCocoConverter

KPT_NAMES = ["nose", "left_eye", "right_eye"]
SKELETON = [[0, 1], [0, 2]]


def _write_kpts_config(tmp_path: Path) -> Path:
    config = tmp_path / "kpts.yaml"
    config.write_text(
        "keypoints:\n  - nose\n  - left_eye\n  - right_eye\nskeleton:\n  - [0, 1]\n  - [0, 2]\n",
        encoding="utf-8",
    )
    return config


def _make_coco_pose(tmp_path: Path, with_kpts: bool = True, k: int = 3) -> Path:
    """Minimal COCO pose dataset: one 300x200 image, one person."""
    kpts = [130, 120, 2, 125, 115, 1, 0, 0, 0] if with_kpts else []
    data = {
        "images": [{"id": 1, "file_name": "test.jpg", "width": 300, "height": 200}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100.0, 100.0, 60.0, 80.0],
                "segmentation": [],
                "area": 4800.0,
                "iscrowd": 0,
                "keypoints": kpts,
                "num_keypoints": 2,
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "none",
                "keypoints": KPT_NAMES,
                "skeleton": SKELETON,
            }
        ],
    }
    if not with_kpts:
        del data["annotations"][0]["keypoints"]
        del data["annotations"][0]["num_keypoints"]
    coco_file = tmp_path / "pose.json"
    coco_file.write_text(json.dumps(data))
    return coco_file


class TestPoseConversionBase:
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp(prefix="test_pose_convert_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def _make_classes(self, tmp_path: Path) -> Path:
        class_file = tmp_path / "classes.txt"
        class_file.write_text("person\n")
        return class_file


class TestCocoToYoloPose(TestPoseConversionBase):
    def test_pose_roundtrip(self, temp_dir):
        """COCO → YOLO pose: keypoints normalized, visibility passthrough,
        seg dropped, 5+3k rows written."""
        coco = _make_coco_pose(temp_dir)
        class_file = self._make_classes(temp_dir)
        image_dir = temp_dir / "images"
        image_dir.mkdir()

        converter = YoloAndCocoConverter(source_to_target=False)
        result = converter.convert(
            str(coco),
            str(temp_dir / "out"),
            class_file=str(class_file),
            image_dir=str(image_dir),
            pose=True,
        )
        assert result.success is True
        line = (temp_dir / "out" / "labels" / "test.txt").read_text(encoding="utf-8")
        tokens = line.split()
        assert len(tokens) == 14  # 5 + 3x3
        # bbox normalized: center (130,140) of 300x200
        assert float(tokens[1]) == pytest.approx(130 / 300, abs=1e-4)
        # keypoint 0: (130, 120) normalized, v=2
        assert float(tokens[5]) == pytest.approx(130 / 300, abs=1e-4)
        assert float(tokens[6]) == pytest.approx(120 / 200, abs=1e-4)
        assert tokens[7] == "2"
        # keypoint 2: v=0 stays (0, 0)
        assert tokens[11:14] == ["0.000000", "0.000000", "0"]

    def test_sidecar_generation(self, temp_dir):
        """--save-kpts-config writes a definition YAML from COCO categories."""
        coco = _make_coco_pose(temp_dir)
        class_file = self._make_classes(temp_dir)
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        sidecar = temp_dir / "generated_kpts.yaml"

        converter = YoloAndCocoConverter(source_to_target=False)
        result = converter.convert(
            str(coco),
            str(temp_dir / "out"),
            class_file=str(class_file),
            image_dir=str(image_dir),
            pose=True,
            save_kpts_config=str(sidecar),
        )
        assert result.success is True
        assert sidecar.exists()
        from dataflow.label.utils import load_kpts_config

        cfg = load_kpts_config(sidecar)
        assert cfg.names_for("person") == KPT_NAMES
        assert cfg.skeleton == SKELETON

    def test_non_pose_objects_skipped_in_pose_mode(self, temp_dir):
        """O3: pose mode skips objects without keypoints with a warning."""
        coco = temp_dir / "mixed.json"
        data = json.loads(_make_coco_pose(temp_dir).read_text(encoding="utf-8"))
        # Second object without keypoints
        data["annotations"].append(
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "segmentation": [],
                "area": 400.0,
                "iscrowd": 0,
            }
        )
        coco.write_text(json.dumps(data))
        class_file = self._make_classes(temp_dir)
        image_dir = temp_dir / "images"
        image_dir.mkdir()

        converter = YoloAndCocoConverter(source_to_target=False)
        result = converter.convert(
            str(coco),
            str(temp_dir / "out"),
            class_file=str(class_file),
            image_dir=str(image_dir),
            pose=True,
        )
        assert result.success is True
        line = (temp_dir / "out" / "labels" / "test.txt").read_text(encoding="utf-8")
        assert len(line.split()) == 14  # only the pose object written


class TestYoloToCocoPose(TestPoseConversionBase):
    @pytest.fixture
    def sample_pose_labels(self, temp_dir):
        import cv2
        import numpy as np

        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\n")
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(image_dir / "test.jpg"), img)
        label_dir = temp_dir / "labels"
        label_dir.mkdir()
        # 3-keypoint pose rows (14 tokens)
        (label_dir / "test.txt").write_text(
            "0 0.433333 0.700000 0.200000 0.400000 "
            "0.433333 0.600000 2 0.416667 0.575000 1 0.900000 0.900000 0\n"
        )
        return {"class_file": class_file, "image_dir": image_dir, "label_dir": label_dir}

    def test_pose_with_kpts_config(self, sample_pose_labels, temp_dir):
        """YOLO → COCO pose: keypoints denormalized, num_keypoints computed,
        categories get names + skeleton from the config."""
        config = _write_kpts_config(temp_dir)
        converter = YoloAndCocoConverter(source_to_target=True)
        result = converter.convert(
            str(sample_pose_labels["label_dir"]),
            str(temp_dir / "out.json"),
            class_file=str(sample_pose_labels["class_file"]),
            image_dir=str(sample_pose_labels["image_dir"]),
            pose=True,
            kpts_config=str(config),
        )
        assert result.success is True
        written = json.loads((temp_dir / "out.json").read_text(encoding="utf-8"))
        ann = written["annotations"][0]
        assert ann["keypoints"][0] == pytest.approx(130.0, abs=1e-3)
        assert ann["keypoints"][1] == pytest.approx(120.0, abs=1e-3)
        assert ann["keypoints"][2] == 2
        assert ann["keypoints"][3] == pytest.approx(125.0, abs=1e-3)
        assert ann["keypoints"][4] == pytest.approx(115.0, abs=1e-3)
        assert ann["keypoints"][5] == 1
        assert ann["keypoints"][6:9] == [0.0, 0.0, 0]
        assert ann["num_keypoints"] == 2
        cat = written["categories"][0]
        assert cat["keypoints"] == KPT_NAMES
        assert cat["skeleton"] == SKELETON

    def test_pose_without_config_uses_placeholders(self, sample_pose_labels, temp_dir):
        """Without kpts_config: placeholder names + no skeleton."""
        converter = YoloAndCocoConverter(source_to_target=True)
        result = converter.convert(
            str(sample_pose_labels["label_dir"]),
            str(temp_dir / "out.json"),
            class_file=str(sample_pose_labels["class_file"]),
            image_dir=str(sample_pose_labels["image_dir"]),
            pose=True,
        )
        assert result.success is True
        written = json.loads((temp_dir / "out.json").read_text(encoding="utf-8"))
        cat = written["categories"][0]
        assert cat["keypoints"] == ["keypoint_0", "keypoint_1", "keypoint_2"]
        assert "skeleton" not in cat

    def test_pose_prediction(self, sample_pose_labels, temp_dir):
        """O5: yolo2coco --prediction --pose → Variant B list with keypoints
        and score."""
        (sample_pose_labels["label_dir"] / "test.txt").write_text(
            "0 0.433333 0.700000 0.200000 0.400000 "
            "0.433333 0.600000 2 0.416667 0.575000 1 0.900000 0.900000 0 0.85\n"
        )
        converter = YoloAndCocoConverter(source_to_target=True, prediction=True)
        result = converter.convert(
            str(sample_pose_labels["label_dir"]),
            str(temp_dir / "pred.json"),
            class_file=str(sample_pose_labels["class_file"]),
            image_dir=str(sample_pose_labels["image_dir"]),
            pose=True,
        )
        assert result.success is True
        written = json.loads((temp_dir / "pred.json").read_text(encoding="utf-8"))
        assert isinstance(written, list)
        ann = written[0]
        assert ann["score"] == pytest.approx(0.85)
        assert len(ann["keypoints"]) == 9
        assert ann["num_keypoints"] == 2


class TestCocoToLabelMePose(TestPoseConversionBase):
    def test_pose_writes_rect_plus_points(self, temp_dir):
        """COCO → LabelMe pose: rectangle + named point shapes; v=1 degrades
        to v=2 with a warning (O2)."""
        coco = _make_coco_pose(temp_dir)
        converter = CocoAndLabelMeConverter(source_to_target=True)
        result = converter.convert(
            str(coco),
            str(temp_dir / "out"),
            pose=True,
        )
        assert result.success is True
        written = json.loads((temp_dir / "out" / "test.json").read_text(encoding="utf-8"))
        rects = [s for s in written["shapes"] if s["shape_type"] == "rectangle"]
        points = [s for s in written["shapes"] if s["shape_type"] == "point"]
        assert len(rects) == 1
        # v=1 (left_eye) and v=2 (nose) are both written; v=0 skipped
        assert len(points) == 2
        labels = sorted(s["label"] for s in points)
        assert labels == ["left_eye", "nose"]
        gids = {s["group_id"] for s in written["shapes"]}
        assert gids == {1}


class TestLabelMeToCocoPose(TestPoseConversionBase):
    @pytest.fixture
    def sample_pose_dir(self, temp_dir):
        label_dir = temp_dir / "labels"
        label_dir.mkdir()
        data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [
                {
                    "label": "person",
                    "points": [[100, 100], [160, 180]],
                    "group_id": 1,
                    "shape_type": "rectangle",
                    "flags": {},
                },
                {
                    "label": "nose",
                    "points": [[130, 120]],
                    "group_id": 1,
                    "shape_type": "point",
                    "flags": {},
                },
                {
                    "label": "left_eye",
                    "points": [[125, 115]],
                    "group_id": 1,
                    "shape_type": "point",
                    "flags": {},
                },
            ],
            "imagePath": "test.jpg",
            "imageData": None,
            "imageHeight": 200,
            "imageWidth": 300,
        }
        (label_dir / "test.json").write_text(json.dumps(data))
        return label_dir

    def test_pose_with_kpts_config(self, sample_pose_dir, temp_dir):
        """LabelMe → COCO pose: config orders keypoints (missing → v=0) and
        supplies COCO category names + skeleton."""
        config = _write_kpts_config(temp_dir)
        class_file = self._make_classes(temp_dir)
        converter = CocoAndLabelMeConverter(source_to_target=False)
        result = converter.convert(
            str(sample_pose_dir),
            str(temp_dir / "out.json"),
            class_file=str(class_file),
            pose=True,
            kpts_config=str(config),
        )
        assert result.success is True
        written = json.loads((temp_dir / "out.json").read_text(encoding="utf-8"))
        ann = written["annotations"][0]
        assert ann["keypoints"] == [130.0, 120.0, 2, 125.0, 115.0, 2, 0.0, 0.0, 0]
        assert ann["num_keypoints"] == 2
        assert written["categories"][0]["keypoints"] == KPT_NAMES
        assert written["categories"][0]["skeleton"] == SKELETON


class TestLabelMeYoloPose(TestPoseConversionBase):
    def test_labelme_to_yolo_pose(self, temp_dir):
        """LabelMe → YOLO pose: 5+3k rows from point+rectangle instances."""
        label_dir = temp_dir / "labels"
        label_dir.mkdir()
        data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [
                {
                    "label": "person",
                    "points": [[100, 100], [160, 180]],
                    "group_id": 1,
                    "shape_type": "rectangle",
                    "flags": {},
                },
                {
                    "label": "nose",
                    "points": [[130, 120]],
                    "group_id": 1,
                    "shape_type": "point",
                    "flags": {},
                },
            ],
            "imagePath": "test.jpg",
            "imageData": None,
            "imageHeight": 200,
            "imageWidth": 300,
        }
        (label_dir / "test.json").write_text(json.dumps(data))
        config = _write_kpts_config(temp_dir)
        class_file = self._make_classes(temp_dir)

        converter = LabelMeAndYoloConverter(source_to_target=True)
        result = converter.convert(
            str(label_dir),
            str(temp_dir / "out"),
            class_file=str(class_file),
            pose=True,
            kpts_config=str(config),
        )
        assert result.success is True
        line = (temp_dir / "out" / "labels" / "test.txt").read_text(encoding="utf-8")
        tokens = line.split()
        assert len(tokens) == 14
        # keypoint 0 = nose (130, 120) normalized, v=2
        assert float(tokens[5]) == pytest.approx(130 / 300, abs=1e-4)
        assert float(tokens[6]) == pytest.approx(120 / 200, abs=1e-4)
        assert tokens[7] == "2"
        # keypoint 2 = right_eye missing → v=0 at (0, 0)
        assert tokens[11:14] == ["0.000000", "0.000000", "0"]

    def test_yolo_to_labelme_pose(self, temp_dir):
        """YOLO → LabelMe pose: point shapes named from the config."""
        import cv2
        import numpy as np

        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\n")
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(image_dir / "test.jpg"), img)
        label_dir = temp_dir / "labels"
        label_dir.mkdir()
        (label_dir / "test.txt").write_text(
            "0 0.433333 0.700000 0.200000 0.400000 "
            "0.433333 0.600000 2 0.416667 0.575000 1 0.900000 0.900000 0\n"
        )
        config = _write_kpts_config(temp_dir)

        converter = LabelMeAndYoloConverter(source_to_target=False)
        result = converter.convert(
            str(label_dir),
            str(temp_dir / "out"),
            class_file=str(class_file),
            image_dir=str(image_dir),
            pose=True,
            kpts_config=str(config),
        )
        assert result.success is True
        written = json.loads((temp_dir / "out" / "test.json").read_text(encoding="utf-8"))
        points = [s for s in written["shapes"] if s["shape_type"] == "point"]
        labels = sorted(s["label"] for s in points)
        assert labels == ["left_eye", "nose"]  # v=0 skipped
        rects = [s for s in written["shapes"] if s["shape_type"] == "rectangle"]
        assert len(rects) == 1


class TestPoseKConsistency(TestPoseConversionBase):
    def test_inconsistent_k_strict_errors(self, temp_dir):
        """Converter-level k consistency: mismatch is an error in strict mode."""
        coco = temp_dir / "mixed_k.json"
        data = json.loads(_make_coco_pose(temp_dir).read_text(encoding="utf-8"))
        data["annotations"].append(
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "segmentation": [],
                "area": 400.0,
                "iscrowd": 0,
                "keypoints": [15, 15, 2, 0, 0, 0],  # k=2 != 3
                "num_keypoints": 1,
            }
        )
        coco.write_text(json.dumps(data))
        class_file = self._make_classes(temp_dir)
        image_dir = temp_dir / "images"
        image_dir.mkdir()

        converter = YoloAndCocoConverter(source_to_target=False)
        result = converter.convert(
            str(coco),
            str(temp_dir / "out"),
            class_file=str(class_file),
            image_dir=str(image_dir),
            pose=True,
        )
        assert result.success is False
        assert "Inconsistent keypoint count" in (result.errors or [""])[0]

    def test_inconsistent_k_non_strict_skips(self, temp_dir):
        """Non-strict mode: the mismatched object is skipped."""
        coco = temp_dir / "mixed_k.json"
        data = json.loads(_make_coco_pose(temp_dir).read_text(encoding="utf-8"))
        data["annotations"].append(
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "segmentation": [],
                "area": 400.0,
                "iscrowd": 0,
                "keypoints": [15, 15, 2, 0, 0, 0],
                "num_keypoints": 1,
            }
        )
        coco.write_text(json.dumps(data))
        class_file = self._make_classes(temp_dir)
        image_dir = temp_dir / "images"
        image_dir.mkdir()

        converter = YoloAndCocoConverter(source_to_target=False, strict_mode=False)
        result = converter.convert(
            str(coco),
            str(temp_dir / "out"),
            class_file=str(class_file),
            image_dir=str(image_dir),
            pose=True,
        )
        assert result.success is True
        line = (temp_dir / "out" / "labels" / "test.txt").read_text(encoding="utf-8")
        assert len(line.split()) == 14  # only the k=3 object written


class TestConvertCLIPoseFlags(TestPoseConversionBase):
    def test_convert_help_shows_pose_options(self):
        from click.testing import CliRunner

        from dataflow.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["convert", "yolo2coco", "--help"])
        assert result.exit_code == 0
        assert "--pose" in result.output
        assert "--kpts-config" in result.output

    def test_coco2yolo_help_shows_sidecar_option(self):
        from click.testing import CliRunner

        from dataflow.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["convert", "coco2yolo", "--help"])
        assert result.exit_code == 0
        assert "--pose" in result.output
        assert "--save-kpts-config" in result.output
