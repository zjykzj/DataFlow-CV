"""
Unit tests for pose (keypoint) support in the Label module.

Covers the P1 contracts: Keypoints data model, kpts_config loader,
YOLO/LabelMe/COCO pose read/write, and the converter pose guard (D8).
"""

import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from dataflow.label.coco_handler import CocoAnnotationHandler
from dataflow.label.labelme_handler import LabelMeAnnotationHandler
from dataflow.label.models import (
    Keypoints,
    ObjectAnnotation,
)
from dataflow.label.utils import KptsConfig, load_kpts_config
from dataflow.label.yolo_handler import YoloAnnotationHandler


KPT_NAMES = ["nose", "left_eye", "right_eye"]


def _make_kpts_config(tmp_path: Path, extra: str = "") -> Path:
    """Write a minimal 3-keypoint definition YAML."""
    config = tmp_path / "kpts.yaml"
    config.write_text(
        "keypoints:\n  - nose\n  - left_eye\n  - right_eye\n" + extra,
        encoding="utf-8",
    )
    return config


class TestKeypointsModel:
    """Keypoints dataclass invariants (spec_label.md §2.6)."""

    def test_valid_construction(self):
        kpts = Keypoints(points=[(1.0, 2.0), (3.0, 4.0)], visibility=[2, 1])
        assert kpts.points == [(1.0, 2.0), (3.0, 4.0)]
        assert kpts.visibility == [2, 1]
        assert kpts.count_labeled() == 2

    def test_count_labeled_counts_v_gt_0(self):
        kpts = Keypoints(points=[(1, 1), (2, 2), (3, 3)], visibility=[2, 0, 1])
        assert kpts.count_labeled() == 2

    def test_empty_points_rejected(self):
        with pytest.raises(ValueError, match="at least 1 point"):
            Keypoints(points=[], visibility=[])

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="length mismatch"):
            Keypoints(points=[(1, 1)], visibility=[2, 2])

    def test_invalid_visibility_rejected(self):
        with pytest.raises(ValueError, match="must be 0, 1, or 2"):
            Keypoints(points=[(1, 1)], visibility=[3])

    def test_object_annotation_accepts_keypoints_only(self):
        obj = ObjectAnnotation(
            class_id=0, class_name="person", keypoints=Keypoints(points=[(1, 1)], visibility=[2])
        )
        assert obj.bbox is None
        assert obj.segmentation is None
        assert obj.keypoints is not None

    def test_object_annotation_requires_one_field(self):
        with pytest.raises(ValueError, match="keypoints must be provided"):
            ObjectAnnotation(class_id=0, class_name="ghost")


class TestKptsConfigLoader:
    """Keypoint-definition YAML loader (spec_label.md §4.4)."""

    def test_global_keypoint_list(self, tmp_path):
        config = _make_kpts_config(tmp_path)
        kpts = load_kpts_config(config)
        assert isinstance(kpts, KptsConfig)
        assert kpts.names_for("anything") == KPT_NAMES
        assert kpts.skeleton is None
        assert kpts.sigmas is None

    def test_per_class_overrides_global(self, tmp_path):
        config = tmp_path / "kpts.yaml"
        config.write_text(
            "keypoints:\n  - nose\n  - left_eye\n  - right_eye\n"
            "classes:\n  bird:\n    - beak\n    - left_wing\n",
            encoding="utf-8",
        )
        kpts = load_kpts_config(config)
        assert kpts.names_for("person") == KPT_NAMES
        assert kpts.names_for("bird") == ["beak", "left_wing"]

    def test_x_anylabeling_style_config(self, tmp_path):
        """has_visible + per-class structure loads unchanged."""
        config = tmp_path / "yolov8_pose.yaml"
        config.write_text(
            "has_visible: true\nclasses:\n  person:\n    - nose\n    - left_eye\n    - right_eye\n",
            encoding="utf-8",
        )
        kpts = load_kpts_config(config)
        assert kpts.names_for("person") == KPT_NAMES
        with pytest.raises(ValueError, match="No keypoint names defined"):
            kpts.names_for("car")

    def test_skeleton_and_sigmas(self, tmp_path):
        config = tmp_path / "kpts.yaml"
        config.write_text(
            "keypoints:\n  - nose\n  - left_eye\n  - right_eye\n"
            "skeleton:\n  - [0, 1]\n  - [0, 2]\n"
            "sigmas:\n  - 0.026\n  - 0.025\n  - 0.025\n",
            encoding="utf-8",
        )
        kpts = load_kpts_config(config)
        assert kpts.skeleton == [[0, 1], [0, 2]]
        assert kpts.sigmas == [0.026, 0.025, 0.025]

    def test_missing_keypoints_and_classes_rejected(self, tmp_path):
        config = tmp_path / "kpts.yaml"
        config.write_text("has_visible: true\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain 'keypoints' or 'classes'"):
            load_kpts_config(config)

    def test_duplicate_names_rejected(self, tmp_path):
        config = tmp_path / "kpts.yaml"
        config.write_text("keypoints:\n  - nose\n  - nose\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Duplicate keypoint name"):
            load_kpts_config(config)

    def test_skeleton_out_of_range_rejected(self, tmp_path):
        config = _make_kpts_config(tmp_path, extra="skeleton:\n  - [0, 5]\n")
        with pytest.raises(ValueError, match="index out of range"):
            load_kpts_config(config)

    def test_sigmas_length_mismatch_rejected(self, tmp_path):
        config = _make_kpts_config(tmp_path, extra="sigmas:\n  - 0.026\n")
        with pytest.raises(ValueError, match="must have 3 values"):
            load_kpts_config(config)

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            load_kpts_config(tmp_path / "nope.yaml")


class TestYoloPose:
    """YOLO pose row read/write (spec_yolo_format.md §2.6)."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_pose_data(self, temp_dir):
        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\n")
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        label_dir = temp_dir / "labels"
        label_dir.mkdir()
        img = np.zeros((100, 200, 3), dtype=np.uint8)  # 200x100 image
        cv2.imwrite(str(image_dir / "test.jpg"), img)
        # 3-keypoint pose rows (14 tokens): kpt2 has v=0 at non-zero coords
        label_file = label_dir / "test.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.2 0.48 0.45 2 0.52 0.45 1 0.9 0.9 0\n")
        return {
            "class_file": class_file,
            "image_dir": image_dir,
            "label_dir": label_dir,
        }

    def _make_handler(self, data, **kwargs):
        return YoloAnnotationHandler(
            label_dir=str(data["label_dir"]),
            class_file=str(data["class_file"]),
            image_dir=str(data["image_dir"]),
            pose=True,
            **kwargs,
        )

    def test_read_pose_row(self, sample_pose_data):
        handler = self._make_handler(sample_pose_data)
        result = handler.read()
        assert result.success is True
        dataset = result.data
        assert dataset.num_objects == 1
        obj = dataset.images[0].objects[0]
        assert obj.class_id == 0
        assert obj.bbox is not None
        assert obj.bbox.x == pytest.approx(0.5)
        assert obj.keypoints is not None
        assert len(obj.keypoints.points) == 3
        assert obj.keypoints.visibility == [2, 1, 0]
        # v=0 keypoints are normalized to (0, 0)
        assert obj.keypoints.points[2] == (0.0, 0.0)
        assert obj.keypoints.count_labeled() == 2
        assert handler.is_pose is True

    def test_read_pose_prediction_row(self, sample_pose_data):
        handler = self._make_handler(sample_pose_data, prediction=True)
        (sample_pose_data["label_dir"] / "test.txt").write_text(
            "0 0.5 0.5 0.2 0.2 0.48 0.45 2 0.52 0.45 1 0.9 0.9 0 0.85\n"
        )
        result = handler.read()
        assert result.success is True
        obj = result.data.images[0].objects[0]
        assert obj.confidence == pytest.approx(0.85)
        assert len(obj.keypoints.points) == 3

    def test_inconsistent_k_strict_errors(self, sample_pose_data):
        (sample_pose_data["label_dir"] / "test.txt").write_text(
            "0 0.5 0.5 0.2 0.2 0.48 0.45 2 0.52 0.45 1 0.9 0.9 0\n"
            "0 0.5 0.5 0.2 0.2 0.48 0.45 2\n"  # 2-keypoint row (11 tokens)
        )
        handler = self._make_handler(sample_pose_data, strict_mode=True)
        result = handler.read()
        assert result.success is False
        assert "Inconsistent keypoint count" in result.message

    def test_inconsistent_k_non_strict_skips(self, sample_pose_data):
        (sample_pose_data["label_dir"] / "test.txt").write_text(
            "0 0.5 0.5 0.2 0.2 0.48 0.45 2 0.52 0.45 1 0.9 0.9 0\n0 0.5 0.5 0.2 0.2 0.48 0.45 2\n"
        )
        handler = self._make_handler(sample_pose_data, strict_mode=False)
        result = handler.read()
        assert result.success is True
        assert result.data.num_objects == 1

    def test_even_k_pose_row_misparsed_as_seg_without_pose(self, sample_pose_data):
        """Det/seg mode cannot see pose rows: an 11-token row (2-keypoint
        pose) silently parses as a 5-point segmentation (contract:
        spec_yolo_format.md §2.3)."""
        (sample_pose_data["label_dir"] / "test.txt").write_text(
            "0 0.5 0.5 0.2 0.2 0.48 0.45 0 0.52 0.45 1\n"
        )
        handler = YoloAnnotationHandler(
            label_dir=str(sample_pose_data["label_dir"]),
            class_file=str(sample_pose_data["class_file"]),
            image_dir=str(sample_pose_data["image_dir"]),
            strict_mode=True,
        )
        result = handler.read()
        assert result.success is True
        obj = result.data.images[0].objects[0]
        assert obj.keypoints is None
        assert obj.segmentation is not None
        assert len(obj.segmentation.points) == 5

    def test_odd_k_pose_row_rejected_without_pose(self, sample_pose_data):
        """A 17-keypoint pose row (56 tokens, even) is invalid in det/seg
        label mode and rejected — never silently misparsed (contract:
        spec_yolo_format.md §2.3)."""
        tokens = ["0", "0.5", "0.5", "0.2", "0.2"]
        tokens += ["0.5", "0.5", "2"] * 17
        (sample_pose_data["label_dir"] / "test.txt").write_text(" ".join(tokens) + "\n")
        handler = YoloAnnotationHandler(
            label_dir=str(sample_pose_data["label_dir"]),
            class_file=str(sample_pose_data["class_file"]),
            image_dir=str(sample_pose_data["image_dir"]),
            strict_mode=True,
        )
        result = handler.read()
        assert result.success is False
        assert "Invalid YOLO label format" in result.message

    def test_det_row_invalid_in_pose_mode(self, sample_pose_data):
        (sample_pose_data["label_dir"] / "test.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        handler = self._make_handler(sample_pose_data, strict_mode=True)
        result = handler.read()
        assert result.success is False
        assert "pose" in result.message

    def test_invalid_visibility_strict_errors(self, sample_pose_data):
        (sample_pose_data["label_dir"] / "test.txt").write_text(
            "0 0.5 0.5 0.2 0.2 0.48 0.45 5 0.52 0.45 1 0.9 0.9 0\n"
        )
        handler = self._make_handler(sample_pose_data, strict_mode=True)
        result = handler.read()
        assert result.success is False
        assert "Invalid visibility" in result.message

    def test_write_pose_row_roundtrip(self, sample_pose_data, temp_dir):
        handler = self._make_handler(sample_pose_data)
        result = handler.read()
        assert result.success is True
        dataset = result.data

        out_dir = temp_dir / "out"
        write_result = handler.write(dataset, str(out_dir))
        assert write_result.success is True
        line = (out_dir / "test.txt").read_text(encoding="utf-8").strip()
        assert line.startswith("0 0.500000 0.500000 0.200000 0.200000")
        # 14 tokens: 5 bbox + 3 x 3 keypoint triplets
        assert len(line.split()) == 14

        # Read back — keypoints preserved
        handler2 = YoloAnnotationHandler(
            label_dir=str(out_dir),
            class_file=str(sample_pose_data["class_file"]),
            image_dir=str(sample_pose_data["image_dir"]),
            pose=True,
        )
        result2 = handler2.read()
        assert result2.success is True
        obj2 = result2.data.images[0].objects[0]
        assert obj2.keypoints.visibility == [2, 1, 0]

    def test_write_non_pose_mode_skips_keypoints(self, sample_pose_data, temp_dir):
        handler = self._make_handler(sample_pose_data)
        dataset = handler.read().data
        det_handler = YoloAnnotationHandler(
            label_dir=str(sample_pose_data["label_dir"]),
            class_file=str(sample_pose_data["class_file"]),
            image_dir=str(sample_pose_data["image_dir"]),
            strict_mode=False,
        )
        out_dir = temp_dir / "out"
        result = det_handler.write(dataset, str(out_dir))
        assert result.success is True
        # The keypoints object is skipped — empty file
        assert (out_dir / "test.txt").read_text(encoding="utf-8") == ""

    def test_validate_pose_file(self, sample_pose_data):
        handler = self._make_handler(sample_pose_data)
        assert handler.validate(str(sample_pose_data["label_dir"] / "test.txt")) is True

    def test_validate_rejects_bad_visibility(self, sample_pose_data):
        handler = self._make_handler(sample_pose_data)
        (sample_pose_data["label_dir"] / "bad.txt").write_text(
            "0 0.5 0.5 0.2 0.2 0.48 0.45 7 0.52 0.45 1 0.9 0.9 0\n"
        )
        assert handler.validate(str(sample_pose_data["label_dir"] / "bad.txt")) is False

    def test_validate_rejects_det_row(self, sample_pose_data):
        handler = self._make_handler(sample_pose_data)
        (sample_pose_data["label_dir"] / "bad.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        assert handler.validate(str(sample_pose_data["label_dir"] / "bad.txt")) is False


class TestLabelMePose:
    """LabelMe point+group_id pose convention (spec_labelme_format.md §3.5)."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_pose_data(self, temp_dir):
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
        (label_dir / "test.json").write_text(json.dumps(data), encoding="utf-8")
        return {"label_dir": label_dir, "config": _make_kpts_config(temp_dir)}

    def _make_handler(self, data, **kwargs):
        return LabelMeAnnotationHandler(
            label_dir=str(data["label_dir"]),
            class_file=None,
            **kwargs,
        )

    def test_read_with_config_orders_and_pads(self, sample_pose_data):
        handler = self._make_handler(sample_pose_data, kpts_config=str(sample_pose_data["config"]))
        result = handler.read()
        assert result.success is True
        dataset = result.data
        assert dataset.num_objects == 1
        obj = dataset.images[0].objects[0]
        assert obj.class_name == "person"
        assert obj.bbox is not None
        assert obj.bbox.x == pytest.approx(100)
        # Config order: nose, left_eye, right_eye — right_eye missing → v=0
        assert obj.keypoints is not None
        assert obj.keypoints.points[0] == (130.0, 120.0)
        assert obj.keypoints.visibility == [2, 2, 0]
        assert obj.keypoints.points[2] == (0.0, 0.0)

    def test_keypoint_labels_do_not_pollute_categories(self, sample_pose_data):
        handler = self._make_handler(sample_pose_data, kpts_config=str(sample_pose_data["config"]))
        result = handler.read()
        assert result.success is True
        assert "nose" not in result.data.categories.values()
        assert "person" in result.data.categories.values()

    def test_iter_images_does_not_pollute_categories(self, sample_pose_data):
        handler = self._make_handler(sample_pose_data, kpts_config=str(sample_pose_data["config"]))
        for image_ann in handler.iter_images():
            assert image_ann.objects[0].keypoints is not None
        assert "nose" not in handler.categories.values()
        assert "person" in handler.categories.values()

    def test_read_without_config_uses_file_order(self, sample_pose_data):
        handler = self._make_handler(sample_pose_data)
        result = handler.read()
        assert result.success is True
        obj = result.data.images[0].objects[0]
        assert obj.keypoints is not None
        assert len(obj.keypoints.points) == 2
        assert obj.keypoints.points[0] == (130.0, 120.0)  # nose first in file
        assert obj.keypoints.visibility == [2, 2]

    def test_unknown_keypoint_label_strict_errors(self, sample_pose_data, temp_dir):
        data = json.loads((sample_pose_data["label_dir"] / "test.json").read_text(encoding="utf-8"))
        data["shapes"].append(
            {
                "label": "tail",
                "points": [[200, 200]],
                "group_id": 1,
                "shape_type": "point",
                "flags": {},
            }
        )
        (sample_pose_data["label_dir"] / "test.json").write_text(json.dumps(data))
        handler = self._make_handler(
            sample_pose_data, kpts_config=str(sample_pose_data["config"]), strict_mode=True
        )
        result = handler.read()
        assert result.success is False
        assert "not in the definition file" in result.message

    def test_point_without_group_id_errors(self, sample_pose_data):
        data = json.loads((sample_pose_data["label_dir"] / "test.json").read_text(encoding="utf-8"))
        data["shapes"].append(
            {
                "label": "lonely",
                "points": [[10, 10]],
                "group_id": None,
                "shape_type": "point",
                "flags": {},
            }
        )
        (sample_pose_data["label_dir"] / "test.json").write_text(json.dumps(data))
        handler = self._make_handler(sample_pose_data, strict_mode=True)
        result = handler.read()
        assert result.success is False
        assert "Unsupported shape type" in result.message

    def test_write_pose_shapes_with_group_id(self, sample_pose_data, temp_dir):
        handler = self._make_handler(sample_pose_data, kpts_config=str(sample_pose_data["config"]))
        dataset = handler.read().data

        out_dir = temp_dir / "out"
        write_result = handler.write(dataset, str(out_dir))
        assert write_result.success is True

        written = json.loads((out_dir / "test.json").read_text(encoding="utf-8"))
        point_shapes = [s for s in written["shapes"] if s["shape_type"] == "point"]
        rect_shapes = [s for s in written["shapes"] if s["shape_type"] == "rectangle"]
        assert len(rect_shapes) == 1
        assert len(point_shapes) == 2  # v=0 keypoint skipped
        gids = {s["group_id"] for s in written["shapes"]}
        assert gids == {1}
        labels = sorted(s["label"] for s in point_shapes)
        assert labels == ["left_eye", "nose"]

    def test_write_without_config_uses_placeholders(self, sample_pose_data, temp_dir):
        handler = self._make_handler(sample_pose_data)
        dataset = handler.read().data
        out_dir = temp_dir / "out"
        write_result = handler.write(dataset, str(out_dir))
        assert write_result.success is True
        written = json.loads((out_dir / "test.json").read_text(encoding="utf-8"))
        point_shapes = [s for s in written["shapes"] if s["shape_type"] == "point"]
        labels = sorted(s["label"] for s in point_shapes)
        assert labels == ["keypoint_0", "keypoint_1"]

    def test_validate_accepts_point_with_group_id(self, sample_pose_data):
        handler = self._make_handler(sample_pose_data)
        assert handler.validate(str(sample_pose_data["label_dir"] / "test.json")) is True

    def test_validate_rejects_point_without_group_id(self, sample_pose_data):
        data = json.loads((sample_pose_data["label_dir"] / "test.json").read_text(encoding="utf-8"))
        data["shapes"].append(
            {
                "label": "lonely",
                "points": [[10, 10]],
                "group_id": None,
                "shape_type": "point",
                "flags": {},
            }
        )
        (sample_pose_data["label_dir"] / "bad.json").write_text(json.dumps(data))
        handler = self._make_handler(sample_pose_data)
        assert handler.validate(str(sample_pose_data["label_dir"] / "bad.json")) is False


class TestCocoPose:
    """COCO keypoints field read/write (spec_coco_format.md §4.5)."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_coco_pose(self, temp_dir):
        coco_file = temp_dir / "pose.json"
        data = {
            "info": {"description": "pose test"},
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
                    "keypoints": [130, 120, 2, 125, 115, 1, 999, 999, 0],
                    "num_keypoints": 2,
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "none",
                    "keypoints": KPT_NAMES,
                    "skeleton": [[0, 1], [0, 2]],
                }
            ],
        }
        coco_file.write_text(json.dumps(data), encoding="utf-8")
        return coco_file

    def test_read_keypoints(self, sample_coco_pose):
        handler = CocoAnnotationHandler(annotation_file=str(sample_coco_pose))
        result = handler.read()
        assert result.success is True
        obj = result.data.images[0].objects[0]
        assert obj.keypoints is not None
        assert obj.keypoints.visibility == [2, 1, 0]
        # v=0 keypoints are normalized to (0, 0)
        assert obj.keypoints.points[2] == (0.0, 0.0)
        assert obj.keypoints.count_labeled() == 2
        assert handler.is_pose is True

    def test_read_preserves_categories_meta(self, sample_coco_pose):
        handler = CocoAnnotationHandler(annotation_file=str(sample_coco_pose))
        result = handler.read()
        assert result.success is True
        meta = result.data.dataset_info["categories_meta"]
        assert meta[1]["keypoints"] == KPT_NAMES
        assert meta[1]["skeleton"] == [[0, 1], [0, 2]]

    def test_write_keypoints_and_recompute_num_keypoints(self, sample_coco_pose, temp_dir):
        handler = CocoAnnotationHandler(annotation_file=str(sample_coco_pose))
        dataset = handler.read().data
        # Corrupt the stored count — write must recompute it
        out_file = temp_dir / "out.json"
        write_result = handler.write(dataset, str(out_file))
        assert write_result.success is True
        written = json.loads(out_file.read_text(encoding="utf-8"))
        ann = written["annotations"][0]
        assert ann["keypoints"] == [130.0, 120.0, 2, 125.0, 115.0, 1, 0.0, 0.0, 0]
        assert ann["num_keypoints"] == 2
        cat = written["categories"][0]
        assert cat["keypoints"] == KPT_NAMES
        assert cat["skeleton"] == [[0, 1], [0, 2]]
        # categories_meta is not leaked as a top-level key
        assert "categories_meta" not in written

    def test_write_derives_bbox_from_labeled_keypoints(self, sample_coco_pose, temp_dir):
        handler = CocoAnnotationHandler(annotation_file=str(sample_coco_pose))
        dataset = handler.read().data
        obj = dataset.images[0].objects[0]
        obj.bbox = None
        out_file = temp_dir / "out.json"
        write_result = handler.write(dataset, str(out_file))
        assert write_result.success is True
        ann = json.loads(out_file.read_text(encoding="utf-8"))["annotations"][0]
        # Extent of labeled points (125,115)-(130,120): [125, 115, 5, 5]
        assert ann["bbox"] == [125.0, 115.0, 5.0, 5.0]
        assert ann["area"] == pytest.approx(25.0)

    def test_validate_accepts_pose_file(self, sample_coco_pose):
        handler = CocoAnnotationHandler(annotation_file=str(sample_coco_pose))
        assert handler.validate() is True

    def test_validate_rejects_bad_visibility(self, sample_coco_pose, temp_dir):
        data = json.loads(sample_coco_pose.read_text(encoding="utf-8"))
        data["annotations"][0]["keypoints"] = [130, 120, 5, 125, 115, 1, 0, 0, 0]
        bad_file = temp_dir / "bad.json"
        bad_file.write_text(json.dumps(data))
        handler = CocoAnnotationHandler(annotation_file=str(bad_file))
        assert handler.validate() is False

    def test_validate_rejects_num_keypoints_mismatch(self, sample_coco_pose, temp_dir):
        data = json.loads(sample_coco_pose.read_text(encoding="utf-8"))
        data["annotations"][0]["num_keypoints"] = 3
        bad_file = temp_dir / "bad.json"
        bad_file.write_text(json.dumps(data))
        handler = CocoAnnotationHandler(annotation_file=str(bad_file))
        assert handler.validate() is False


class TestNonPoseKeypointDrop:
    """Non-pose mode drops keypoints with a warning (spec_conversion.md §3.7.1)."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_coco_pose(self, temp_dir):
        coco_file = temp_dir / "pose.json"
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
                    "keypoints": [130, 120, 2, 125, 115, 2, 0, 0, 0],
                    "num_keypoints": 2,
                }
            ],
            "categories": [{"id": 1, "name": "person", "keypoints": KPT_NAMES}],
        }
        coco_file.write_text(json.dumps(data), encoding="utf-8")
        return coco_file

    def test_streaming_coco_to_yolo_drops_keypoints_in_non_pose_mode(
        self, sample_coco_pose, temp_dir, capsys
    ):
        """v3.4.0: without pose=True, keypoints are dropped with a warning —
        the conversion proceeds (spec_conversion.md §3.7.1)."""
        from dataflow.convert import YoloAndCocoConverter

        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\n")
        image_dir = temp_dir / "images"
        image_dir.mkdir()

        converter = YoloAndCocoConverter(source_to_target=False)
        result = converter.convert(
            str(sample_coco_pose),
            str(temp_dir / "out"),
            class_file=str(class_file),
            image_dir=str(image_dir),
        )
        assert result.success is True
        captured = capsys.readouterr()
        assert "Dropped keypoints" in captured.out + captured.err
        # The output row is a plain detection row (5 tokens, no keypoints)
        line = (temp_dir / "out" / "labels" / "test.txt").read_text(encoding="utf-8")
        assert len(line.split()) == 5

    def test_batch_labelme_to_coco_drops_keypoints_in_non_pose_mode(self, temp_dir, capsys):
        """v3.4.0: without pose=True, keypoints are dropped with a warning —
        the conversion proceeds (spec_conversion.md §3.7.1)."""
        from dataflow.convert import CocoAndLabelMeConverter

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
        (label_dir / "test.json").write_text(json.dumps(data), encoding="utf-8")

        # source_to_target=False → LabelMe (source) → COCO (target), the
        # batch pipeline.
        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\n")
        converter = CocoAndLabelMeConverter(source_to_target=False)
        result = converter.convert(
            str(label_dir), str(temp_dir / "out.json"), class_file=str(class_file)
        )
        assert result.success is True
        captured = capsys.readouterr()
        assert "Dropped keypoints" in captured.out + captured.err
        written = json.loads((temp_dir / "out.json").read_text(encoding="utf-8"))
        assert "keypoints" not in written["annotations"][0]
