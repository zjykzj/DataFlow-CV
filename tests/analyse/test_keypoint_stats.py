"""
Unit tests for keypoint statistics in StatsAnalyser (P5).

Covers the v3.6.0 contract (spec_analyse.md §3.2a): per-class keypoint
tallies, per-keypoint labeling rates, name resolution (COCO categories /
kpts_config / placeholders), k consistency, and detection-only datasets.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from dataflow.analyse import StatsAnalyser

KPT_NAMES = ["nose", "left_eye", "right_eye"]


class TestKeypointStats:
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp(prefix="test_kpt_stats_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def _make_coco_pose(self, tmp_path: Path) -> Path:
        data = {
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 300, "height": 200},
                {"id": 2, "file_name": "b.jpg", "width": 300, "height": 200},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10.0, 10.0, 100.0, 100.0],
                    "area": 10000.0,
                    "iscrowd": 0,
                    "keypoints": [50, 50, 2, 60, 60, 1, 0, 0, 0],
                    "num_keypoints": 2,
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [10.0, 10.0, 100.0, 100.0],
                    "area": 10000.0,
                    "iscrowd": 0,
                    "keypoints": [70, 70, 2, 80, 80, 2, 90, 90, 2],
                    "num_keypoints": 3,
                },
            ],
            "categories": [{"id": 1, "name": "person", "keypoints": KPT_NAMES}],
        }
        f = tmp_path / "pose.json"
        f.write_text(json.dumps(data))
        return f

    def test_coco_pose_stats_with_names(self, temp_dir):
        """Keypoint tallies + ratios + COCO category names."""
        coco = self._make_coco_pose(temp_dir)
        result = StatsAnalyser().analyse([coco])
        assert result.success is True
        kps = result.data.keypoint_stats
        assert kps is not None
        person = kps["person"]
        assert person["instances"] == 2
        assert person["k"] == 3
        # ann1: v=[2,1,0]; ann2: v=[2,2,2]
        assert person["labeled"] == 5
        assert person["visible"] == 4
        assert person["occluded"] == 1
        assert person["missing"] == 1
        assert person["labeled_ratio"] == pytest.approx(5 / 6)
        assert person["visible_ratio"] == pytest.approx(4 / 6)
        assert person["kpt_names"] == KPT_NAMES
        # per_keypoint: nose labeled 2/2 visible 2/2; left_eye labeled 2/2
        # visible 1/2 occluded 1/2; right_eye labeled 1/2
        pk = person["per_keypoint"]
        assert pk[0]["name"] == "nose"
        assert pk[0]["labeled_ratio"] == pytest.approx(1.0)
        assert pk[1]["occluded"] == 1
        assert pk[1]["labeled_ratio"] == pytest.approx(1.0)
        assert pk[1]["visible_ratio"] == pytest.approx(0.5)
        assert pk[2]["labeled_ratio"] == pytest.approx(0.5)

    def test_detection_only_dataset_has_no_keypoint_stats(self, temp_dir):
        data = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 300, "height": 200}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10.0, 10.0, 100.0, 100.0],
                    "area": 10000.0,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "person"}],
        }
        f = temp_dir / "det.json"
        f.write_text(json.dumps(data))
        result = StatsAnalyser().analyse([f])
        assert result.success is True
        assert result.data.keypoint_stats is None

    def test_kpts_config_names_for_yolo_source(self, temp_dir):
        """YOLO pose source gets names from the kpts_config."""
        import cv2
        import numpy as np

        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\n")
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        cv2.imwrite(str(image_dir / "a.jpg"), np.zeros((200, 300, 3), dtype=np.uint8))
        label_dir = temp_dir / "labels"
        label_dir.mkdir()
        (label_dir / "a.txt").write_text("0 0.5 0.5 0.4 0.4 0.5 0.5 2 0.6 0.6 2 0.0 0.0 0\n")
        config = temp_dir / "kpts.yaml"
        config.write_text(
            "keypoints:\n  - nose\n  - left_eye\n  - right_eye\n",
            encoding="utf-8",
        )

        result = StatsAnalyser().analyse(
            [label_dir], class_file=class_file, image_dir=image_dir, kpts_config=config
        )
        assert result.success is True
        kps = result.data.keypoint_stats
        assert kps is not None
        assert kps["person"]["kpt_names"] == KPT_NAMES
        assert kps["person"]["instances"] == 1
        assert kps["person"]["labeled"] == 2

    def test_placeholder_names_without_any_config(self, temp_dir):
        coco = self._make_coco_pose(temp_dir)
        # Strip keypoints names from categories
        data = json.loads(coco.read_text(encoding="utf-8"))
        del data["categories"][0]["keypoints"]
        coco.write_text(json.dumps(data))
        result = StatsAnalyser().analyse([coco])
        assert result.success is True
        assert result.data.keypoint_stats["person"]["kpt_names"] == [
            "keypoint_0",
            "keypoint_1",
            "keypoint_2",
        ]

    def test_inconsistent_k_errors(self, temp_dir):
        data = json.loads(self._make_coco_pose(temp_dir).read_text(encoding="utf-8"))
        data["annotations"].append(
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [10.0, 10.0, 50.0, 50.0],
                "area": 2500.0,
                "iscrowd": 0,
                "keypoints": [20, 20, 2, 30, 30, 2],  # k=2 != 3
                "num_keypoints": 2,
            }
        )
        f = temp_dir / "mixed_k.json"
        f.write_text(json.dumps(data))
        result = StatsAnalyser().analyse([f])
        assert result.success is False
        assert any("Inconsistent keypoint count" in e for e in result.errors)


class TestStatsCLIKptsConfig:
    def test_help_shows_kpts_config(self):
        from click.testing import CliRunner

        from dataflow.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["analyse", "stats", "--help"])
        assert result.exit_code == 0
        assert "--kpts-config" in result.output
