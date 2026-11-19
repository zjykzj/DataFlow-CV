"""
Unit tests for keypoint (pose) evaluation (P4).

Covers the v3.5.0 contracts (spec_evaluate.md §3.4/§4.2/§5.3 rule 8):
OKS matrix computation, KeypointEvaluator (iouType='keypoints'), sigmas
resolution, PRF1-OKS matching, and the evaluate keypoints CLI.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dataflow.evaluate.metrics import _compute_oks_matrix


def _make_kpt_dataset(tmp_path: Path, k: int = 3) -> dict:
    """Two images, one person each, k keypoints (image 500x400)."""
    kpts_1 = []
    kpts_2 = []
    for i in range(k):
        kpts_1 += [250 + 20 * i, 200 + 10 * i, 2]
        kpts_2 += [100 + 25 * i, 300 + 15 * i, 2]
    return {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 500, "height": 400},
            {"id": 2, "file_name": "b.jpg", "width": 500, "height": 400},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [200.0, 150.0, 150.0, 200.0],
                "area": 30000.0,
                "iscrowd": 0,
                "keypoints": kpts_1,
                "num_keypoints": k,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [50.0, 250.0, 160.0, 180.0],
                "area": 28800.0,
                "iscrowd": 0,
                "keypoints": kpts_2,
                "num_keypoints": k,
            },
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "keypoints": [f"kpt_{i}" for i in range(k)],
            }
        ],
    }


class TestOKSMatrix:
    def test_perfect_keypoints_oks_one(self):
        gt = [{"keypoints": [100, 100, 2, 200, 200, 2, 0, 0, 0], "area": 40000.0}]
        dt = [{"keypoints": [100, 100, 2, 200, 200, 2, 0, 0, 0]}]
        matrix = _compute_oks_matrix(dt, gt, np.array([0.026, 0.025, 0.035]))
        assert matrix[0, 0] == pytest.approx(1.0)

    def test_offset_keypoints_reduce_oks(self):
        gt = [{"keypoints": [100, 100, 2, 200, 200, 2, 0, 0, 0], "area": 40000.0}]
        dt = [{"keypoints": [110, 110, 2, 210, 210, 2, 0, 0, 0]}]
        matrix = _compute_oks_matrix(dt, gt, np.array([0.026, 0.025, 0.035]))
        assert 0.0 < matrix[0, 0] < 1.0

    def test_unlabeled_gt_keypoints_excluded(self):
        """A GT keypoint with v=0 does not contribute — moving a DT point
        there changes nothing."""
        gt = [{"keypoints": [100, 100, 2, 0, 0, 0], "area": 40000.0}]
        dt1 = [{"keypoints": [100, 100, 2, 500, 500, 2]}]
        dt2 = [{"keypoints": [100, 100, 2, 1, 1, 2]}]
        sigmas = np.array([0.026, 0.025])
        m1 = _compute_oks_matrix(dt1, gt, sigmas)
        m2 = _compute_oks_matrix(dt2, gt, sigmas)
        assert m1[0, 0] == pytest.approx(1.0)
        assert m2[0, 0] == pytest.approx(1.0)

    def test_no_labeled_gt_never_matches(self):
        gt = [{"keypoints": [0, 0, 0, 0, 0, 0], "area": 40000.0}]
        dt = [{"keypoints": [100, 100, 2, 200, 200, 2]}]
        matrix = _compute_oks_matrix(dt, gt, np.array([0.026, 0.025]))
        assert matrix[0, 0] == 0.0

    def test_k_mismatch_no_match(self):
        gt = [{"keypoints": [100, 100, 2, 200, 200, 2, 300, 300, 2], "area": 40000.0}]
        dt = [{"keypoints": [100, 100, 2, 200, 200, 2]}]
        matrix = _compute_oks_matrix(dt, gt, np.array([0.026, 0.025, 0.035]))
        assert matrix[0, 0] == 0.0


class TestKeypointEvaluator:
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp(prefix="test_kpt_eval_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def _write_gt_dt(self, tmp_path: Path, data: dict, with_score: bool = True):
        gt_file = tmp_path / "gt.json"
        gt_file.write_text(json.dumps(data))
        dt_data = {
            "images": data["images"],
            "categories": data["categories"],
            "annotations": [{**ann, "score": 0.95} for ann in data["annotations"] if with_score],
        }
        dt_file = tmp_path / "dt.json"
        dt_file.write_text(json.dumps(dt_data))
        return gt_file, dt_file

    def test_perfect_predictions_ap_one(self, temp_dir):
        """GT as DT (perfect predictions) → AP = 1.0."""
        from dataflow.evaluate import KeypointEvaluator

        gt_file, dt_file = self._write_gt_dt(temp_dir, _make_kpt_dataset(temp_dir, k=3))
        result = KeypointEvaluator().evaluate(str(gt_file), str(dt_file))
        assert result.success is True
        assert result.metrics is not None
        assert result.metrics.ap == pytest.approx(1.0)
        assert result.iou_type == "keypoints"

    def test_bbox_only_data_errors(self, temp_dir):
        """Rule 8: keypoint evaluation on bbox-only data is a hard error."""
        from dataflow.evaluate import KeypointEvaluator

        data = _make_kpt_dataset(temp_dir, k=3)
        for ann in data["annotations"]:
            del ann["keypoints"]
            del ann["num_keypoints"]
        gt_file, dt_file = self._write_gt_dt(temp_dir, data)
        result = KeypointEvaluator().evaluate(str(gt_file), str(dt_file))
        assert result.success is False
        assert any("no keypoint data" in e for e in result.errors)

    def test_explicit_sigmas(self, temp_dir):
        from dataflow.evaluate import KeypointEvaluator

        gt_file, dt_file = self._write_gt_dt(temp_dir, _make_kpt_dataset(temp_dir, k=3))
        result = KeypointEvaluator(sigmas=[0.05, 0.05, 0.05]).evaluate(str(gt_file), str(dt_file))
        assert result.success is True
        assert result.metrics.ap == pytest.approx(1.0)

    def test_sigmas_length_mismatch_errors(self, temp_dir):
        from dataflow.evaluate import KeypointEvaluator

        gt_file, dt_file = self._write_gt_dt(temp_dir, _make_kpt_dataset(temp_dir, k=3))
        result = KeypointEvaluator(sigmas=[0.05]).evaluate(str(gt_file), str(dt_file))
        assert result.success is False
        assert any("sigmas must have 3 values" in e for e in result.errors)


class TestPRF1Keypoints:
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp(prefix="test_kpt_prf1_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def _write(self, tmp_path: Path, data: dict):
        gt_file = tmp_path / "gt.json"
        gt_file.write_text(json.dumps(data))
        dt_file = tmp_path / "dt.json"
        dt_file.write_text(
            json.dumps(
                {
                    "images": data["images"],
                    "categories": data["categories"],
                    "annotations": [{**ann, "score": 0.95} for ann in data["annotations"]],
                }
            )
        )
        return gt_file, dt_file

    def test_perfect_match(self, temp_dir):
        from dataflow.evaluate import compute_pr_f1

        gt_file, dt_file = self._write(temp_dir, _make_kpt_dataset(temp_dir, k=3))
        result = compute_pr_f1(str(gt_file), str(dt_file), iou_threshold=0.5, iou_type="keypoints")
        assert result.success is True
        assert result.overall is not None
        assert result.overall.precision == pytest.approx(1.0)
        assert result.overall.recall == pytest.approx(1.0)
        assert result.overall.tp == 2
        assert result.overall.fp == 0
        assert result.overall.fn == 0

    def test_oks_threshold_filters(self, temp_dir):
        """A heavily offset DT fails at a strict OKS threshold."""
        from dataflow.evaluate import compute_pr_f1

        data = _make_kpt_dataset(temp_dir, k=3)
        gt_file = temp_dir / "gt.json"
        gt_file.write_text(json.dumps(data))
        # Offset every keypoint by +100px → low OKS
        dt_anns = []
        for ann in data["annotations"]:
            kpts = [(v if i % 3 == 2 else v + 100) for i, v in enumerate(ann["keypoints"])]
            dt_anns.append({**ann, "keypoints": kpts, "score": 0.95})
        dt_file = temp_dir / "dt.json"
        dt_file.write_text(
            json.dumps(
                {
                    "images": data["images"],
                    "categories": data["categories"],
                    "annotations": dt_anns,
                }
            )
        )

        result = compute_pr_f1(str(gt_file), str(dt_file), iou_threshold=0.5, iou_type="keypoints")
        assert result.success is True
        assert result.overall is not None
        assert result.overall.tp == 0
        assert result.overall.fp == 2
        assert result.overall.fn == 2

    def test_invalid_iou_type_rejected(self, temp_dir):
        from dataflow.evaluate import compute_pr_f1

        gt_file, dt_file = self._write(temp_dir, _make_kpt_dataset(temp_dir, k=3))
        with pytest.raises(ValueError, match="iou_type"):
            compute_pr_f1(str(gt_file), str(dt_file), iou_type="pose")


class TestKeypointEvalCLI:
    def test_help_shows_keypoints_subcommand(self):
        from click.testing import CliRunner

        from dataflow.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "keypoints", "--help"])
        assert result.exit_code == 0
        assert "--kpts-config" in result.output
        assert "--prf1" in result.output
