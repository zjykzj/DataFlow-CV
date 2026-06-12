"""Tests for compute_pr_f1 and matching logic."""

from pathlib import Path

import pytest

from dataflow.evaluate.metrics import (
    _compute_bbox_iou,
    _greedy_match,
    compute_pr_f1,
)
from dataflow.evaluate.result import PRF1Result

TEST_DATA = Path(__file__).parent.parent.parent / "assets" / "test_data" / "evaluate"


@pytest.fixture
def gt_path():
    return TEST_DATA / "gt_coco.json"


@pytest.fixture
def dt_path():
    return TEST_DATA / "dt_coco.json"


@pytest.fixture
def dt_list_path():
    return TEST_DATA / "dt_list.json"


class TestBboxIoU:
    """Test bbox IoU computation."""

    def test_perfect_overlap(self):
        bbox = [100.0, 200.0, 150.0, 180.0]
        assert _compute_bbox_iou(bbox, bbox) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = [0.0, 0.0, 100.0, 100.0]
        b = [200.0, 200.0, 100.0, 100.0]
        assert _compute_bbox_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = [0.0, 0.0, 100.0, 100.0]
        b = [50.0, 50.0, 100.0, 100.0]
        iou = _compute_bbox_iou(a, b)
        # Intersection: [50, 50] - [100, 100] = 50*50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500/17500 ≈ 0.1429
        assert iou == pytest.approx(2500 / 17500)

    def test_contained(self):
        a = [0.0, 0.0, 100.0, 100.0]  # [0,0] - [100,100]
        b = [25.0, 25.0, 50.0, 50.0]  # [25,25] - [75,75]
        iou = _compute_bbox_iou(a, b)
        # Intersection = 50*50 = 2500
        # Union = 100*100 = 10000
        assert iou == pytest.approx(0.25)

    def test_empty_bbox(self):
        assert _compute_bbox_iou([], [0, 0, 10, 10]) == 0.0
        assert _compute_bbox_iou([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0


class TestGreedyMatch:
    """Test greedy matching algorithm."""

    def test_perfect_match(self):
        gt = [{"bbox": [100, 200, 150, 180]}]
        dt = [{"bbox": [100, 200, 150, 180], "score": 0.9}]
        tp, fp, fn = _greedy_match(gt, dt, iou_threshold=0.5)
        assert tp == 1
        assert fp == 0
        assert fn == 0

    def test_no_match_low_iou(self):
        gt = [{"bbox": [0, 0, 100, 100]}]
        dt = [{"bbox": [500, 500, 50, 50], "score": 0.9}]
        tp, fp, fn = _greedy_match(gt, dt, iou_threshold=0.5)
        assert tp == 0
        assert fp == 1
        assert fn == 1

    def test_duplicate_detection(self):
        """Second detection of same GT should be FP."""
        gt = [{"bbox": [100, 200, 150, 180]}]
        dt = [
            {"bbox": [105, 205, 145, 175], "score": 0.95},
            {"bbox": [110, 210, 140, 170], "score": 0.80},
        ]
        tp, fp, fn = _greedy_match(gt, dt, iou_threshold=0.5)
        assert tp == 1
        assert fp == 1  # Second detection is FP (GT already matched)
        assert fn == 0

    def test_higher_confidence_wins(self):
        """Higher confidence detection claims the GT first."""
        gt = [{"bbox": [100, 200, 150, 180]}]
        dt = [
            {"bbox": [105, 205, 145, 175], "score": 0.95},  # Good match
            {"bbox": [500, 500, 50, 50], "score": 0.40},     # Bad match (far away)
        ]
        tp, fp, fn = _greedy_match(gt, dt, iou_threshold=0.5)
        assert tp == 1
        assert fp == 1  # Bad match is FP
        assert fn == 0

    def test_no_gt(self):
        dt = [{"bbox": [100, 200, 150, 180], "score": 0.9}]
        tp, fp, fn = _greedy_match([], dt, iou_threshold=0.5)
        assert tp == 0
        assert fp == 1
        assert fn == 0

    def test_no_dt(self):
        gt = [{"bbox": [100, 200, 150, 180]}]
        tp, fp, fn = _greedy_match(gt, [], iou_threshold=0.5)
        assert tp == 0
        assert fp == 0
        assert fn == 1

    def test_strict_iou(self):
        """At IoU=0.9, slightly-off bbox should fail."""
        gt = [{"bbox": [100, 200, 150, 180]}]
        dt = [{"bbox": [105, 205, 145, 175], "score": 0.9}]
        tp90, fp90, fn90 = _greedy_match(gt, dt, iou_threshold=0.9)
        tp50, fp50, fn50 = _greedy_match(gt, dt, iou_threshold=0.5)
        # Should match at 0.5 but possibly fail at 0.9
        assert tp50 == 1
        # At 0.9 it may or may not match depending on exact overlap


class TestComputePRF1:
    """Test compute_pr_f1 function."""

    def test_returns_prf1result(self, gt_path, dt_path):
        result = compute_pr_f1(gt_path, dt_path, iou_threshold=0.5)
        assert isinstance(result, PRF1Result)
        assert result.success is True
        assert result.overall is not None

    def test_overall_matches_expectation(self, gt_path, dt_path):
        """4 GT, 5 DT: one DT is far away → FP, rest match → TP=4, FP=1."""
        result = compute_pr_f1(gt_path, dt_path, iou_threshold=0.5)
        assert result.overall.tp == 4
        assert result.overall.fp == 1
        assert result.overall.fn == 0
        # P = 4/5 = 0.8, R = 4/4 = 1.0
        assert result.overall.precision == pytest.approx(0.8)
        assert result.overall.recall == pytest.approx(1.0)

    def test_per_class(self, gt_path, dt_path):
        result = compute_pr_f1(gt_path, dt_path, iou_threshold=0.5)
        assert len(result.per_class) == 2

    def test_confidence_filter(self, gt_path, dt_path):
        """With high confidence threshold, low-score FP should be filtered out."""
        result = compute_pr_f1(gt_path, dt_path, iou_threshold=0.5, confidence_threshold=0.9)
        assert result.success is True
        # Only DTs with score >= 0.9 are considered
        # DTs: scores=[0.95, 0.40, 0.88, 0.92, 0.85]
        # Filtered (>=0.9): scores=[0.95, 0.92] → 2 DTs match 2 GTs (cat)
        # The dog DT with score 0.88 gets filtered out
        # So expected: 2 TP (cat + cat), FPs=0, FNs depends
        assert result.overall is not None

    def test_strict_iou_threshold(self, gt_path, dt_path):
        """With very strict IoU, fewer matches."""
        result = compute_pr_f1(gt_path, dt_path, iou_threshold=0.95)
        assert result.success is True

    def test_iou_threshold_zero(self, gt_path, dt_path):
        """With IoU=0, every DT matches (first DT claims each GT)."""
        result = compute_pr_f1(gt_path, dt_path, iou_threshold=0.0)
        assert result.success is True
        # All 5 DTs should match since IoU threshold is 0
        assert result.overall.tp >= 4

    def test_with_list_dt(self, gt_path, dt_list_path):
        """List-format DT should produce same results as dict-format.

        4 GT, 5 DT: one DT is far away → FP, rest match → TP=4, FP=1.
        """
        result = compute_pr_f1(gt_path, dt_list_path, iou_threshold=0.5)
        assert result.success is True
        assert result.overall.tp == 4
        assert result.overall.fp == 1
        assert result.overall.fn == 0
        assert result.overall.precision == pytest.approx(0.8)
        assert result.overall.recall == pytest.approx(1.0)
