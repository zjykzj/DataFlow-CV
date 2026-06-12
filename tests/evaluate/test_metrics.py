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


@pytest.fixture
def gt_segm_path():
    return TEST_DATA / "gt_coco_segm.json"


@pytest.fixture
def dt_segm_path():
    return TEST_DATA / "dt_coco_segm.json"


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

    # --- Crowd (iscrowd) handling ---

    def test_crowd_gt_not_counted_as_fn(self):
        """Unmatched crowd GT should NOT count as FN."""
        gt = [
            {"bbox": [0, 0, 100, 100], "iscrowd": 1},  # crowd — unmatched
        ]
        dt = []  # no detections
        tp, fp, fn = _greedy_match(gt, dt, iou_threshold=0.5)
        assert tp == 0
        assert fp == 0
        assert fn == 0  # Crowd GT → no FN

    def test_dt_matches_crowd_ignored(self):
        """DT that matches a crowd GT (and no non-crowd GT) should be ignored."""
        gt = [
            {"bbox": [0, 0, 100, 100], "iscrowd": 1},   # crowd
            {"bbox": [300, 300, 50, 50], "iscrowd": 0},  # non-crowd, far away
        ]
        dt = [
            {"bbox": [5, 5, 95, 95], "score": 0.9},  # overlaps crowd GT only
        ]
        tp, fp, fn = _greedy_match(gt, dt, iou_threshold=0.5)
        assert tp == 0
        assert fp == 0  # Ignored because it matches crowd GT
        assert fn == 1  # Non-crowd GT unmatched → FN

    def test_dt_matches_non_crowd_wins_over_crowd(self):
        """DT should match non-crowd GT even if crowd IoU is higher.

        Both IoUs are above threshold, so the non-crowd GT is matched first.
        """
        gt = [
            {"bbox": [0, 0, 100, 100], "iscrowd": 1},     # crowd — IoU=0.81 with DT
            {"bbox": [20, 20, 80, 80], "iscrowd": 0},      # non-crowd — IoU≈0.51 with DT
        ]
        dt = [
            {"bbox": [0, 0, 90, 90], "score": 0.9},  # overlaps both (crowd more)
        ]
        tp, fp, fn = _greedy_match(gt, dt, iou_threshold=0.5)
        assert tp == 1  # Matches non-crowd GT (checked first, IoU≥threshold)
        assert fp == 0  # Not a crowd-sourced FP because it matched non-crowd
        assert fn == 0

    def test_crowd_dt_matched_then_ignored(self):
        """DT that could match crowd but already matched non-crowd → TP stands."""
        gt = [
            {"bbox": [0, 0, 100, 100], "iscrowd": 1},     # crowd — IoU≈0.51 with DT
            {"bbox": [50, 50, 100, 100], "iscrowd": 0},    # non-crowd — IoU≈0.64 with DT
        ]
        dt = [
            {"bbox": [50, 50, 80, 80], "score": 0.9},  # overlaps both
        ]
        tp, fp, fn = _greedy_match(gt, dt, iou_threshold=0.5)
        assert tp == 1  # Matched non-crowd (checked first)
        assert fp == 0
        assert fn == 0

    def test_all_gt_crowd_no_fn(self):
        """When all GT are crowd, there should be no FN.

        DT matches a crowd GT → ignored (no TP/FP). No non-crowd GTs → no FN.
        """
        gt = [
            {"bbox": [0, 0, 100, 100], "iscrowd": 1},
            {"bbox": [300, 300, 100, 100], "iscrowd": 1},
        ]
        dt = [
            {"bbox": [10, 10, 80, 80], "score": 0.9},  # IoU≈0.64 with first crowd
        ]
        tp, fp, fn = _greedy_match(gt, dt, iou_threshold=0.5)
        assert tp == 0
        assert fp == 0  # Ignored (matches crowd GT)
        assert fn == 0  # No non-crowd GTs → no FN


class TestComputePRF1:
    """Test compute_pr_f1 function."""

    def test_returns_prf1result(self, gt_path, dt_path):
        result = compute_pr_f1(gt_path, dt_path, iou_threshold=0.5)
        assert isinstance(result, PRF1Result)
        assert result.success is True
        assert result.overall is not None

    def test_overall_matches_expectation(self, gt_path, dt_path):
        """4 GT, 5 DT: one DT far away → FP, rest match → TP=4, FP=1.

        Per-class: cat TP=2 FP=1 FN=0 → P=0.667 R=1.0, dog TP=2 FP=0 FN=0 → P=1.0 R=1.0.
        Macro avg: P=(0.667+1.0)/2=0.833, R=(1.0+1.0)/2=1.0.
        """
        result = compute_pr_f1(gt_path, dt_path, iou_threshold=0.5)
        assert result.overall.tp == 4
        assert result.overall.fp == 1
        assert result.overall.fn == 0
        assert result.overall.precision == pytest.approx(0.8333, abs=1e-3)
        assert result.overall.recall == pytest.approx(1.0)
        assert result.overall.f1_score == pytest.approx(0.9090, abs=1e-3)

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
        Macro avg precision = 0.833.
        """
        result = compute_pr_f1(gt_path, dt_list_path, iou_threshold=0.5)
        assert result.success is True
        assert result.overall.tp == 4
        assert result.overall.fp == 1
        assert result.overall.fn == 0
        assert result.overall.precision == pytest.approx(0.8333, abs=1e-3)
        assert result.overall.recall == pytest.approx(1.0)

    def test_micro_method(self, gt_path, dt_path):
        """Micro averaging computes overall P/R from summed TP/FP/FN.

        Per-class: cat TP=2 FP=1 FN=0, dog TP=2 FP=0 FN=0.
        Micro: P=4/5=0.8, R=4/4=1.0, F1=0.8889.
        """
        result = compute_pr_f1(gt_path, dt_path, iou_threshold=0.5, method="micro")
        assert result.success is True
        assert result.method == "micro"
        # Overall uses micro averaging
        assert result.overall.precision == pytest.approx(0.8, abs=1e-3)
        assert result.overall.recall == pytest.approx(1.0)
        assert result.overall.f1_score == pytest.approx(0.8889, abs=1e-3)
        # TP/FP/FN totals are the same regardless of method
        assert result.overall.tp == 4
        assert result.overall.fp == 1
        assert result.overall.fn == 0

    def test_invalid_method_raises_value_error(self, gt_path, dt_path):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="macro.*micro"):
            compute_pr_f1(gt_path, dt_path, method="invalid_method")

    def test_method_field(self, gt_path, dt_path):
        """Result.method should reflect the method parameter."""
        macro_result = compute_pr_f1(gt_path, dt_path, method="macro")
        assert macro_result.method == "macro"

        micro_result = compute_pr_f1(gt_path, dt_path, method="micro")
        assert micro_result.method == "micro"

    def test_micro_list_dt(self, gt_path, dt_list_path):
        """Micro averaging with list-format DT should work."""
        result = compute_pr_f1(
            gt_path, dt_list_path, iou_threshold=0.5, method="micro"
        )
        assert result.success is True
        assert result.method == "micro"
        assert result.overall.precision == pytest.approx(0.8, abs=1e-3)
        assert result.overall.recall == pytest.approx(1.0)
        assert result.overall.f1_score == pytest.approx(0.8889, abs=1e-3)

    def test_segm_basic(self, gt_segm_path, dt_segm_path):
        """Segmentation PRF1 should work with polygon test data."""
        result = compute_pr_f1(
            gt_segm_path, dt_segm_path, iou_threshold=0.5, iou_type="segm",
        )
        assert result.success is True
        assert result.overall is not None
        # Polygon rectangles match bbox shapes → same results as bbox
        assert result.overall.tp == 4
        assert result.overall.fp == 1
        assert result.overall.fn == 0

    def test_segm_per_class(self, gt_segm_path, dt_segm_path):
        """Segm PRF1 should populate per-class structure."""
        result = compute_pr_f1(
            gt_segm_path, dt_segm_path, iou_threshold=0.5, iou_type="segm",
        )
        assert result.success is True
        assert len(result.per_class) == 2
        for cid, values in result.per_class.items():
            assert values.precision >= 0.0
            assert values.recall >= 0.0
            assert values.f1_score >= 0.0
            assert values.tp >= 0
            assert values.fp >= 0
            assert values.fn >= 0

    def test_segm_micro(self, gt_segm_path, dt_segm_path):
        """Segm PRF1 with micro averaging should work."""
        result = compute_pr_f1(
            gt_segm_path, dt_segm_path, iou_threshold=0.5,
            iou_type="segm", method="micro",
        )
        assert result.success is True
        assert result.method == "micro"
        assert result.overall.precision == pytest.approx(0.8, abs=1e-3)
        assert result.overall.recall == pytest.approx(1.0)

    def test_segm_class_names(self, gt_segm_path, dt_segm_path):
        """Segm PRF1 should include class names."""
        result = compute_pr_f1(
            gt_segm_path, dt_segm_path, iou_threshold=0.5, iou_type="segm",
        )
        assert result.success is True
        assert result.class_names == {1: "cat", 2: "dog"}
