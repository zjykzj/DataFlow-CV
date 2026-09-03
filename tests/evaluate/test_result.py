"""Tests for evaluate result data models."""

from dataflow.evaluate.result import (
    EvaluationMetrics,
    EvaluationResult,
    PerClassMetrics,
    PRF1Result,
    PRF1Values,
)


class TestEvaluationMetrics:
    """Test EvaluationMetrics dataclass."""

    def test_default_values(self):
        m = EvaluationMetrics()
        assert m.ap == -1.0
        assert m.ap50 == -1.0
        assert m.ap75 == -1.0
        assert m.ap_small == -1.0
        assert m.ap_medium == -1.0
        assert m.ap_large == -1.0
        assert m.ar_max_1 == -1.0
        assert m.ar_max_10 == -1.0
        assert m.ar_max_100 == -1.0
        assert m.ar_small == -1.0
        assert m.ar_medium == -1.0
        assert m.ar_large == -1.0

    def test_custom_values(self):
        m = EvaluationMetrics(ap=0.352, ap50=0.568, ar_max_100=0.467)
        assert m.ap == 0.352
        assert m.ap50 == 0.568
        assert m.ar_max_100 == 0.467
        assert m.ap75 == -1.0  # Not set


class TestPerClassMetrics:
    """Test PerClassMetrics dataclass."""

    def test_default_values(self):
        m = PerClassMetrics(class_id=1, class_name="cat")
        assert m.class_id == 1
        assert m.class_name == "cat"
        assert m.gt_count == 0
        assert m.dt_count == 0
        assert m.tp == 0
        assert m.fp == 0
        assert m.fn == 0
        assert m.ap == -1.0

    def test_populated(self):
        m = PerClassMetrics(
            class_id=1,
            class_name="cat",
            gt_count=10,
            dt_count=12,
            tp=8,
            fp=4,
            fn=2,
            ap=0.45,
            ap50=0.68,
            ap75=0.42,
            precision=0.667,
            recall=0.8,
            f1_score=0.727,
        )
        assert m.gt_count == 10
        assert m.tp == 8
        assert m.fp == 4
        assert m.fn == 2
        assert m.f1_score == 0.727


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""

    def test_success_result(self):
        r = EvaluationResult(success=True, iou_type="bbox")
        assert r.success is True
        assert r.iou_type == "bbox"
        assert r.metrics is None
        assert r.per_class is None
        assert r.errors == []
        assert r.warnings == []

    def test_add_error(self):
        r = EvaluationResult(success=True)
        r.add_error("Something went wrong")
        assert r.success is False
        assert "Something went wrong" in r.errors

    def test_add_warning(self):
        r = EvaluationResult(success=True)
        r.add_warning("Low confidence detections")
        assert r.success is True  # Warnings don't fail
        assert "Low confidence detections" in r.warnings

    def test_get_summary_failed(self):
        r = EvaluationResult(success=False)
        r.add_error("Failed")
        assert "failed" in r.get_summary().lower()

    def test_get_summary_success(self):
        metrics = EvaluationMetrics(ap=0.5, ap50=0.7, ap75=0.4)
        r = EvaluationResult(success=True, metrics=metrics, iou_type="bbox")
        summary = r.get_summary()
        assert "AP=0.500" in summary
        assert "AP50=0.700" in summary
        assert "bbox" in summary

    def test_get_summary_no_metrics(self):
        r = EvaluationResult(success=True)
        assert "no metrics" in r.get_summary().lower()


class TestPRF1Values:
    """Test PRF1Values dataclass."""

    def test_default_values(self):
        v = PRF1Values()
        assert v.precision == 0.0
        assert v.recall == 0.0
        assert v.f1_score == 0.0
        assert v.tp == 0
        assert v.fp == 0
        assert v.fn == 0


class TestPRF1Result:
    """Test PRF1Result dataclass."""

    def test_success_result(self):
        overall = PRF1Values(precision=0.8, recall=0.9, f1_score=0.847, tp=8, fp=2, fn=1)
        r = PRF1Result(success=True, iou_threshold=0.5, overall=overall)
        assert r.success is True
        assert r.overall.precision == 0.8
        assert r.iou_threshold == 0.5

    def test_add_error(self):
        r = PRF1Result(success=True)
        r.add_error("Computation failed")
        assert r.success is False
        assert "Computation failed" in r.errors
