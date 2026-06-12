"""Tests for concrete evaluators (DetectionEvaluator, SegmentationEvaluator)."""

from pathlib import Path

import pytest

from dataflow.evaluate import DetectionEvaluator, SegmentationEvaluator

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


class TestDetectionEvaluator:
    """Test DetectionEvaluator."""

    def test_init(self):
        ev = DetectionEvaluator()
        assert ev._iou_type() == "bbox"

    def test_verbose_init(self):
        ev = DetectionEvaluator(verbose=True)
        assert ev.verbose is True

    def test_evaluate(self, gt_path, dt_path):
        ev = DetectionEvaluator(verbose=False)
        result = ev.evaluate(gt_path, dt_path)
        assert result.success is True
        assert result.iou_type == "bbox"
        assert result.metrics is not None
        assert result.metrics.ap50 > 0

    def test_evaluate_verbose(self, gt_path, dt_path):
        ev = DetectionEvaluator(verbose=True)
        result = ev.evaluate(gt_path, dt_path)
        assert result.success is True
        assert result.per_class is not None
        assert len(result.per_class) == 2
        cat_ids = {m.class_id for m in result.per_class.values()}
        assert 1 in cat_ids
        assert 2 in cat_ids

    def test_evaluate_per_class_names(self, gt_path, dt_path):
        ev = DetectionEvaluator(verbose=True)
        result = ev.evaluate(gt_path, dt_path)
        names = {m.class_name for m in result.per_class.values()}
        assert "cat" in names
        assert "dog" in names

    def test_strict_mode(self):
        ev = DetectionEvaluator(strict_mode=False)
        assert ev.strict_mode is False

    def test_evaluate_with_list_dt(self, gt_path, dt_list_path):
        """List-format DT should work with DetectionEvaluator."""
        ev = DetectionEvaluator(verbose=False)
        result = ev.evaluate(gt_path, dt_list_path)
        assert result.success is True
        assert result.metrics is not None
        assert result.metrics.ap50 > 0
        assert result.dt_stats["annotations"] == 5
        assert result.dt_stats["images"] == 2


class TestSegmentationEvaluator:
    """Test SegmentationEvaluator."""

    def test_init(self):
        ev = SegmentationEvaluator()
        assert ev._iou_type() == "segm"

    def test_verbose_init(self):
        ev = SegmentationEvaluator(verbose=True)
        assert ev.verbose is True

    def test_evaluate_bbox_data_fails(self, gt_path, dt_path):
        """Segmentation eval on bbox-only data should fail gracefully.

        pycocotools requires segmentation data for iouType='segm'.
        The evaluator returns a failed result rather than crashing.
        """
        ev = SegmentationEvaluator(verbose=False)
        result = ev.evaluate(gt_path, dt_path)
        # Segmentation eval without segmentation data should fail
        assert result.success is False
        assert len(result.errors) > 0
