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
        from dataflow.util.logging import LogConfig

        log_config = LogConfig(name="test_det", verbose=True)
        ev = DetectionEvaluator(log_config=log_config)
        assert ev._log_manager.log_path is not None

    def test_evaluate(self, gt_path, dt_path):
        ev = DetectionEvaluator()
        result = ev.evaluate(gt_path, dt_path)
        assert result.success is True
        assert result.iou_type == "bbox"
        assert result.metrics is not None
        assert result.metrics.ap50 > 0

    def test_evaluate_verbose(self, gt_path, dt_path):
        from dataflow.util.logging import LogConfig

        log_config = LogConfig(name="test_det_v", verbose=True)
        ev = DetectionEvaluator(log_config=log_config)
        result = ev.evaluate(gt_path, dt_path)
        assert result.success is True
        assert result.per_class is not None
        assert len(result.per_class) == 2
        cat_ids = {m.class_id for m in result.per_class.values()}
        assert 1 in cat_ids
        assert 2 in cat_ids

    def test_evaluate_per_class_names(self, gt_path, dt_path):
        from dataflow.util.logging import LogConfig

        log_config = LogConfig(name="test_det_names", verbose=True)
        ev = DetectionEvaluator(log_config=log_config)
        result = ev.evaluate(gt_path, dt_path)
        names = {m.class_name for m in result.per_class.values()}
        assert "cat" in names
        assert "dog" in names

    def test_evaluate_with_list_dt(self, gt_path, dt_list_path):
        """List-format DT should work with DetectionEvaluator."""
        ev = DetectionEvaluator()
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
        from dataflow.util.logging import LogConfig

        log_config = LogConfig(name="test_segm", verbose=True)
        ev = SegmentationEvaluator(log_config=log_config)
        assert ev._log_manager.log_path is not None

    def test_evaluate_bbox_data_fails(self, gt_path, dt_path):
        """Segmentation eval on bbox-only data should fail gracefully.

        pycocotools requires segmentation data for iouType='segm'.
        The evaluator returns a failed result rather than crashing.
        """
        ev = SegmentationEvaluator()
        result = ev.evaluate(gt_path, dt_path)
        # Segmentation eval without segmentation data should fail
        assert result.success is False
        assert len(result.errors) > 0


class TestSegmentationEvaluatorSuccess:
    """Test SegmentationEvaluator with real segmentation data."""

    @pytest.fixture
    def gt_segm_path(self):
        return TEST_DATA / "gt_coco_segm.json"

    @pytest.fixture
    def dt_segm_path(self):
        return TEST_DATA / "dt_coco_segm.json"

    def test_evaluate_segmentation(self, gt_segm_path, dt_segm_path):
        """SegmentationEvaluator with real segm data should succeed."""
        ev = SegmentationEvaluator()
        result = ev.evaluate(gt_segm_path, dt_segm_path)
        assert result.success is True
        assert result.iou_type == "segm"
        assert result.metrics is not None
        assert result.metrics.ap >= 0

    def test_evaluate_segmentation_verbose(self, gt_segm_path, dt_segm_path):
        """Verbose segmentation eval should produce per-class breakdown."""
        from dataflow.util.logging import LogConfig

        log_config = LogConfig(name="test_segm_verbose", verbose=True)
        ev = SegmentationEvaluator(log_config=log_config)
        result = ev.evaluate(gt_segm_path, dt_segm_path)
        assert result.success is True
        assert result.per_class is not None
        assert len(result.per_class) > 0

    def test_evaluate_segmentation_stats(self, gt_segm_path, dt_segm_path):
        """Segmentation eval should report correct stats."""
        ev = SegmentationEvaluator()
        result = ev.evaluate(gt_segm_path, dt_segm_path)
        assert result.gt_stats["annotations"] > 0
        assert result.dt_stats["annotations"] > 0
