"""Tests for BaseEvaluator."""

from pathlib import Path

import pytest

from dataflow.evaluate.base import BaseEvaluator
from dataflow.evaluate.result import EvaluationResult, EvaluationMetrics

TEST_DATA = Path(__file__).parent.parent.parent / "assets" / "test_data" / "evaluate"


class ConcreteEvaluator(BaseEvaluator):
    """Concrete evaluator subclass for testing BaseEvaluator."""

    def _iou_type(self) -> str:
        return "bbox"

    def _create_cocoeval(self, coco_gt, coco_dt):
        from pycocotools.cocoeval import COCOeval
        return COCOeval(coco_gt, coco_dt, iouType="bbox")


@pytest.fixture
def gt_path():
    return TEST_DATA / "gt_coco.json"


@pytest.fixture
def dt_path():
    return TEST_DATA / "dt_coco.json"


@pytest.fixture
def dt_list_path():
    return TEST_DATA / "dt_list.json"


class TestBaseEvaluatorInit:
    """Test BaseEvaluator constructor."""

    def test_default_init(self):
        ev = ConcreteEvaluator()
        assert ev.strict_mode is True
        assert ev.verbose is False

    def test_non_strict(self):
        ev = ConcreteEvaluator(strict_mode=False)
        assert ev.strict_mode is False

    def test_verbose(self):
        ev = ConcreteEvaluator(verbose=True)
        assert ev.verbose is True
        assert ev.log_file_path is not None

    def test_custom_logger(self):
        import logging
        logger = logging.getLogger("test_eval")
        ev = ConcreteEvaluator(logger=logger)
        assert ev.logger is logger


class TestEvaluate:
    """Test the evaluate() template method."""

    def test_successful_evaluation(self, gt_path, dt_path):
        ev = ConcreteEvaluator(verbose=False)
        result = ev.evaluate(gt_path, dt_path)
        assert result.success is True
        assert result.metrics is not None
        assert isinstance(result.metrics, EvaluationMetrics)
        assert result.iou_type == "bbox"

    def test_metrics_reasonable(self, gt_path, dt_path):
        ev = ConcreteEvaluator(verbose=False)
        result = ev.evaluate(gt_path, dt_path)
        m = result.metrics
        # All 4 GT are well-matched
        assert m.ap50 >= 0.0
        assert m.ap >= 0.0
        assert m.ar_max_100 >= 0.0

    def test_verbose_evaluation(self, gt_path, dt_path):
        ev = ConcreteEvaluator(verbose=True)
        result = ev.evaluate(gt_path, dt_path)
        assert result.success is True
        assert result.per_class is not None
        assert len(result.per_class) > 0

    def test_nonexistent_file(self):
        ev = ConcreteEvaluator(verbose=False)
        result = ev.evaluate("/nonexistent.json", "/nonexistent.json")
        assert result.success is False
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower()

    def test_stats_populated(self, gt_path, dt_path):
        ev = ConcreteEvaluator(verbose=False)
        result = ev.evaluate(gt_path, dt_path)
        assert result.gt_stats["images"] == 2
        assert result.gt_stats["annotations"] == 4
        assert result.dt_stats["annotations"] == 5

    def test_successful_evaluation_with_list_dt(self, gt_path, dt_list_path):
        """List-format DT (plain JSON array) should work."""
        ev = ConcreteEvaluator(verbose=False)
        result = ev.evaluate(gt_path, dt_list_path)
        assert result.success is True
        assert result.metrics is not None
        assert result.metrics.ap50 > 0
        # Same annotation count as dict-format DT
        assert result.dt_stats["annotations"] == 5
        assert result.dt_stats["images"] == 2


class TestValidateInputs:
    """Test input validation."""

    def test_valid_inputs(self, gt_path, dt_path):
        from dataflow.evaluate.utils import _load_coco
        coco_gt = _load_coco(gt_path)
        coco_dt = _load_coco(dt_path)
        ev = ConcreteEvaluator()
        valid, warnings = ev.validate_inputs(coco_gt, coco_dt)
        assert valid is True

    def test_empty_gt(self):
        from dataflow.evaluate.utils import _load_coco
        empty_gt = {
            "images": [],
            "annotations": [],
            "categories": [],
        }
        dt_data = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9, "area": 100}],
            "categories": [{"id": 1, "name": "cat"}],
        }
        coco_gt = _load_coco(empty_gt)
        coco_dt = _load_coco(dt_data)

        # Strict mode: should raise
        ev_strict = ConcreteEvaluator(strict_mode=True)
        with pytest.raises(ValueError):
            ev_strict.validate_inputs(coco_gt, coco_dt)

        # Non-strict mode: errors become warnings, still returns valid=True
        # (but evaluation will fail later in COCOeval)
        ev_nonstrict = ConcreteEvaluator(strict_mode=False)
        valid, warnings = ev_nonstrict.validate_inputs(coco_gt, coco_dt)
        assert valid is True  # Non-strict downgrades errors to warnings
        assert len(warnings) > 0  # The errors appear as warnings


class TestExtractMetrics:
    """Test metric extraction from COCOeval."""

    def test_extract_from_stats(self, gt_path, dt_path):
        from dataflow.evaluate.utils import _load_coco
        from pycocotools.cocoeval import COCOeval

        coco_gt = _load_coco(gt_path)
        coco_dt = _load_coco(dt_path)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = BaseEvaluator._extract_metrics(coco_eval)
        assert isinstance(metrics, EvaluationMetrics)
        assert len(coco_eval.stats) == 12
        # stats[0] = AP
        assert metrics.ap == pytest.approx(float(coco_eval.stats[0]))
