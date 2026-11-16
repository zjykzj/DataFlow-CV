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

    def test_per_class_prf1_at_max_f1_operating_point(self, tmp_path):
        """Per-class P/R/F1 come from a single (max-F1) operating point.

        Regression test for the max-P precision + max-R recall bug: with FP
        scores ranked above TP scores, P=0.5 at R=1.0 is unreachable, so the
        max-F1 point (P=0.4, R=1.0) must be reported — not the impossible
        combination (P=0.5, R=1.0, F1=0.667). See spec_evaluate.md §2.2.
        """
        import json

        from dataflow.util.logging import LogConfig

        gt = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "cat"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 10, 10],
                    "area": 100.0,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [50, 50, 10, 10],
                    "area": 100.0,
                    "iscrowd": 0,
                },
            ],
        }
        dt = [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [20, 20, 10, 10],
                "area": 100.0,
                "score": 0.95,
            },  # FP
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100.0,
                "score": 0.90,
            },  # TP
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [30, 30, 10, 10],
                "area": 100.0,
                "score": 0.85,
            },  # FP
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [40, 40, 10, 10],
                "area": 100.0,
                "score": 0.80,
            },  # FP
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [50, 50, 10, 10],
                "area": 100.0,
                "score": 0.75,
            },  # TP
        ]
        gt_file = tmp_path / "gt.json"
        dt_file = tmp_path / "dt.json"
        gt_file.write_text(json.dumps(gt), encoding="utf-8")
        dt_file.write_text(json.dumps(dt), encoding="utf-8")

        ev = DetectionEvaluator(log_config=LogConfig(name="test_maxf1", verbose=True))
        result = ev.evaluate(str(gt_file), str(dt_file))
        assert result.success is True
        assert result.per_class is not None

        m = result.per_class[1]
        # Reaching R=1.0 requires all 5 detections → P=0.4; that point also
        # maximizes F1 (0.571 vs 0.5 at R=0.5, P=0.5).
        assert m.precision == pytest.approx(0.4, abs=1e-3)
        assert m.recall == pytest.approx(1.0)
        assert m.f1_score == pytest.approx(0.5714, abs=1e-3)
        assert m.tp == 2
        assert m.fp == 3
        assert m.fn == 0

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
