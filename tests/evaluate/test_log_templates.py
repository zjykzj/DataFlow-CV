"""Tests for evaluate log templates."""

from dataflow.evaluate.log_templates import format_eval_header


GT_STATS = {"images": 1, "annotations": 2, "categories": 1}
DT_STATS = {"images": 1, "annotations": 2, "categories": 1}


class TestFormatEvalHeader:
    """Task-correct header labels (spec_evaluate.md §6 log pipeline)."""

    def test_detection_label(self):
        header = format_eval_header("bbox", GT_STATS, DT_STATS)
        assert "Detection (bbox IoU)" in header

    def test_segmentation_label(self):
        header = format_eval_header("segm", GT_STATS, DT_STATS)
        assert "Segmentation (mask IoU)" in header

    def test_keypoints_label(self):
        header = format_eval_header("keypoints", GT_STATS, DT_STATS)
        assert "Pose (OKS)" in header
        assert "Segmentation" not in header
