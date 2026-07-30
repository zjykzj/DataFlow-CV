"""
Evaluation module for DataFlow-CV.

Provides COCO-standard evaluation metrics for object detection and
instance segmentation.  Wraps pycocotools with a clean Python API and
structured result containers.

Key features:
- Detection evaluation (bbox IoU) via DetectionEvaluator
- Segmentation evaluation (mask IoU) via SegmentationEvaluator
- Single-threshold P/R/F1 via compute_pr_f1()
- Per-class breakdown (verbose mode)
- COCO-standard 12-metric output

Example usage:
    >>> from dataflow.evaluate import DetectionEvaluator, compute_pr_f1
    >>> from dataflow.util.logging import LogConfig
    >>> evaluator = DetectionEvaluator(log_config=LogConfig(name="eval", verbose=True))
    >>> result = evaluator.evaluate("gt.json", "dt.json")
    >>> print(result.get_summary())
    >>> # Quick P/R/F1 at IoU=0.5:
    >>> prf1 = compute_pr_f1("gt.json", "dt.json", iou_threshold=0.5)
    >>> print(f"F1={prf1.overall.f1_score:.3f}")
"""

from . import utils
from .base import BaseEvaluator
from .evaluator import DetectionEvaluator, SegmentationEvaluator
from .metrics import compute_pr_f1
from .result import (
    EvaluationMetrics,
    EvaluationResult,
    PerClassMetrics,
    PRF1Result,
    PRF1Values,
)

__all__ = [
    "BaseEvaluator",
    "DetectionEvaluator",
    "SegmentationEvaluator",
    "compute_pr_f1",
    "EvaluationResult",
    "EvaluationMetrics",
    "PerClassMetrics",
    "PRF1Result",
    "PRF1Values",
    "utils",
]
