"""
Evaluation result data models for DataFlow-CV.

Defines structured containers for COCO evaluation metrics, per-class
breakdowns, and single-threshold P/R/F1 results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvaluationMetrics:
    """COCO standard 12 evaluation metrics.

    All values are float in [0, 1] or -1.0 if undefined (e.g., no GT for
    that object scale).
    """

    # Average Precision
    ap: float = -1.0          # IoU=0.50:0.95, area=all,      maxDets=100
    ap50: float = -1.0        # IoU=0.50,      area=all,      maxDets=100
    ap75: float = -1.0        # IoU=0.75,      area=all,      maxDets=100
    ap_small: float = -1.0    # IoU=0.50:0.95, area=small,   maxDets=100
    ap_medium: float = -1.0   # IoU=0.50:0.95, area=medium,  maxDets=100
    ap_large: float = -1.0    # IoU=0.50:0.95, area=large,   maxDets=100

    # Average Recall
    ar_max_1: float = -1.0    # IoU=0.50:0.95, area=all,     maxDets=1
    ar_max_10: float = -1.0   # IoU=0.50:0.95, area=all,     maxDets=10
    ar_max_100: float = -1.0  # IoU=0.50:0.95, area=all,     maxDets=100
    ar_small: float = -1.0    # IoU=0.50:0.95, area=small,   maxDets=100
    ar_medium: float = -1.0   # IoU=0.50:0.95, area=medium,  maxDets=100
    ar_large: float = -1.0    # IoU=0.50:0.95, area=large,   maxDets=100


@dataclass
class PerClassMetrics:
    """Per-category detailed evaluation metrics."""

    class_id: int
    class_name: str
    gt_count: int = 0          # Number of GT annotations for this class
    dt_count: int = 0          # Number of DT annotations for this class
    tp: int = 0                # True Positives
    fp: int = 0                # False Positives
    fn: int = 0                # False Negatives
    ap: float = -1.0           # AP (IoU=0.50:0.95)
    ap50: float = -1.0         # AP at IoU=0.50
    ap75: float = -1.0         # AP at IoU=0.75
    precision: float = -1.0    # P at optimal confidence
    recall: float = -1.0       # R at optimal confidence
    f1_score: float = -1.0     # F1 at optimal confidence


@dataclass
class EvaluationResult:
    """Top-level evaluation result container.

    Returned by BaseEvaluator.evaluate(). Contains the 12 COCO standard
    metrics, optional per-class breakdown, and diagnostic information.
    """

    success: bool
    metrics: Optional[EvaluationMetrics] = None
    per_class: Optional[Dict[int, PerClassMetrics]] = None
    iou_type: str = ""
    gt_stats: Dict[str, int] = field(default_factory=dict)
    dt_stats: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    log_path: Optional[str] = None

    def add_warning(self, warning: str) -> None:
        """Add a non-fatal warning message."""
        self.warnings.append(warning)

    def add_error(self, error: str) -> None:
        """Add an error message and mark the result as failed."""
        self.errors.append(error)
        self.success = False

    def get_summary(self) -> str:
        """Get a human-readable summary of the evaluation result."""
        if not self.success:
            return f"Evaluation failed with {len(self.errors)} errors"

        m = self.metrics
        if m is None:
            return "Evaluation completed (no metrics)"

        return (
            f"Evaluation ({self.iou_type}): "
            f"AP={m.ap:.3f}, AP50={m.ap50:.3f}, AP75={m.ap75:.3f}"
        )


@dataclass
class PRF1Values:
    """Precision / Recall / F1-score at a single IoU threshold."""

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0


@dataclass
class PRF1Result:
    """Result of single-threshold P/R/F1 computation.

    Returned by compute_pr_f1(). Provides per-class and overall P/R/F1.

    Attributes:
        method: Aggregation method used — ``"macro"`` or ``"micro"``.
    """

    success: bool
    iou_threshold: float = 0.5
    confidence_threshold: float = 0.0
    method: str = "macro"
    overall: Optional[PRF1Values] = None
    per_class: Dict[int, PRF1Values] = field(default_factory=dict)
    class_names: Dict[int, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_warning(self, warning: str) -> None:
        """Add a non-fatal warning message."""
        self.warnings.append(warning)

    def add_error(self, error: str) -> None:
        """Add an error message and mark the result as failed."""
        self.errors.append(error)
        self.success = False
