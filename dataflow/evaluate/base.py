"""
Base evaluator abstract class for DataFlow-CV.

Defines the template-method pipeline for COCO evaluation and shared
logging / validation logic used by all concrete evaluators.
"""

import logging
import os
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .result import (
    EvaluationMetrics,
    EvaluationResult,
    PerClassMetrics,
)
from .utils import (
    _extract_stats,
    _load_coco,
    _load_dt,
    _validate_common_categories,
    _validate_coco_available,
    _validate_dt_scores,
)


class BaseEvaluator(ABC):
    """Abstract base class for COCO evaluation.

    Implements the template method :meth:`evaluate` which orchestrates
    the full evaluation pipeline.  Subclasses override
    :meth:`_create_cocoeval` to select the IoU type (bbox or segm).
    """

    def __init__(
        self,
        log_config: Optional[Any] = None,
    ):
        """Initialize the evaluator.

        Args:
            log_config: Optional ``LogConfig`` instance. If None, a
                default ``LogConfig(name="evaluate")`` is used.
                Per-class metrics are computed only when
                ``log_config.verbose=True``.
        """
        # Configure logger via unified LogManager
        from ..util.logging import LogConfig, LogManager

        if log_config is None:
            log_config = LogConfig(name="evaluate")
        self._log_manager = LogManager(log_config)
        self.logger = self._log_manager.logger

    # ------------------------------------------------------------------
    # Template method
    # ------------------------------------------------------------------

    def evaluate(
        self,
        gt_source: Union[str, Path, Dict, Any],
        dt_source: Union[str, Path, Dict, Any],
    ) -> EvaluationResult:
        """Run full COCO evaluation.

        Args:
            gt_source: Ground truth COCO data (file path, dict, or
                DatasetAnnotations).
            dt_source: Detection/prediction COCO data (file path, dict,
                list of annotation dicts, or DatasetAnnotations).
                Annotations must include ``score``. List-format files
                (plain JSON array) are loaded via ``loadRes()`` with
                images and categories sourced from GT.

        Returns:
            EvaluationResult with 12 standard metrics and optional
            per-class breakdown.
        """
        result = EvaluationResult(success=False)

        try:
            # 1. Check pycocotools availability
            _validate_coco_available()

            # 2. Load GT and DT
            self._log_info("Loading ground truth annotations...")
            coco_gt = _load_coco(gt_source)
            coco_dt = _load_dt(dt_source, coco_gt)

            # 3. Validate inputs
            valid, warnings = self.validate_inputs(coco_gt, coco_dt)
            for w in warnings:
                result.add_warning(w)
            if not valid:
                return result

            # 4. Create COCOeval and run
            self._log_info("Running COCO evaluation...")
            coco_eval = self._create_cocoeval(coco_gt, coco_dt)

            coco_eval.evaluate()
            coco_eval.accumulate()
            # summarize() populates coco_eval.stats (needed for
            # _extract_metrics) but also prints to stdout. Suppress
            # the print — the CLI handles output formatting.
            with open(os.devnull, "w") as f, redirect_stdout(f):
                coco_eval.summarize()

            # 5. Extract 12 standard metrics
            metrics = self._extract_metrics(coco_eval)
            result.metrics = metrics
            result.iou_type = self._iou_type()

            # 6. Stats
            gt_stats, dt_stats = _extract_stats(coco_gt, coco_dt)
            result.gt_stats = gt_stats
            result.dt_stats = dt_stats

            # 7. Per-class (verbose only)
            if self._log_manager.log_path is not None:
                self._log_info("Computing per-class metrics...")
                result.per_class = self._compute_per_class(
                    coco_eval, coco_gt, coco_dt
                )

            result.success = True
            result.log_path = self._log_manager.log_path
            self._log_info(
                f"Evaluation complete: {result.get_summary()}"
            )

        except Exception as e:
            result.add_error(str(e))
            self.logger.error(str(e))

        return result

    # ------------------------------------------------------------------
    # Abstract hook
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_cocoeval(self, coco_gt: Any, coco_dt: Any) -> Any:
        """Create a pycocotools.COCOeval instance.

        Subclasses override to set ``iouType`` to ``'bbox'`` or ``'segm'``.
        """
        ...

    @abstractmethod
    def _iou_type(self) -> str:
        """Return the IoU type string for this evaluator."""
        ...

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_inputs(
        self, coco_gt: Any, coco_dt: Any
    ) -> Tuple[bool, List[str]]:
        """Validate GT and DT before evaluation.

        Returns:
            Tuple of ``(valid, warnings)``.  If ``valid`` is False the
            evaluation should abort.
        """
        warnings: List[str] = []
        errors: List[str] = []

        # Check GT has categories
        if len(coco_gt.getCatIds()) == 0:
            errors.append("GT contains no categories.")

        # Check GT has images
        if len(coco_gt.getImgIds()) == 0:
            errors.append("GT contains no images.")

        # Check GT has annotations
        gt_ann_ids = coco_gt.getAnnIds()
        if len(gt_ann_ids) == 0:
            errors.append("GT contains no annotations.")

        # Check DT has annotations
        dt_ann_ids = coco_dt.getAnnIds()
        if len(dt_ann_ids) == 0:
            errors.append("DT contains no annotations.")

        # Validate DT scores
        try:
            _validate_dt_scores(coco_dt)
        except ValueError as e:
            errors.append(str(e))

        # Category cross-check
        warnings.extend(_validate_common_categories(coco_gt, coco_dt))

        if errors:
            for err in errors:
                self._log_error(err)
            return False, warnings

        return True, warnings

    # ------------------------------------------------------------------
    # Metric extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metrics(coco_eval: Any) -> EvaluationMetrics:
        """Extract the 12 COCO standard metrics from a completed COCOeval.

        Args:
            coco_eval: A COCOeval instance after ``summarize()``.

        Returns:
            EvaluationMetrics populated from ``coco_eval.stats``.
        """
        stats = coco_eval.stats
        return EvaluationMetrics(
            ap=float(stats[0]),
            ap50=float(stats[1]),
            ap75=float(stats[2]),
            ap_small=float(stats[3]),
            ap_medium=float(stats[4]),
            ap_large=float(stats[5]),
            ar_max_1=float(stats[6]),
            ar_max_10=float(stats[7]),
            ar_max_100=float(stats[8]),
            ar_small=float(stats[9]),
            ar_medium=float(stats[10]),
            ar_large=float(stats[11]),
        )

    def _compute_per_class(
        self, coco_eval: Any, coco_gt: Any, coco_dt: Any
    ) -> Dict[int, PerClassMetrics]:
        """Extract per-category metrics from COCOeval internal arrays.

        Uses ``coco_eval.eval['precision']`` and
        ``coco_eval.eval['recall']``, which have shapes:
        - precision: [T, R, K, A, M]  (IoU thrs × recall thrs × cats × areas × maxDets)
        - recall:    [T, K, A, M]      (IoU thrs × cats × areas × maxDets)

        Returns:
            Mapping of ``category_id → PerClassMetrics``.
        """
        cat_ids = coco_eval.params.catIds
        if not cat_ids:
            cat_ids = coco_gt.getCatIds()

        cat_id_to_name: Dict[int, str] = {}
        for cat in coco_gt.loadCats(cat_ids):
            cat_id_to_name[cat["id"]] = cat["name"]

        # Count GT and DT per class
        gt_counts: Dict[int, int] = {}
        dt_counts: Dict[int, int] = {}
        for ann in coco_gt.loadAnns(coco_gt.getAnnIds()):
            cid = ann["category_id"]
            gt_counts[cid] = gt_counts.get(cid, 0) + 1
        for ann in coco_dt.loadAnns(coco_dt.getAnnIds()):
            cid = ann["category_id"]
            dt_counts[cid] = dt_counts.get(cid, 0) + 1

        per_class: Dict[int, PerClassMetrics] = {}

        for idx, cat_id in enumerate(cat_ids):
            m = PerClassMetrics(
                class_id=cat_id,
                class_name=cat_id_to_name.get(cat_id, f"class_{cat_id}"),
                gt_count=gt_counts.get(cat_id, 0),
                dt_count=dt_counts.get(cat_id, 0),
            )

            try:
                # Precision for this class: [T, R, K=idx, A=0 (all), M=2 (maxDets=100)]
                prec = coco_eval.eval["precision"][:, :, idx, 0, 2]
                # Recall for this class: [T, K=idx, A=0 (all), M=2 (maxDets=100)]
                rec = coco_eval.eval["recall"][:, idx, 0, 2]

                # AP across all IoU thresholds (mean over T)
                valid_prec = prec[prec > -1]
                m.ap = float(np.mean(valid_prec)) if len(valid_prec) > 0 else -1.0

                # AP at IoU=0.50 (T=0) and IoU=0.75 (T=5)
                if prec.shape[0] > 0:
                    prec50 = prec[0][prec[0] > -1]
                    m.ap50 = float(np.mean(prec50)) if len(prec50) > 0 else -1.0

                if prec.shape[0] > 5:
                    prec75 = prec[5][prec[5] > -1]
                    m.ap75 = float(np.mean(prec75)) if len(prec75) > 0 else -1.0

                # Best P/R/F1 from recall array (IoU=0.50, area=all, maxDets=100)
                valid_rec = rec[0][rec[0] > -1]
                if len(valid_rec) > 0:
                    m.recall = float(np.max(valid_rec))

                    # Precision at the max-recall operating point
                    r_max_idx = int(np.argmax(rec[0]))
                    if r_max_idx < prec.shape[1]:
                        p_at_r = prec[0, r_max_idx]
                        m.precision = float(p_at_r) if p_at_r > -1 else 0.0

                    if m.precision + m.recall > 0:
                        m.f1_score = (
                            2.0 * m.precision * m.recall / (m.precision + m.recall)
                        )

                # Estimate TP/FP/FN from recall and precision at IoU=0.50
                if m.recall > 0 and m.gt_count > 0:
                    m.tp = int(round(m.recall * m.gt_count))
                    if m.precision > 0:
                        m.fp = int(round(m.tp / m.precision)) - m.tp
                    else:
                        m.fp = 0
                    m.fn = m.gt_count - m.tp
                else:
                    m.fn = m.gt_count

            except (IndexError, KeyError) as e:
                self._log_warning(
                    f"Could not compute per-class metrics for '{m.class_name}': {e}"
                )

            per_class[cat_id] = m

        return per_class

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_info(self, message: str) -> None:
        """Log an info-level message."""
        self.logger.info(message)

    def _log_warning(self, message: str) -> None:
        """Log a warning-level message."""
        self.logger.warning(message)

    def _log_error(self, message: str) -> None:
        """Log an error and raise ValueError."""
        self.logger.error(message)
        raise ValueError(message)
