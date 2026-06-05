"""
Single-threshold P/R/F1 computation for DataFlow-CV.

Provides :func:`compute_pr_f1` which performs manual greedy matching at
a fixed IoU threshold — independent of the full COCOeval pipeline for
speed and simplicity.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .result import PRF1Result, PRF1Values
from .utils import _load_coco, _validate_coco_available


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pr_f1(
    gt_source: Union[str, Path, Dict, Any],
    dt_source: Union[str, Path, Dict, Any],
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.0,
    iou_type: str = "bbox",
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> PRF1Result:
    """Compute Precision / Recall / F1-score at a single IoU threshold.

    Uses manual greedy matching — does NOT require the full 10×101
    COCOeval pipeline.  This is faster for single-threshold evaluation
    and returns intuitive per-class TP/FP/FN counts.

    Args:
        gt_source: Ground truth COCO data (file path, dict, or
            DatasetAnnotations).
        dt_source: Detection/prediction COCO data. Annotations must
            include ``score``.
        iou_threshold: IoU threshold for matching. Default 0.5.
        confidence_threshold: Minimum detection score. Default 0.0 (keep
            all). Detections below this are filtered out.
        iou_type: ``'bbox'`` for detection, ``'segm'`` for segmentation.
            Currently only ``'bbox'`` is supported via manual matching.
        verbose: If True, log per-class progress.
        logger: Optional logger instance.

    Returns:
        PRF1Result with per-class and overall P/R/F1.

    Raises:
        ImportError: If pycocotools is not installed (needed to load COCO
            data).
        NotImplementedError: If ``iou_type='segm'`` (mask IoU requires
            pycocotools decoding which is not yet implemented in the
            manual matching path).
    """
    _validate_coco_available()

    if logger is None:
        logger = logging.getLogger(__name__)

    result = PRF1Result(
        success=False,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
    )

    try:
        # Load data
        coco_gt = _load_coco(gt_source)
        coco_dt = _load_coco(dt_source)

        # Prepare category / image lookup
        cat_ids = coco_gt.getCatIds()
        img_ids = coco_gt.getImgIds()

        # Accumulate per-class TP/FP/FN
        per_class_tp: Dict[int, int] = {}
        per_class_fp: Dict[int, int] = {}
        per_class_fn: Dict[int, int] = {}
        for cid in cat_ids:
            per_class_tp[cid] = 0
            per_class_fp[cid] = 0
            per_class_fn[cid] = 0

        total_tp = 0
        total_fp = 0
        total_fn = 0

        # Match per (image, category)
        for img_id in img_ids:
            for cat_id in cat_ids:
                # GT for this (image, category)
                gt_anns = coco_gt.loadAnns(
                    coco_gt.getAnnIds(
                        imgIds=[img_id], catIds=[cat_id], iscrowd=None
                    )
                )

                # DT for this (image, category), filtered by confidence,
                # sorted by score descending
                dt_anns = coco_dt.loadAnns(
                    coco_dt.getAnnIds(imgIds=[img_id], catIds=[cat_id])
                )
                dt_anns = [
                    d for d in dt_anns if d.get("score", 0.0) >= confidence_threshold
                ]
                dt_anns.sort(key=lambda x: x.get("score", 0.0), reverse=True)

                tp, fp, fn = _greedy_match(gt_anns, dt_anns, iou_threshold, iou_type)

                per_class_tp[cat_id] += tp
                per_class_fp[cat_id] += fp
                per_class_fn[cat_id] += fn
                total_tp += tp
                total_fp += fp
                total_fn += fn

        # Compute per-class P/R/F1
        for cat_id in cat_ids:
            tp = per_class_tp[cat_id]
            fp = per_class_fp[cat_id]
            fn = per_class_fn[cat_id]

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0

            result.per_class[cat_id] = PRF1Values(
                precision=p,
                recall=r,
                f1_score=f1,
                tp=tp,
                fp=fp,
                fn=fn,
            )

        # Overall P/R/F1
        p_overall = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        r_overall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_overall = (
            2.0 * p_overall * r_overall / (p_overall + r_overall)
            if (p_overall + r_overall) > 0
            else 0.0
        )

        result.overall = PRF1Values(
            precision=p_overall,
            recall=r_overall,
            f1_score=f1_overall,
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
        )

        result.success = True

    except Exception as e:
        result.add_error(str(e))

    return result


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def _greedy_match(
    gt_anns: List[Dict],
    dt_anns: List[Dict],
    iou_threshold: float,
    iou_type: str = "bbox",
) -> Tuple[int, int, int]:
    """Run greedy one-to-one matching for a single (image, category).

    Args:
        gt_anns: Ground truth annotations (all same image + category).
        dt_anns: Detection annotations, sorted by score descending.
        iou_threshold: Minimum IoU for a match.
        iou_type: ``'bbox'`` or ``'segm'``.

    Returns:
        Tuple of ``(tp, fp, fn)`` counts.
    """
    if not gt_anns:
        # No GT → all DT are FP
        return 0, len(dt_anns), 0

    if not dt_anns:
        # No DT → all GT are FN
        return 0, 0, len(gt_anns)

    matched_gt: set = set()
    tp = 0
    fp = 0

    for dt in dt_anns:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_anns):
            if gt_idx in matched_gt:
                continue

            if iou_type == "bbox":
                iou = _compute_bbox_iou(gt.get("bbox", []), dt.get("bbox", []))
            else:
                # Mask IoU not yet supported in manual matching path
                raise NotImplementedError(
                    "Mask IoU (iou_type='segm') is not yet implemented "
                    "in the manual matching path. Use the full evaluation "
                    "pipeline with SegmentationEvaluator instead."
                )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_anns) - len(matched_gt)
    return tp, fp, fn


def _compute_bbox_iou(bbox_a: List[float], bbox_b: List[float]) -> float:
    """Compute IoU between two COCO bboxes.

    Each bbox is ``[x, y, width, height]`` in absolute pixels with
    (x, y) = top-left corner.

    Args:
        bbox_a: First bbox ``[x, y, w, h]``.
        bbox_b: Second bbox ``[x, y, w, h]``.

    Returns:
        IoU in [0, 1]. Returns 0.0 if either bbox has zero area.
    """
    if len(bbox_a) < 4 or len(bbox_b) < 4:
        return 0.0

    # Convert to [x1, y1, x2, y2]
    a_x1 = bbox_a[0]
    a_y1 = bbox_a[1]
    a_x2 = a_x1 + bbox_a[2]
    a_y2 = a_y1 + bbox_a[3]

    b_x1 = bbox_b[0]
    b_y1 = bbox_b[1]
    b_x2 = b_x1 + bbox_b[2]
    b_y2 = b_y1 + bbox_b[3]

    # Intersection
    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union
    area_a = bbox_a[2] * bbox_a[3]
    area_b = bbox_b[2] * bbox_b[3]
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area
