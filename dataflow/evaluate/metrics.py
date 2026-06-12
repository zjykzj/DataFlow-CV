"""
Single-threshold P/R/F1 computation for DataFlow-CV.

Provides :func:`compute_pr_f1` which performs manual greedy matching at
a fixed IoU threshold — independent of the full COCOeval pipeline for
speed and simplicity.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .result import PRF1Result, PRF1Values
from .utils import _load_coco, _load_dt, _validate_coco_available


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pr_f1(
    gt_source: Union[str, Path, Dict, Any],
    dt_source: Union[str, Path, Dict, Any],
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.0,
    iou_type: str = "bbox",
    method: str = "macro",
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
        dt_source: Detection/prediction COCO data (file path, dict,
            list of annotation dicts, or DatasetAnnotations).
            Annotations must include ``score``. List-format files
            (plain JSON array) are loaded via ``loadRes()``.
        iou_threshold: IoU threshold for matching. Default 0.5.
        confidence_threshold: Minimum detection score. Default 0.0 (keep
            all). Detections below this are filtered out.
        iou_type: ``'bbox'`` for detection, ``'segm'`` for segmentation.
            Currently only ``'bbox'`` is supported via manual matching.
        method: Aggregation method for overall P/R/F1.
            ``"macro"`` (default) — mean of per-class P and R.
            ``"micro"`` — computed from summed TP/FP/FN across all
            categories.
        verbose: If True, log per-class progress.
        logger: Optional logger instance.

    Returns:
        PRF1Result with per-class and overall P/R/F1.
        ``result.method`` records the aggregation method used.

    Raises:
        ImportError: If pycocotools is not installed (needed to load COCO
            data).
        NotImplementedError: If ``iou_type='segm'`` and pycocotools is
            not installed (mask IoU requires the pycocotools ``mask``
            module).
    """
    _validate_coco_available()

    # Validate method parameter before any computation
    if method not in ("macro", "micro"):
        raise ValueError(
            f"Invalid `method` parameter: '{method}'. "
            "Expected 'macro' or 'micro'."
        )

    if logger is None:
        logger = logging.getLogger(__name__)

    result = PRF1Result(
        success=False,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
        method=method,
    )

    try:
        # Load data
        coco_gt = _load_coco(gt_source)
        coco_dt = _load_dt(dt_source, coco_gt)

        # Prepare category / image lookup
        cat_ids = coco_gt.getCatIds()
        img_ids = coco_gt.getImgIds()

        # Load category names
        cats = coco_gt.loadCats(cat_ids)
        result.class_names = {c["id"]: c["name"] for c in cats}

        # Accumulate per-class TP/FP/FN
        per_class_tp: Dict[int, int] = {}
        per_class_fp: Dict[int, int] = {}
        per_class_fn: Dict[int, int] = {}
        for cid in cat_ids:
            per_class_tp[cid] = 0
            per_class_fp[cid] = 0
            per_class_fn[cid] = 0

        # Match per (image, category)
        for img_id in img_ids:
            # Fetch image dimensions (needed for mask IoU polygon→RLE)
            img_info = coco_gt.loadImgs([img_id])[0]
            img_h = img_info["height"]
            img_w = img_info["width"]

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

                tp, fp, fn = _greedy_match(
                    gt_anns, dt_anns, iou_threshold, iou_type,
                    img_h, img_w,
                )

                per_class_tp[cat_id] += tp
                per_class_fp[cat_id] += fp
                per_class_fn[cat_id] += fn

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

        # Overall P/R/F1 — method-dependent aggregation
        if method == "macro":
            # Macro averaging: mean of per-class P and R
            precisions = []
            recalls = []
            for cat_id in cat_ids:
                p_val = result.per_class[cat_id].precision
                r_val = result.per_class[cat_id].recall
                precisions.append(p_val)
                recalls.append(r_val)

            p_overall = float(np.mean(precisions)) if precisions else 0.0
            r_overall = float(np.mean(recalls)) if recalls else 0.0
        else:
            # Micro averaging: overall P/R from summed TP/FP/FN
            total_tp = sum(v.tp for v in result.per_class.values())
            total_fp = sum(v.fp for v in result.per_class.values())
            total_fn = sum(v.fn for v in result.per_class.values())

            p_overall = (
                total_tp / (total_tp + total_fp)
                if (total_tp + total_fp) > 0
                else 0.0
            )
            r_overall = (
                total_tp / (total_tp + total_fn)
                if (total_tp + total_fn) > 0
                else 0.0
            )

        f1_overall = (
            2.0 * p_overall * r_overall / (p_overall + r_overall)
            if (p_overall + r_overall) > 0
            else 0.0
        )

        result.overall = PRF1Values(
            precision=p_overall,
            recall=r_overall,
            f1_score=f1_overall,
            tp=sum(v.tp for v in result.per_class.values()),
            fp=sum(v.fp for v in result.per_class.values()),
            fn=sum(v.fn for v in result.per_class.values()),
        )

        result.success = True

    except Exception as e:
        result.add_error(str(e))

    return result


# ---------------------------------------------------------------------------
# Mask IoU helpers
# ---------------------------------------------------------------------------

def _polygon_to_rle(segmentation, h: int, w: int):
    """Convert COCO segmentation (polygon or RLE) to RLE dict for mask IoU.

    Handles three cases:
    - Already RLE (dict with ``counts``): return unchanged.
    - Polygon (list of coord lists): convert via ``mask.frPyObjects()``
      + ``mask.merge()``.
    - Empty / None: return ``None``.

    Args:
        segmentation: COCO segmentation field value.
        h: Image height in pixels.
        w: Image width in pixels.

    Returns:
        RLE dict or ``None``.
    """
    if segmentation is None or not segmentation:
        return None
    if isinstance(segmentation, dict) and "counts" in segmentation:
        return segmentation
    if isinstance(segmentation, list):
        from pycocotools import mask as maskUtils

        rles = maskUtils.frPyObjects(segmentation, h, w)
        return maskUtils.merge(rles)
    return None


def _compute_mask_iou(
    dt_anns: List[Dict],
    gt_anns: List[Dict],
    h: int,
    w: int,
):
    """Compute mask IoU matrix between DT and GT annotations.

    Uses pycocotools ``mask.iou()`` which handles crowd annotations
    natively (crowd IoU = intersection / dt_area).

    Args:
        dt_anns: Detection annotations (with ``segmentation`` field).
        gt_anns: Ground truth annotations (with ``segmentation`` +
            ``iscrowd`` fields).
        h: Image height in pixels.
        w: Image width in pixels.

    Returns:
        ``np.ndarray`` of shape ``(len(dt), len(gt))`` with IoU values
        in [0, 1]. Returns all-zeros matrix if either list is empty.
    """
    from pycocotools import mask as maskUtils

    dt_rles = [_polygon_to_rle(d.get("segmentation"), h, w) for d in dt_anns]
    gt_rles = [_polygon_to_rle(g.get("segmentation"), h, w) for g in gt_anns]
    gt_iscrowd = [g.get("iscrowd", 0) for g in gt_anns]

    if not dt_rles or not gt_rles:
        return np.zeros((len(dt_anns), len(gt_anns)))

    return maskUtils.iou(dt_rles, gt_rles, gt_iscrowd)


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def _greedy_match(
    gt_anns: List[Dict],
    dt_anns: List[Dict],
    iou_threshold: float,
    iou_type: str = "bbox",
    img_h: int = 0,
    img_w: int = 0,
) -> Tuple[int, int, int]:
    """Run greedy one-to-one matching for a single (image, category).

    Crowd annotations (``iscrowd=1``) are handled per pycocotools behavior
    (see ``spec_evaluate_fundamentals.md`` §7.4):

    * Crowd GTs do **not** participate in matching and never generate FN.
    * A DT that fails to match any non-crowd GT but matches a crowd GT
      (IoU ≥ threshold) is **ignored** — it counts as neither TP nor FP.

    Args:
        gt_anns: Ground truth annotations (all same image + category).
        dt_anns: Detection annotations, sorted by score descending.
        iou_threshold: Minimum IoU for a match.
        iou_type: ``'bbox'`` or ``'segm'``.
        img_h: Image height in pixels (required for ``iou_type='segm'``).
        img_w: Image width in pixels (required for ``iou_type='segm'``).

    Returns:
        Tuple of ``(tp, fp, fn)`` counts.
    """
    # Split GT into crowd and non-crowd
    crowd_gts = [g for g in gt_anns if g.get("iscrowd", 0) == 1]
    non_crowd_gts = [g for g in gt_anns if g.get("iscrowd", 0) != 1]

    if not non_crowd_gts and not crowd_gts:
        # No GT at all → all DT are FP
        return 0, len(dt_anns), 0

    if not dt_anns:
        # No DT → only non-crowd GT count as FN
        return 0, 0, len(non_crowd_gts)

    # Pre-compute IoU matrix for segm (avoids repeated RLE conversion)
    iou_matrix = None
    crowd_iou_matrix = None
    if iou_type == "segm":
        iou_matrix = _compute_mask_iou(dt_anns, non_crowd_gts, img_h, img_w)
        if crowd_gts:
            crowd_iou_matrix = _compute_mask_iou(
                dt_anns, crowd_gts, img_h, img_w,
            )

    matched_gt: set = set()
    tp = 0
    fp = 0

    for dt_idx, dt in enumerate(dt_anns):
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(non_crowd_gts):
            if gt_idx in matched_gt:
                continue

            if iou_type == "bbox":
                iou = _compute_bbox_iou(
                    gt.get("bbox", []), dt.get("bbox", []),
                )
            else:
                iou = iou_matrix[dt_idx][gt_idx]

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        elif crowd_gts:
            # Check if DT matches any crowd GT → ignore (not counted as FP)
            crowd_match = _check_crowd_match(
                dt_idx, dt.get("bbox", []), crowd_gts,
                iou_threshold, iou_type, crowd_iou_matrix,
            )
            if not crowd_match:
                fp += 1
            # else: ignored (neither TP nor FP)
        else:
            fp += 1

    fn = len(non_crowd_gts) - len(matched_gt)
    return tp, fp, fn


def _check_crowd_match(
    dt_idx: int,
    dt_bbox: List[float],
    crowd_gts: List[Dict],
    iou_threshold: float,
    iou_type: str,
    crowd_iou_matrix=None,
) -> bool:
    """Check whether a DT matches any crowd GT.

    Args:
        dt_idx: Index of the DT in the IoU matrix (segm only).
        dt_bbox: DT bbox ``[x, y, w, h]`` (bbox only).
        crowd_gts: Crowd GT annotations.
        iou_threshold: Minimum IoU for a match.
        iou_type: ``'bbox'`` or ``'segm'``.
        crowd_iou_matrix: Pre-computed IoU matrix for segm (optional).

    Returns:
        ``True`` if the DT matches at least one crowd GT.
    """
    if iou_type == "segm" and crowd_iou_matrix is not None:
        return any(
            crowd_iou_matrix[dt_idx][c_idx] >= iou_threshold
            for c_idx in range(len(crowd_gts))
        )
    return any(
        _compute_bbox_iou(crowd_gt.get("bbox", []), dt_bbox) >= iou_threshold
        for crowd_gt in crowd_gts
    )


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
