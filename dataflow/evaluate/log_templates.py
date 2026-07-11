"""
Log formatting templates for the Evaluate module.

These are **pure functions** that return strings — they never call
``logger.info()`` or any other logging method.  The caller decides the
log level and passes the result to the logger.

Includes the moved ``format_metric_table``, ``format_per_class_table``,
and ``format_prf1_output`` functions (previously in ``evaluate/utils.py``).
"""

from typing import Any, Dict, List, Optional

from dataflow.util.logging import format_divider, format_section, format_kv, format_result_block


# ---------------------------------------------------------------------------
# Evaluate-specific header / result / phase
# ---------------------------------------------------------------------------


def format_eval_header(
    iou_type: str,
    gt_stats: Dict[str, int],
    dt_stats: Dict[str, int],
) -> str:
    """Return a header block for evaluation start.

    Args:
        iou_type: IoU type — ``"bbox"`` or ``"segm"``.
        gt_stats: GT statistics dict with keys ``images``, ``annotations``, ``categories``.
        dt_stats: DT statistics dict with keys ``images``, ``annotations``, ``categories``.

    Returns:
        Formatted header string.
    """
    lines: List[str] = []
    lines.append(format_divider())
    lines.append(f"Evaluate: {'Detection (bbox IoU)' if iou_type == 'bbox' else 'Segmentation (mask IoU)'}")
    lines.append(
        f"  GT:  {gt_stats.get('images', 0)} images, "
        f"{gt_stats.get('annotations', 0)} annotations, "
        f"{gt_stats.get('categories', 0)} categories"
    )
    lines.append(
        f"  DT:  {dt_stats.get('images', 0)} images, "
        f"{dt_stats.get('annotations', 0)} detections, "
        f"{dt_stats.get('categories', 0)} categories"
    )
    lines.append("")
    return "\n".join(lines)


def format_eval_phase(phase: str, message: str) -> str:
    """Return a phase marker.

    Args:
        phase: Phase name — ``"Load"``, ``"Validate"``, ``"Compute"``.
        message: Phase-specific status message.

    Returns:
        Formatted phase string.
    """
    return f"{format_section(phase)}\n  {message}\n"


def format_eval_result(
    iou_type: str,
    success: bool,
    duration_sec: float,
    log_path: Optional[str] = None,
) -> str:
    """Return a final result block for evaluation.

    Args:
        iou_type: IoU type string.
        success: Whether evaluation succeeded.
        duration_sec: Wall-clock duration in seconds.
        log_path: Optional log file path.

    Returns:
        Formatted result block.
    """
    status = "✓ Success" if success else "✗ Failed"
    items: Dict[str, Any] = {
        "Type": "bbox" if iou_type == "bbox" else "segm",
        "Duration": f"{duration_sec:.2f}s",
    }
    return format_result_block(status, items, log_path=log_path)


# ---------------------------------------------------------------------------
# Metric tables (moved from dataflow/evaluate/utils.py)
# ---------------------------------------------------------------------------


def format_metric_table(metrics: Any) -> str:
    """Format the 12 COCO standard metrics as a readable table.

    Args:
        metrics: An ``EvaluationMetrics`` instance.

    Returns:
        Formatted string suitable for console output.
    """
    lines = [
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}".format(metrics.ap),
        " Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:.3f}".format(metrics.ap50),
        " Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {:.3f}".format(metrics.ap75),
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}".format(metrics.ap_small),
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}".format(metrics.ap_medium),
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}".format(metrics.ap_large),
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {:.3f}".format(metrics.ar_max_1),
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {:.3f}".format(metrics.ar_max_10),
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}".format(metrics.ar_max_100),
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}".format(metrics.ar_small),
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}".format(metrics.ar_medium),
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}".format(metrics.ar_large),
    ]
    return "\n".join(lines)


def format_per_class_table(per_class: Dict[int, Any]) -> str:
    """Format per-class metrics as a readable table.

    Args:
        per_class: Mapping of ``class_id → PerClassMetrics``.

    Returns:
        Formatted string suitable for console output.
    """
    if not per_class:
        return ""

    header = (
        f"{'Class':<14s} {'GT':>5s} {'DT':>5s} {'TP':>5s} {'FP':>5s} {'FN':>5s} "
        f"{'AP':>7s} {'AP50':>7s} {'AP75':>7s} {'P':>7s} {'R':>7s} {'F1':>7s}"
    )
    sep = "─" * len(header)
    lines = [
        "Per-Class Breakdown (AP/AR: IoU 0.50:0.95 | P/R/F1: IoU 0.50):",
        sep,
        header,
        sep,
    ]

    for class_id in sorted(per_class.keys()):
        m = per_class[class_id]
        lines.append(
            f"{m.class_name:<14s} "
            f"{m.gt_count:>5d} {m.dt_count:>5d} "
            f"{m.tp:>5d} {m.fp:>5d} {m.fn:>5d} "
            f"{m.ap:>7.3f} {m.ap50:>7.3f} {m.ap75:>7.3f} "
            f"{m.precision:>7.3f} {m.recall:>7.3f} {m.f1_score:>7.3f}"
        )

    lines.append(sep)
    return "\n".join(lines)


def format_prf1_output(result: Any) -> str:
    """Format ``PRF1Result`` as a per-class table + overall summary.

    Args:
        result: A ``PRF1Result`` instance.

    Returns:
        Formatted string.
    """
    if not result.success or result.overall is None:
        return "P/R/F1 computation failed."

    lines = [
        f"Precision / Recall / F1-Score (IoU={result.iou_threshold:.2f}, "
        f"Conf={result.confidence_threshold:.2f}, "
        f"Method={result.method}):",
    ]

    # Per-class table
    if result.per_class:
        header = (
            f"{'Class':<14s} {'GT':>5s} {'TP':>5s} {'FP':>5s} {'FN':>5s} "
            f"{'P':>7s} {'R':>7s} {'F1':>7s}"
        )
        sep = "─" * (len(header) + 2)
        lines.append(sep)
        lines.append(header)
        lines.append(sep)

        for cid in sorted(result.per_class.keys()):
            v = result.per_class[cid]
            class_name = result.class_names.get(cid, str(cid))
            gt_count = v.tp + v.fn
            lines.append(
                f"{class_name:<14s} "
                f"{gt_count:>5d} {v.tp:>5d} {v.fp:>5d} {v.fn:>5d} "
                f"{v.precision:>7.4f} {v.recall:>7.4f} {v.f1_score:>7.4f}"
            )
        lines.append(sep)

    # Overall
    o = result.overall
    lines.append(
        f"  Overall:  P={o.precision:.3f}  R={o.recall:.3f}  "
        f"F1={o.f1_score:.3f}  TP={o.tp}  FP={o.fp}  FN={o.fn}"
    )

    return "\n".join(lines)
