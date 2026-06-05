"""
CLI evaluate subcommands for DataFlow-CV.

Provides ``detection`` and ``segmentation`` subcommands under the
``evaluate`` command group.
"""

import json
from functools import wraps
from pathlib import Path
from typing import Any

import click

from dataflow.cli.commands.utils import FormattedCommand, validate_path_exists
from dataflow.cli.exceptions import RuntimeCLIError, SystemError
from dataflow.util.logging_util import LoggingOperations, VerboseLoggingOperations


# ---------------------------------------------------------------------------
# Decorator — evaluate-specific options
# ---------------------------------------------------------------------------

def add_evaluate_options(func):
    """Decorator: add evaluate-specific options to subcommands."""

    @click.option(
        "--verbose",
        is_flag=True,
        help="Enable verbose output: per-class metrics table and file logging",
    )
    @click.option(
        "--prf1",
        is_flag=True,
        help="Additionally compute and display P/R/F1",
    )
    @click.option(
        "--prf1-iou",
        type=float,
        default=0.5,
        show_default=True,
        help="IoU threshold for P/R/F1 calculation",
    )
    @click.option(
        "--prf1-conf",
        type=float,
        default=0.0,
        show_default=True,
        help="Confidence threshold for P/R/F1 calculation",
    )
    @click.option(
        "--output",
        "-o",
        type=click.Path(path_type=Path),
        default=None,
        help="Save full evaluation result as JSON to this path",
    )
    @click.pass_context
    @wraps(func)
    def wrapper(ctx, *args, **kwargs):
        return func(ctx, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Evaluate command group
# ---------------------------------------------------------------------------

@click.group(
    name="evaluate",
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 100,
        "show_default": True,
    },
)
def evaluate_group():
    """Evaluate detection/segmentation model outputs.

    Supports COCO-standard evaluation with mAP, mAP50, mAP75, AR,
    and per-class breakdowns.
    """
    pass


# ---------------------------------------------------------------------------
# Detection subcommand
# ---------------------------------------------------------------------------

@evaluate_group.command(cls=FormattedCommand)
@add_evaluate_options
@click.argument(
    "gt_json",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "dt_json",
    type=click.Path(exists=True, path_type=Path),
)
def detection(ctx, gt_json, dt_json, verbose, prf1, prf1_iou, prf1_conf, output):
    """Evaluate object detection results (bbox IoU).

    GT_JSON: COCO format Ground Truth JSON file.

    DT_JSON: COCO format Detection/Prediction JSON file (annotations must include 'score').
    """
    from dataflow.evaluate import DetectionEvaluator, compute_pr_f1

    # Setup logging
    if verbose:
        logger, log_file_path = VerboseLoggingOperations().get_verbose_logger(
            name="evaluate.detection", verbose=True, log_dir=Path("./logs")
        )
    else:
        logger = LoggingOperations().get_logger("evaluate.detection")
        log_file_path = None

    # Validate inputs
    validate_path_exists(gt_json, "GT JSON")
    validate_path_exists(dt_json, "DT JSON")

    # Check pycocotools availability
    try:
        from pycocotools.coco import COCO  # noqa: F401
    except ImportError:
        raise SystemError(
            "pycocotools is required for evaluation. "
            "Install with: pip install pycocotools"
        )

    logger.info(f"GT: {gt_json}")
    logger.info(f"DT: {dt_json}")

    # Run evaluation
    evaluator = DetectionEvaluator(
        verbose=verbose, logger=logger, log_file_path=log_file_path
    )
    result = evaluator.evaluate(str(gt_json), str(dt_json))

    # Print results
    if result.success and result.metrics is not None:
        _print_detection_result(result, verbose, prf1, prf1_iou, prf1_conf, gt_json, dt_json, logger)

        if output:
            _save_result_json(result, output)
            logger.info(f"Evaluation result saved to: {output}")

        if verbose and result.log_file_path:
            logger.info(f"Verbose log saved to: {result.log_file_path}")

        for w in result.warnings:
            logger.warning(w)
    else:
        error_msg = result.errors[0] if result.errors else "Evaluation failed"
        logger.error(f"Evaluation failed: {error_msg}")
        raise RuntimeCLIError(f"Evaluation failed: {error_msg}")


# ---------------------------------------------------------------------------
# Segmentation subcommand
# ---------------------------------------------------------------------------

@evaluate_group.command(cls=FormattedCommand)
@add_evaluate_options
@click.argument(
    "gt_json",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "dt_json",
    type=click.Path(exists=True, path_type=Path),
)
def segmentation(ctx, gt_json, dt_json, verbose, prf1, prf1_iou, prf1_conf, output):
    """Evaluate instance segmentation results (mask IoU).

    GT_JSON: COCO format Ground Truth JSON file (annotations must include 'segmentation').

    DT_JSON: COCO format Prediction JSON file (annotations must include 'segmentation' and 'score').
    """
    from dataflow.evaluate import SegmentationEvaluator

    # Setup logging
    if verbose:
        logger, log_file_path = VerboseLoggingOperations().get_verbose_logger(
            name="evaluate.segmentation", verbose=True, log_dir=Path("./logs")
        )
    else:
        logger = LoggingOperations().get_logger("evaluate.segmentation")
        log_file_path = None

    # Validate inputs
    validate_path_exists(gt_json, "GT JSON")
    validate_path_exists(dt_json, "DT JSON")

    # Check pycocotools availability
    try:
        from pycocotools.coco import COCO  # noqa: F401
    except ImportError:
        raise SystemError(
            "pycocotools is required for evaluation. "
            "Install with: pip install pycocotools"
        )

    logger.info(f"GT: {gt_json}")
    logger.info(f"DT: {dt_json}")

    # Run evaluation
    evaluator = SegmentationEvaluator(
        verbose=verbose, logger=logger, log_file_path=log_file_path
    )
    result = evaluator.evaluate(str(gt_json), str(dt_json))

    # Print results
    if result.success and result.metrics is not None:
        _print_segmentation_result(result, verbose, prf1, prf1_iou, prf1_conf, gt_json, dt_json, logger)

        if output:
            _save_result_json(result, output)
            logger.info(f"Evaluation result saved to: {output}")

        if verbose and result.log_file_path:
            logger.info(f"Verbose log saved to: {result.log_file_path}")

        for w in result.warnings:
            logger.warning(w)
    else:
        error_msg = result.errors[0] if result.errors else "Evaluation failed"
        logger.error(f"Evaluation failed: {error_msg}")
        raise RuntimeCLIError(f"Evaluation failed: {error_msg}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_detection_result(result, verbose, prf1, prf1_iou, prf1_conf, gt_json, dt_json, logger):
    """Print detection evaluation results."""
    from dataflow.evaluate.utils import format_metric_table, format_per_class_table, format_prf1_output

    click.echo()
    click.echo(
        f"Evaluation: detection (bbox)\n"
        f"Ground Truth: {result.gt_stats.get('images', 0)} images, "
        f"{result.gt_stats.get('annotations', 0)} annotations, "
        f"{result.gt_stats.get('categories', 0)} categories\n"
        f"Detections:   {result.dt_stats.get('images', 0)} images, "
        f"{result.dt_stats.get('annotations', 0)} detections, "
        f"{result.dt_stats.get('categories', 0)} categories"
    )
    click.echo()
    click.echo(format_metric_table(result.metrics))

    if verbose and result.per_class:
        click.echo()
        click.echo(format_per_class_table(result.per_class))

    if prf1:
        from dataflow.evaluate import compute_pr_f1
        click.echo()
        prf1_result = compute_pr_f1(
            str(gt_json), str(dt_json),
            iou_threshold=prf1_iou,
            confidence_threshold=prf1_conf,
            iou_type="bbox", verbose=verbose, logger=logger,
        )
        click.echo(format_prf1_output(prf1_result))


def _print_segmentation_result(result, verbose, prf1, prf1_iou, prf1_conf, gt_json, dt_json, logger):
    """Print segmentation evaluation results."""
    from dataflow.evaluate.utils import format_metric_table, format_per_class_table, format_prf1_output

    click.echo()
    click.echo(
        f"Evaluation: segmentation (mask)\n"
        f"Ground Truth: {result.gt_stats.get('images', 0)} images, "
        f"{result.gt_stats.get('annotations', 0)} annotations, "
        f"{result.gt_stats.get('categories', 0)} categories\n"
        f"Detections:   {result.dt_stats.get('images', 0)} images, "
        f"{result.dt_stats.get('annotations', 0)} detections, "
        f"{result.dt_stats.get('categories', 0)} categories"
    )
    click.echo()
    click.echo(format_metric_table(result.metrics))

    if verbose and result.per_class:
        click.echo()
        click.echo(format_per_class_table(result.per_class))

    if prf1:
        from dataflow.evaluate import compute_pr_f1
        click.echo()
        # Manual mask IoU not yet supported — fallback to bbox
        try:
            prf1_result = compute_pr_f1(
                str(gt_json), str(dt_json),
                iou_threshold=prf1_iou,
                confidence_threshold=prf1_conf,
                iou_type="segm", verbose=verbose, logger=logger,
            )
        except NotImplementedError:
            logger.warning("Mask IoU P/R/F1 not yet supported in manual matching. Falling back to bbox IoU.")
            prf1_result = compute_pr_f1(
                str(gt_json), str(dt_json),
                iou_threshold=prf1_iou,
                confidence_threshold=prf1_conf,
                iou_type="bbox", verbose=verbose, logger=logger,
            )
        click.echo(format_prf1_output(prf1_result))


def _save_result_json(result: Any, output_path: Path) -> None:
    """Serialize an EvaluationResult to JSON."""

    def _convert(obj: Any) -> Any:
        import numpy as np
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _convert(v) for k, v in obj.__dict__.items()}
        return obj

    data = {
        "success": result.success,
        "iou_type": getattr(result, "iou_type", ""),
        "metrics": _convert(result.metrics),
        "per_class": _convert(result.per_class),
        "gt_stats": result.gt_stats,
        "dt_stats": result.dt_stats,
        "warnings": result.warnings,
        "errors": result.errors,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
