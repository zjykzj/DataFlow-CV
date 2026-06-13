#!/usr/bin/env python3
"""Detection evaluation demo — mAP (default) or P/R/F1 (--prf1)."""

import argparse
from pathlib import Path

from dataflow.evaluate import DetectionEvaluator, compute_pr_f1
from dataflow.evaluate.utils import format_metric_table, format_prf1_output
from dataflow.util.logging import LogConfig, LogManager

project_root = Path(__file__).parent.parent.parent

EVAL_DATA = project_root / "assets" / "test_data" / "evaluate"
GT_FILE = EVAL_DATA / "gt_coco.json"
DT_FILE = EVAL_DATA / "dt_coco.json"


def main():
    parser = argparse.ArgumentParser(description="Detection evaluation")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--prf1", action="store_true", help="P/R/F1 only (skip mAP)")
    parser.add_argument("--prf1-method", choices=["macro", "micro"], default="macro")
    args = parser.parse_args()

    log_config = LogConfig(name="eval_detection", verbose=args.verbose)
    logger = LogManager(log_config).logger

    if args.prf1:
        result = compute_pr_f1(
            str(GT_FILE), str(DT_FILE),
            iou_type="bbox", method=args.prf1_method,
        )
        if result.success:
            print(format_prf1_output(result))
        else:
            logger.error(f"✗ {result.errors[0] if result.errors else 'Failed'}")
    else:
        evaluator = DetectionEvaluator(log_config=log_config)
        result = evaluator.evaluate(str(GT_FILE), str(DT_FILE))
        if result.success:
            print(format_metric_table(result.metrics))
        else:
            logger.error(f"✗ {result.errors[0] if result.errors else 'Failed'}")


if __name__ == "__main__":
    main()
