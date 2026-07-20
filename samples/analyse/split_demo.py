#!/usr/bin/env python3
"""Train/test split demo — YOLO / LabelMe.

Usage:  python split_demo.py [--verbose]
"""

import argparse
from pathlib import Path

from dataflow.analyse import SplitAnalyser
from dataflow.util import LogConfig, LogManager

project_root = Path(__file__).parent.parent.parent


def demo_yolo_labels_only(log_config):
    """YOLO labels-only train/test split."""
    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    if not data_dir.exists():
        print(f"  (skip — data not found: {data_dir})")
        return

    output_dir = project_root / "samples" / "analyse" / "output" / "yolo_labels_split"
    output_dir.mkdir(parents=True, exist_ok=True)

    analyser = SplitAnalyser(log_config=log_config)
    result = analyser.analyse(
        output_dir=output_dir,
        ratio=0.8,
        seed=42,
        label_dir=data_dir / "labels",
        class_file=data_dir / "classes.txt",
    )
    if result.success:
        s = result.data
        print(f"  YOLO labels-only: train={s.train_count}, val={s.val_count} → {output_dir}")
    else:
        print(f"  ✗ {result.errors[0]}")


def demo_yolo_both(log_config):
    """YOLO labels+images train/test split."""
    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    if not data_dir.exists():
        print(f"  (skip — data not found: {data_dir})")
        return

    output_dir = project_root / "samples" / "analyse" / "output" / "yolo_both_split"
    output_dir.mkdir(parents=True, exist_ok=True)

    analyser = SplitAnalyser(log_config=log_config)
    result = analyser.analyse(
        output_dir=output_dir,
        ratio=0.8,
        seed=42,
        label_dir=data_dir / "labels",
        image_dir=data_dir / "images",
        class_file=data_dir / "classes.txt",
    )
    if result.success:
        s = result.data
        print(f"  YOLO both: train={s.train_count}, val={s.val_count} → {output_dir}")
    else:
        print(f"  ✗ {result.errors[0]}")


def main():
    parser = argparse.ArgumentParser(description="Train/test split demo")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_config = LogConfig(name="split_demo", verbose=args.verbose)
    logger = LogManager(log_config).logger

    logger.info("── Train / Test Split ──")
    demo_yolo_labels_only(log_config)
    demo_yolo_both(log_config)
    logger.info("Done.")


if __name__ == "__main__":
    main()
