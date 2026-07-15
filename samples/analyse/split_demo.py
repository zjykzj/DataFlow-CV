#!/usr/bin/env python3
"""Train/test split demo — YOLO / COCO.

Usage:  python split_demo.py [--verbose]
"""

import argparse
from pathlib import Path

from dataflow.analyse import SplitAnalyser
from dataflow.util import LogConfig, LogManager

project_root = Path(__file__).parent.parent.parent


def demo_yolo_split(log_config):
    """YOLO format train/test split."""
    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    if not data_dir.exists():
        print(f"  (skip — data not found: {data_dir})")
        return

    output_dir = project_root / "samples" / "analyse" / "output" / "yolo_split"
    output_dir.mkdir(parents=True, exist_ok=True)

    analyser = SplitAnalyser(log_config=log_config)
    result = analyser.analyse(
        label_path=data_dir / "labels",
        output_dir=output_dir,
        ratio=0.8,
        seed=42,
        class_file=data_dir / "classes.txt",
        image_dir=data_dir / "images",
    )
    if result.success:
        s = result.data
        print(f"  YOLO split: train={s.train_count}, val={s.val_count} → {output_dir}")
    else:
        print(f"  ✗ {result.errors[0]}")


def demo_coco_split(log_config):
    """COCO format train/test split."""
    data_dir = project_root / "assets" / "test_data" / "det" / "coco"
    coco_file = data_dir / "annotations.json"
    if not coco_file.exists():
        print(f"  (skip — data not found: {coco_file})")
        return

    output_dir = project_root / "samples" / "analyse" / "output" / "coco_split"
    output_dir.mkdir(parents=True, exist_ok=True)

    analyser = SplitAnalyser(log_config=log_config)
    result = analyser.analyse(
        label_path=coco_file,
        output_dir=output_dir,
        ratio=0.8,
        seed=42,
    )
    if result.success:
        s = result.data
        print(f"  COCO split: train={s.train_count}, val={s.val_count} → {output_dir}")
    else:
        print(f"  ✗ {result.errors[0]}")


def main():
    parser = argparse.ArgumentParser(description="Train/test split demo")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_config = LogConfig(name="split_demo", verbose=args.verbose)
    logger = LogManager(log_config).logger

    logger.info("── Train / Test Split ──")
    demo_yolo_split(log_config)
    demo_coco_split(log_config)
    logger.info("Done.")


if __name__ == "__main__":
    main()
