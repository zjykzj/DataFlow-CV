#!/usr/bin/env python3
"""Dataset statistics demo — YOLO / LabelMe / COCO.

Usage:  python stats_demo.py [--verbose]
"""

import argparse
from pathlib import Path

from dataflow.analyse import StatsAnalyser
from dataflow.util import LogConfig, LogManager

project_root = Path(__file__).parent.parent.parent


def demo_yolo_stats(log_config):
    """YOLO format statistics."""
    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    if not data_dir.exists():
        print(f"  (skip — data not found: {data_dir})")
        return

    analyser = StatsAnalyser(log_config=log_config)
    result = analyser.analyse(
        label_paths=[data_dir / "labels"],
        class_file=data_dir / "classes.txt",
        image_dir=data_dir / "images",
    )
    if result.success:
        s = result.data
        print(f"  YOLO: {s.total_files} images, {s.total_annotations} objects, "
              f"{len(s.per_class)} classes")
        # Top-5 classes
        for name, count in list(s.per_class.items())[:5]:
            print(f"    {name}: {count}")
    else:
        print(f"  ✗ {result.errors[0]}")


def demo_coco_stats(log_config):
    """COCO format statistics."""
    data_dir = project_root / "assets" / "test_data" / "det" / "coco"
    coco_file = data_dir / "annotations.json"
    if not coco_file.exists():
        print(f"  (skip — data not found: {coco_file})")
        return

    analyser = StatsAnalyser(log_config=log_config)
    result = analyser.analyse(label_paths=[coco_file])
    if result.success:
        s = result.data
        print(f"  COCO: {s.total_files} images, {s.total_annotations} objects, "
              f"{len(s.per_class)} classes")
    else:
        print(f"  ✗ {result.errors[0]}")


def demo_labelme_stats(log_config):
    """LabelMe format statistics."""
    data_dir = project_root / "assets" / "test_data" / "det" / "labelme"
    if not data_dir.exists():
        print(f"  (skip — data not found: {data_dir})")
        return

    analyser = StatsAnalyser(log_config=log_config)
    result = analyser.analyse(label_paths=[data_dir])
    if result.success:
        s = result.data
        print(f"  LabelMe: {s.total_files} images, {s.total_annotations} objects, "
              f"{len(s.per_class)} classes")
    else:
        print(f"  ✗ {result.errors[0]}")


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics demo")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_config = LogConfig(name="stats_demo", verbose=args.verbose)
    logger = LogManager(log_config).logger

    logger.info("── Dataset Statistics ──")
    demo_yolo_stats(log_config)
    demo_coco_stats(log_config)
    demo_labelme_stats(log_config)
    logger.info("Done.")


if __name__ == "__main__":
    main()
