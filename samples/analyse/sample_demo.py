#!/usr/bin/env python3
"""File sampling demo — random / sequential collection.

Usage:  python sample_demo.py [--verbose]
"""

import argparse
from pathlib import Path

from dataflow.analyse import SampleAnalyser
from dataflow.util import LogConfig, LogManager

project_root = Path(__file__).parent.parent.parent


def demo_yolo_labels_random(log_config):
    """Random sample of 1 YOLO label file."""
    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    if not data_dir.exists():
        print(f"  (skip — data not found: {data_dir})")
        return

    output_dir = project_root / "samples" / "analyse" / "output" / "yolo_labels_sample"
    output_dir.mkdir(parents=True, exist_ok=True)

    analyser = SampleAnalyser(log_config=log_config)
    result = analyser.analyse(
        output_dir=output_dir,
        count=1,
        shuffle=True,
        seed=42,
        label_dir=data_dir / "labels",
        class_file=data_dir / "classes.txt",
    )
    if result.success:
        s = result.data
        print(f"  YOLO labels random: {s.sampled_count}/{s.total_count} files → {output_dir}")
    else:
        print(f"  ✗ {result.errors[0]}")


def demo_yolo_labels_sequential(log_config):
    """Sequential sample of 1 YOLO label file."""
    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    if not data_dir.exists():
        print(f"  (skip — data not found: {data_dir})")
        return

    output_dir = project_root / "samples" / "analyse" / "output" / "yolo_labels_seq_sample"
    output_dir.mkdir(parents=True, exist_ok=True)

    analyser = SampleAnalyser(log_config=log_config)
    result = analyser.analyse(
        output_dir=output_dir,
        count=1,
        shuffle=False,
        label_dir=data_dir / "labels",
    )
    if result.success:
        s = result.data
        print(f"  YOLO labels sequential: {s.sampled_count}/{s.total_count} files → {output_dir}")
    else:
        print(f"  ✗ {result.errors[0]}")


def demo_images_only(log_config):
    """Random sample of 1 image file."""
    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    if not data_dir.exists():
        print(f"  (skip — data not found: {data_dir})")
        return

    output_dir = project_root / "samples" / "analyse" / "output" / "images_sample"
    output_dir.mkdir(parents=True, exist_ok=True)

    analyser = SampleAnalyser(log_config=log_config)
    result = analyser.analyse(
        output_dir=output_dir,
        count=1,
        shuffle=True,
        seed=42,
        image_dir=data_dir / "images",
    )
    if result.success:
        s = result.data
        print(f"  Images random: {s.sampled_count}/{s.total_count} files → {output_dir}")
    else:
        print(f"  ✗ {result.errors[0]}")


def demo_yolo_both(log_config):
    """Random sample of 1 label + matched image."""
    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    if not data_dir.exists():
        print(f"  (skip — data not found: {data_dir})")
        return

    output_dir = project_root / "samples" / "analyse" / "output" / "yolo_both_sample"
    output_dir.mkdir(parents=True, exist_ok=True)

    analyser = SampleAnalyser(log_config=log_config)
    result = analyser.analyse(
        output_dir=output_dir,
        count=1,
        shuffle=True,
        seed=42,
        label_dir=data_dir / "labels",
        image_dir=data_dir / "images",
        class_file=data_dir / "classes.txt",
    )
    if result.success:
        s = result.data
        print(f"  YOLO both random: {s.sampled_count}/{s.total_count} files → {output_dir}")
    else:
        print(f"  ✗ {result.errors[0]}")


def main():
    parser = argparse.ArgumentParser(description="File sampling demo")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_config = LogConfig(name="sample_demo", verbose=args.verbose)
    logger = LogManager(log_config).logger

    logger.info("── File Sampling ──")
    demo_yolo_labels_random(log_config)
    demo_yolo_labels_sequential(log_config)
    demo_images_only(log_config)
    demo_yolo_both(log_config)
    logger.info("Done.")


if __name__ == "__main__":
    main()
