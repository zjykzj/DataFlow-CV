#!/usr/bin/env python3
"""Full conversion chain demo: LabelMe → YOLO → COCO → LabelMe."""

import argparse
from pathlib import Path

from dataflow.convert import (
    CocoAndLabelMeConverter,
    LabelMeAndYoloConverter,
    YoloAndCocoConverter,
)
from dataflow.util import LogConfig, LogManager

project_root = Path(__file__).parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Full conversion chain demo")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_config = LogConfig(name="full_chain", verbose=args.verbose)
    logger = LogManager(log_config).logger

    data_dir = project_root / "assets" / "test_data" / "det" / "labelme"
    out = project_root / "samples" / "convert" / "output" / "full_chain"
    out.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        logger.error(f"Data not found: {data_dir}")
        return

    # LabelMe → YOLO
    logger.info("── LabelMe → YOLO")
    r1 = LabelMeAndYoloConverter(source_to_target=True, log_config=log_config).convert(
        source_path=str(data_dir),
        target_path=str(out / "yolo"),
        class_file=str(data_dir / "classes.txt"),
    )
    if not r1.success:
        return logger.error(f"✗ {r1.errors[0]}")
    logger.info(f"  ✓ {r1.num_images_converted} images")

    # YOLO → COCO
    logger.info("── YOLO → COCO")
    r2 = YoloAndCocoConverter(source_to_target=True, log_config=log_config).convert(
        source_path=str(out / "yolo" / "labels"),
        target_path=str(out / "coco.json"),
        class_file=str(data_dir / "classes.txt"),
        image_dir=str(data_dir),
    )
    if not r2.success:
        return logger.error(f"✗ {r2.errors[0]}")
    logger.info(f"  ✓ {r2.num_images_converted} images → coco.json")

    # COCO → LabelMe
    logger.info("── COCO → LabelMe")
    r3 = CocoAndLabelMeConverter(source_to_target=True, log_config=log_config).convert(
        source_path=str(out / "coco.json"),
        target_path=str(out / "labelme"),
    )
    if not r3.success:
        return logger.error(f"✗ {r3.errors[0]}")
    logger.info(f"  ✓ {r3.num_images_converted} images → labelme/")

    logger.info("Full chain complete ✓")


if __name__ == "__main__":
    main()
