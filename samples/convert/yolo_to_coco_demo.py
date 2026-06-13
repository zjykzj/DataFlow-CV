#!/usr/bin/env python3
"""YOLO → COCO conversion demo.

Usage:  python yolo_to_coco_demo.py [--verbose] [--prediction]
"""

import argparse
from pathlib import Path

from dataflow.convert import YoloAndCocoConverter
from dataflow.util.logging import LogConfig, LogManager

project_root = Path(__file__).parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="YOLO → COCO conversion")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--prediction", action="store_true", help="Treat input as YOLO predictions")
    args = parser.parse_args()

    log_config = LogConfig(name="yolo2coco", verbose=args.verbose)
    logger = LogManager(log_config).logger

    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    output_file = project_root / "samples" / "convert" / "output" / "yolo2coco.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        logger.error(f"Data not found: {data_dir}")
        return

    converter = YoloAndCocoConverter(
        source_to_target=True,
        prediction=args.prediction,
        log_config=log_config,
    )
    result = converter.convert(
        source_path=str(data_dir / "labels"),
        target_path=str(output_file),
        class_file=str(data_dir / "classes.txt"),
        image_dir=str(data_dir / "images"),
    )

    if result.success:
        logger.info(f"✓ Converted {result.num_images_converted} images → {output_file}")
    else:
        logger.error(f"✗ {result.errors[0] if result.errors else 'Failed'}")


if __name__ == "__main__":
    main()
