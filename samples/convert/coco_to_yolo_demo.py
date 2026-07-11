#!/usr/bin/env python3
"""COCO → YOLO conversion demo."""

import argparse
from pathlib import Path

from dataflow.convert import YoloAndCocoConverter
from dataflow.util import LogConfig, LogManager

project_root = Path(__file__).parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="COCO → YOLO conversion")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_config = LogConfig(name="coco2yolo", verbose=args.verbose)
    logger = LogManager(log_config).logger

    data_dir = project_root / "assets" / "test_data" / "det/coco"
    output_dir = project_root / "samples/convert/output/coco2yolo"

    converter = YoloAndCocoConverter(source_to_target=False, log_config=log_config)
    result = converter.convert(
        source_path=str(data_dir / 'annotations.json'),
        target_path=str(output_dir),
        class_file=str(data_dir / "classes.txt"),
    )

    logger.info(f"✓ {result.num_images_converted} images → {output_dir}" if result.success else f"✗ {result.errors[0] if result.errors else 'Failed'}")


if __name__ == "__main__":
    main()
