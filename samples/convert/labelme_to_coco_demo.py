#!/usr/bin/env python3
"""LabelMe → COCO conversion demo."""

import argparse
from pathlib import Path

from dataflow.convert import CocoAndLabelMeConverter
from dataflow.util import LogConfig, LogManager

project_root = Path(__file__).parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="LabelMe → COCO conversion")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_config = LogConfig(name="labelme2coco", verbose=args.verbose)
    logger = LogManager(log_config).logger

    data_dir = project_root / "assets" / "test_data" / "det/labelme"
    output_file = project_root / "samples/convert/output/labelme2coco.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    converter = CocoAndLabelMeConverter(source_to_target=False, log_config=log_config)
    result = converter.convert(
        source_path=str(data_dir),
        target_path=str(output_file),
        class_file=str(data_dir / "classes.txt"),
    )

    logger.info(
        f"✓ {result.num_images_converted} images → {output_file}"
        if result.success
        else f"✗ {result.errors[0] if result.errors else 'Failed'}"
    )


if __name__ == "__main__":
    main()
