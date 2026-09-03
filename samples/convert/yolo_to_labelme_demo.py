#!/usr/bin/env python3
"""YOLO → LabelMe conversion demo."""

import argparse
from pathlib import Path

from dataflow.convert import LabelMeAndYoloConverter
from dataflow.util import LogConfig, LogManager

project_root = Path(__file__).parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="YOLO → LabelMe conversion")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_config = LogConfig(name="yolo2labelme", verbose=args.verbose)
    logger = LogManager(log_config).logger

    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    output_dir = project_root / "samples" / "convert" / "output" / "yolo2labelme"

    converter = LabelMeAndYoloConverter(source_to_target=False, log_config=log_config)
    result = converter.convert(
        source_path=str(data_dir / "labels"),
        target_path=str(output_dir),
        class_file=str(data_dir / "classes.txt"),
        image_dir=str(data_dir / "images"),
    )

    logger.info(
        f"✓ {result.num_images_converted} images → {output_dir}"
        if result.success
        else f"✗ {result.errors[0] if result.errors else 'Failed'}"
    )


if __name__ == "__main__":
    main()
