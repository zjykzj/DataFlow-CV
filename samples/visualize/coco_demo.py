#!/usr/bin/env python3
"""COCO visualization demo."""

import argparse
from pathlib import Path

from dataflow.util.logging import LogConfig, LogManager
from dataflow.visualize import COCOVisualizer

project_root = Path(__file__).parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="COCO visualization")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_config = LogConfig(name="viz_coco", verbose=args.verbose)
    logger = LogManager(log_config).logger

    data_dir = project_root / "assets" / "test_data" / "det" / "coco"
    image_dir = data_dir / "images"

    if not data_dir.exists():
        logger.error(f"Data not found: {data_dir}")
        return

    visualizer = COCOVisualizer(
        annotation_file=str(data_dir / "annotations.json"),
        image_dir=str(image_dir),
        is_show=True,
        is_save=True,
        output_dir=data_dir / "visualized_output",
        log_config=log_config,
    )

    logger.info("Starting visualization (Enter=next, q=quit)...")
    result = visualizer.visualize()

    if result.success:
        logger.info(f"✓ {result.data.get('processed_count', 0)} images processed")
    else:
        logger.error(f"✗ {result.message or result.errors}")


if __name__ == "__main__":
    main()
