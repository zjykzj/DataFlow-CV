#!/usr/bin/env python3
"""
COCO annotation visualization example

Demonstrates how to use COCOVisualizer to visualize COCO format annotations.
Supports object detection and instance segmentation annotations, as well as polygon format and RLE format.

Usage:
    python coco_demo.py [--task {det,seg}] [--format {polygon,rle}] [--verbose]

Examples:
    python coco_demo.py                      # Visualize object detection annotations (polygon format, default)
    python coco_demo.py --task seg           # Visualize instance segmentation annotations (polygon format)
    python coco_demo.py --task seg --format rle  # Visualize instance segmentation annotations (RLE format)
    python coco_demo.py --verbose            # Enable verbose logging mode
    python coco_demo.py --task seg --verbose # Instance segmentation + verbose logging

Note: Object detection tasks only support polygon format.
"""

import argparse
import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.util import LoggingOperations, VerboseLoggingOperations
from dataflow.visualize import COCOVisualizer


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="COCO annotation visualization example")
    parser.add_argument(
        "--task",
        choices=["det", "seg"],
        default="det",
        help="Task type: det=object detection, seg=instance segmentation (default: det)",
    )
    parser.add_argument(
        "--format",
        choices=["polygon", "rle"],
        default="polygon",
        help="Annotation format: polygon=polygon format, rle=RLE format (default: polygon)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging mode")
    args = parser.parse_args()

    # Parameter validation
    if args.task == "det" and args.format == "rle":
        print("Error: Object detection tasks only support polygon format")
        print("Please use: python coco_demo.py --task det --format polygon")
        return

    # Configure logging
    if args.verbose:
        log_ops = VerboseLoggingOperations()
        logger = log_ops.get_verbose_logger(
            name="coco_visualize_demo",
            verbose=True,
            log_dir=str(project_root / "logs")
        )
        logger.info("Verbose logging mode enabled")
    else:
        log_ops = LoggingOperations()
        logger = log_ops.get_logger("coco_visualize_demo", level="INFO")

    # Select data path based on task type and format
    if args.task == "det":
        task_name = "object detection"
        data_dir = project_root / "assets" / "test_data" / "det" / "coco"
        annotation_file = data_dir / "annotations.json"
        format_name = "polygon format"
    else:  # args.task == "seg"
        task_name = "instance segmentation"
        data_dir = project_root / "assets" / "test_data" / "seg" / "coco"
        if args.format == "polygon":
            annotation_file = data_dir / "annotations.json"
            format_name = "polygon format"
        else:  # args.format == "rle"
            annotation_file = data_dir / "annotations-rle.json"
            format_name = "RLE format"

    image_dir = data_dir / "images"

    if not annotation_file.exists():
        logger.error(f"Annotation file does not exist: {annotation_file}")
        logger.info("Please ensure sample data is prepared")
        return

    logger.info("=" * 50)
    logger.info(f"COCO {task_name} annotation visualization example ({format_name})")
    logger.info("=" * 50)

    # Create visualizer
    logger.info(f"Creating COCO visualizer:")
    logger.info(f"  Task type: {task_name}")
    logger.info(f"  Annotation format: {format_name}")
    logger.info(f"  Annotation file: {annotation_file}")
    logger.info(f"  Image directory: {image_dir}")

    visualizer = COCOVisualizer(
        annotation_file=str(annotation_file),
        image_dir=str(image_dir),
        verbose=args.verbose,  # Verbose logging mode
        is_show=True,  # Display window
        is_save=False,  # Do not save
        strict_mode=True,
        logger=logger,
    )

    # Perform visualization
    logger.info("\nStarting visualization (press Enter for next image, press q to quit)...")
    result = visualizer.visualize()

    if result.success:
        logger.info(f"Visualization completed: {result.message}")
    else:
        logger.error(f"Visualization failed: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")


if __name__ == "__main__":
    main()
