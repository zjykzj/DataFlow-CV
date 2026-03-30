#!/usr/bin/env python3
"""
YOLO to COCO format conversion example

Demonstrates how to use YoloAndCocoConverter to convert YOLO format annotations to COCO format.

Usage:
    python yolo_to_coco_demo.py [--verbose]

Examples:
    python yolo_to_coco_demo.py           # Normal mode
    python yolo_to_coco_demo.py --verbose # Verbose logging mode
"""

import argparse
import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.convert import YoloAndCocoConverter
from dataflow.util import LoggingOperations, VerboseLoggingOperations


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="YOLO to COCO format conversion example")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging mode")
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        log_ops = VerboseLoggingOperations()
        logger = log_ops.get_verbose_logger(
            name="yolo_to_coco_demo",
            verbose=True,
            log_dir=str(project_root / "logs")
        )
        logger.info("Verbose logging mode enabled")
    else:
        log_ops = LoggingOperations()
        logger = log_ops.get_logger("yolo_to_coco_demo", level="INFO")

    # Example data paths
    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    class_file = data_dir / "classes.txt"
    image_dir = data_dir / "images"
    output_dir = project_root / "samples" / "convert" / "output" / "yolo_to_coco"
    output_dir.mkdir(parents=True, exist_ok=True)
    coco_file = output_dir / "annotations.json"

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.info("Please ensure sample data is prepared")
        return

    if not class_file.exists():
        logger.error(f"Class file does not exist: {class_file}")
        return

    if not image_dir.exists():
        logger.error(f"Image directory does not exist: {image_dir}")
        return

    logger.info("=" * 50)
    logger.info("YOLO to COCO format conversion example")
    logger.info("=" * 50)

    # Create converter (YOLO→COCO direction)
    logger.info("Creating YOLO→COCO converter")
    converter = YoloAndCocoConverter(
        source_to_target=True,  # YOLO→COCO
        verbose=args.verbose,  # Verbose logging mode
        strict_mode=True,
        logger=logger,
    )

    # Perform conversion (no RLE, keep polygon format)
    logger.info(f"Performing conversion (do_rle=False):")
    logger.info(f"  Source directory: {data_dir / 'labels'}")
    logger.info(f"  Target file: {coco_file}")
    logger.info(f"  Class file: {class_file}")
    logger.info(f"  Image directory: {image_dir}")

    result = converter.convert(
        source_path=str(data_dir / "labels"),
        target_path=str(coco_file),
        class_file=str(class_file),
        image_dir=str(image_dir),
        do_rle=False,  # No RLE, output polygon point list
    )

    # Display results
    logger.info("\nConversion results:")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Images converted: {result.num_images_converted}")
    logger.info(f"  Objects converted: {result.num_objects_converted}")

    if result.success:
        logger.info(f"  Output file: {coco_file}")
        logger.info(
            f"  File size: {coco_file.stat().st_size if coco_file.exists() else 0} bytes"
        )

        # Display generated category information
        if "categories" in result.metadata:
            categories = result.metadata["categories"]
            logger.info(f"  Categories generated: {len(categories)}")
            for cat_id, cat_name in categories.items():
                logger.info(f"    - ID {cat_id}: {cat_name}")
    else:
        logger.error(f"  Error count: {len(result.errors)}")
        for error in result.errors:
            logger.error(f"    - {error}")

    if result.warnings:
        logger.warning(f"  Warning count: {len(result.warnings)}")
        for warning in result.warnings:
            logger.warning(f"    - {warning}")

    logger.info("\nExample completed!")


if __name__ == "__main__":
    main()
