#!/usr/bin/env python3
"""
COCO to YOLO format conversion example

Demonstrates how to use YoloAndCocoConverter to convert COCO format annotations to YOLO format.

Usage:
    python coco_to_yolo_demo.py [--verbose]

Examples:
    python coco_to_yolo_demo.py           # Normal mode
    python coco_to_yolo_demo.py --verbose # Verbose logging mode
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
    parser = argparse.ArgumentParser(description="COCO to YOLO format conversion example")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging mode")
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        log_ops = VerboseLoggingOperations()
        logger = log_ops.get_verbose_logger(
            name="coco_to_yolo_demo",
            verbose=True,
            log_dir=str(project_root / "logs")
        )
        logger.info("Verbose logging mode enabled")
    else:
        log_ops = LoggingOperations()
        logger = log_ops.get_logger("coco_to_yolo_demo", level="INFO")

    # Example data paths
    data_dir = project_root / "assets" / "test_data" / "det" / "coco"
    coco_file = data_dir / "annotations.json"
    output_dir = project_root / "samples" / "convert" / "output" / "coco_to_yolo"

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.info("Please ensure sample data is prepared")
        return

    if not coco_file.exists():
        logger.error(f"COCO file does not exist: {coco_file}")
        return

    logger.info("=" * 50)
    logger.info("COCO to YOLO format conversion example")
    logger.info("=" * 50)

    # Create converter (COCO→YOLO direction)
    logger.info("Creating COCO→YOLO converter")
    converter = YoloAndCocoConverter(
        source_to_target=False,  # COCO→YOLO
        verbose=args.verbose,  # Verbose logging mode
        strict_mode=True,
        logger=logger,
    )

    # Perform conversion
    logger.info(f"Performing conversion:")
    logger.info(f"  Source file: {coco_file}")
    logger.info(f"  Target directory: {output_dir}")

    # COCO→YOLO conversion automatically extracts category information from COCO JSON
    result = converter.convert(
        source_path=str(coco_file),
        target_path=str(output_dir),
        # image_dir is optional, if not provided will try to extract from COCO JSON
    )

    # Display results
    logger.info("\nConversion results:")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Images converted: {result.num_images_converted}")
    logger.info(f"  Objects converted: {result.num_objects_converted}")

    if result.success:
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Generated files:")
        for file in output_dir.rglob("*"):
            if file.is_file():
                logger.info(f"    - {file.relative_to(output_dir)}")

        # Check if classes.txt was generated
        classes_file = output_dir / "classes.txt"
        if classes_file.exists():
            logger.info(f"  Generated class file: {classes_file}")
            try:
                with open(classes_file, "r", encoding="utf-8") as f:
                    classes = [line.strip() for line in f if line.strip()]
                    logger.info(f"  Number of categories included: {len(classes)}")
                    for i, cls in enumerate(classes):
                        logger.info(f"    - ID {i}: {cls}")
            except Exception as e:
                logger.warning(f"  Failed to read class file: {e}")
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
