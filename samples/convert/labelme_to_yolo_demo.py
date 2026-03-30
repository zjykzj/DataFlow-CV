#!/usr/bin/env python3
"""
LabelMe to YOLO format conversion example

Demonstrates how to use LabelMeAndYoloConverter to convert LabelMe format annotations to YOLO format.

Usage:
    python labelme_to_yolo_demo.py [--verbose]

Examples:
    python labelme_to_yolo_demo.py           # Normal mode
    python labelme_to_yolo_demo.py --verbose # Verbose logging mode
"""

import argparse
import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.convert import LabelMeAndYoloConverter
from dataflow.util import LoggingOperations, VerboseLoggingOperations


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LabelMe to YOLO format conversion example")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging mode")
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        log_ops = VerboseLoggingOperations()
        logger = log_ops.get_verbose_logger(
            name="labelme_to_yolo_demo",
            verbose=True,
            log_dir=str(project_root / "logs")
        )
        logger.info("Verbose logging mode enabled")
    else:
        log_ops = LoggingOperations()
        logger = log_ops.get_logger("labelme_to_yolo_demo", level="INFO")

    # Example data paths
    data_dir = project_root / "assets" / "test_data" / "det" / "labelme"
    class_file = data_dir / "classes.txt"
    output_dir = project_root / "samples" / "convert" / "output" / "labelme_to_yolo"

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.info("Please ensure sample data is prepared")
        return

    logger.info("=" * 50)
    logger.info("LabelMe to YOLO format conversion example")
    logger.info("=" * 50)

    # Create converter (LabelMe→YOLO direction)
    logger.info("Creating LabelMe→YOLO converter")
    converter = LabelMeAndYoloConverter(
        source_to_target=True,  # LabelMe→YOLO
        verbose=args.verbose,  # Verbose logging mode
        strict_mode=True,
        logger=logger,
    )

    # Perform conversion
    logger.info(f"Performing conversion:")
    logger.info(f"  Source directory: {data_dir}")
    logger.info(f"  Target directory: {output_dir}")
    logger.info(f"  Class file: {class_file}")

    result = converter.convert(
        source_path=str(data_dir),
        target_path=str(output_dir),
        class_file=str(class_file),
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
