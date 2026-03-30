#!/usr/bin/env python3
"""
COCO to LabelMe format conversion example

Demonstrates how to use CocoAndLabelMeConverter to convert COCO format annotations to LabelMe format.

Usage:
    python coco_to_labelme_demo.py [--verbose]

Examples:
    python coco_to_labelme_demo.py           # Normal mode
    python coco_to_labelme_demo.py --verbose # Verbose logging mode
"""

import argparse
import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.convert import CocoAndLabelMeConverter
from dataflow.util import LoggingOperations, VerboseLoggingOperations


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="COCO to LabelMe format conversion example")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging mode")
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        log_ops = VerboseLoggingOperations()
        logger = log_ops.get_verbose_logger(
            name="coco_to_labelme_demo",
            verbose=True,
            log_dir=str(project_root / "logs")
        )
        logger.info("Verbose logging mode enabled")
    else:
        log_ops = LoggingOperations()
        logger = log_ops.get_logger("coco_to_labelme_demo", level="INFO")

    # Example data paths
    data_dir = project_root / "assets" / "test_data" / "det" / "coco"
    coco_file = data_dir / "annotations.json"
    output_dir = project_root / "samples" / "convert" / "output" / "coco_to_labelme"

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.info("Please ensure sample data is prepared")
        return

    if not coco_file.exists():
        logger.error(f"COCO file does not exist: {coco_file}")
        return

    logger.info("=" * 50)
    logger.info("COCO to LabelMe format conversion example")
    logger.info("=" * 50)

    # Create converter (COCO→LabelMe direction)
    logger.info("Creating COCO→LabelMe converter")
    converter = CocoAndLabelMeConverter(
        source_to_target=True,  # COCO→LabelMe
        verbose=args.verbose,  # Verbose logging mode
        strict_mode=True,
        logger=logger,
    )

    # Perform conversion
    logger.info(f"Performing conversion:")
    logger.info(f"  Source file: {coco_file}")
    logger.info(f"  Target directory: {output_dir}")

    # COCO→LabelMe conversion automatically extracts category information from COCO JSON and generates classes.txt
    result = converter.convert(
        source_path=str(coco_file),
        target_path=str(output_dir),
        # image_dir is optional, if not provided will try to extract image paths from COCO JSON
    )

    # Display results
    logger.info("\nConversion results:")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Images converted: {result.num_images_converted}")
    logger.info(f"  Objects converted: {result.num_objects_converted}")

    if result.success:
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Generated files:")
        json_files = list(output_dir.glob("*.json"))
        txt_files = list(output_dir.glob("*.txt"))

        for file in json_files:
            logger.info(f"    - {file.relative_to(output_dir)}")

        for file in txt_files:
            logger.info(f"    - {file.relative_to(output_dir)}")

        logger.info(f"  Total generated {len(json_files)} LabelMe JSON files")
        logger.info(f"  Total generated {len(txt_files)} text files (including classes.txt)")

        # Check if classes.txt was generated
        classes_file = output_dir / "classes.txt"
        if classes_file.exists():
            logger.info(f"  Generated class file: {classes_file}")
            try:
                with open(classes_file, "r", encoding="utf-8") as f:
                    classes = [line.strip() for line in f if line.strip()]
                    logger.info(f"  Number of categories included: {len(classes)}")
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
