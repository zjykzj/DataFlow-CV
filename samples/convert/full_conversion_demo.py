#!/usr/bin/env python3
"""
Complete format conversion chain example

Demonstrates how to convert LabelMe format to YOLO, then to COCO, then back to LabelMe,
verifying lossless conversion (except for RLE precision loss).

Usage:
    python full_conversion_demo.py [--verbose]

Examples:
    python full_conversion_demo.py           # Normal mode
    python full_conversion_demo.py --verbose # Verbose logging mode
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.convert import (CocoAndLabelMeConverter, LabelMeAndYoloConverter,
                              YoloAndCocoConverter)
from dataflow.util import LoggingOperations, VerboseLoggingOperations


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Complete format conversion chain example")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging mode")
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        log_ops = VerboseLoggingOperations()
        logger = log_ops.get_verbose_logger(
            name="full_conversion_demo",
            verbose=True,
            log_dir=str(project_root / "logs")
        )
        logger.info("Verbose logging mode enabled")
    else:
        log_ops = LoggingOperations()
        logger = log_ops.get_logger("full_conversion_demo", level="INFO")

    # Create temporary working directory
    temp_dir = Path(tempfile.mkdtemp(prefix="dataflow_convert_"))
    logger.info(f"Created temporary working directory: {temp_dir}")

    try:
        # Example data paths
        data_dir = project_root / "assets" / "test_data" / "det" / "labelme"
        class_file = data_dir / "classes.txt"

        if not data_dir.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
            return

        logger.info("=" * 50)
        logger.info("Complete format conversion chain example")
        logger.info("=" * 50)

        # Step 1: LabelMe → YOLO
        logger.info("\n1. LabelMe → YOLO conversion")
        yolo_dir = temp_dir / "yolo_output"

        converter1 = LabelMeAndYoloConverter(
            source_to_target=True, strict_mode=True, logger=logger
        )

        result1 = converter1.convert(
            source_path=str(data_dir),
            target_path=str(yolo_dir),
            class_file=str(class_file),
        )

        if not result1.success:
            logger.error("LabelMe→YOLO conversion failed")
            return

        # Step 2: YOLO → COCO
        logger.info("\n2. YOLO → COCO conversion")
        coco_file = temp_dir / "coco_output.json"

        converter2 = YoloAndCocoConverter(
            source_to_target=True, strict_mode=True, logger=logger
        )

        result2 = converter2.convert(
            source_path=str(yolo_dir / "labels"),
            target_path=str(coco_file),
            class_file=str(yolo_dir / "classes.txt"),
            image_dir=str(yolo_dir / "images"),
            do_rle=False,  # No RLE to ensure lossless
        )

        if not result2.success:
            logger.error("YOLO→COCO conversion failed")
            return

        # Step 3: COCO → LabelMe
        logger.info("\n3. COCO → LabelMe conversion")
        labelme_dir = temp_dir / "labelme_output"

        converter3 = CocoAndLabelMeConverter(
            source_to_target=True, strict_mode=True, logger=logger
        )

        result3 = converter3.convert(
            source_path=str(coco_file), target_path=str(labelme_dir)
        )

        if not result3.success:
            logger.error("COCO→LabelMe conversion failed")
            return

        # Verify results
        logger.info("\nConversion chain completed!")
        logger.info(f"Original LabelMe file count: {len(list(data_dir.glob('*.json')))}")
        logger.info(f"Final LabelMe file count: {len(list(labelme_dir.glob('*.json')))}")

        # Simple verification: file count consistency
        original_count = len(list(data_dir.glob("*.json")))
        final_count = len(list(labelme_dir.glob("*.json")))

        if original_count == final_count:
            logger.info("✓ File counts match, conversion chain complete")
        else:
            logger.warning(
                f"⚠ File counts differ: original={original_count}, final={final_count}"
            )

        logger.info(f"\nTemporary working directory: {temp_dir}")
        logger.info("(Will be automatically cleaned up after program ends)")

    finally:
        # Clean up temporary directory (may be kept for debugging in actual use)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")

    logger.info("\nExample completed!")


if __name__ == "__main__":
    main()
