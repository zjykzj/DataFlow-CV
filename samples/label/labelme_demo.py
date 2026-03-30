#!/usr/bin/env python3
"""
LabelMe annotation format processing example

Demonstrates how to use LabelMeAnnotationHandler to read, process, and write LabelMe format annotations.
"""

import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.label import LabelMeAnnotationHandler
from dataflow.util import LoggingOperations


def main():
    """Main function"""
    # Configure logging
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("labelme_demo", level="INFO")

    # Example data path (using LabelMe test data for object detection)
    data_dir = project_root / "assets" / "test_data" / "det" / "labelme"
    class_file = data_dir / "classes.txt"

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.info("Please ensure sample data is prepared")
        return

    logger.info("=" * 50)
    logger.info("LabelMe annotation processing example")
    logger.info("=" * 50)

    # Create LabelMe handler
    logger.info(f"Creating LabelMe handler, data directory: {data_dir}")
    handler = LabelMeAnnotationHandler(
        label_dir=str(data_dir),
        class_file=str(class_file) if class_file.exists() else None,
        strict_mode=True,
        logger=logger,
    )

    # Read annotation data
    logger.info("Reading LabelMe annotations...")
    result = handler.read()

    if not result.success:
        logger.error(f"Read failed: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")
        return

    logger.info(f"Successfully read annotations for {len(result.data.images)} images")
    logger.info(f"Number of categories: {len(result.data.categories)}")

    # Display partial information
    for i, image_ann in enumerate(result.data.images[:3]):  # Only show first 3 images
        logger.info(f"\nImage {i+1}: {image_ann.image_id}")
        logger.info(f"  Path: {image_ann.image_path}")
        logger.info(f"  Dimensions: {image_ann.width}x{image_ann.height}")
        logger.info(f"  Number of objects: {len(image_ann.objects)}")

        for j, obj in enumerate(image_ann.objects[:2]):  # Display first 2 objects per image
            logger.info(f"    Object {j+1}: {obj.class_name} (ID: {obj.class_id})")
            if obj.bbox:
                logger.info(
                    f"      Bounding box: x={obj.bbox.x:.3f}, y={obj.bbox.y:.3f}, "
                    f"w={obj.bbox.width:.3f}, h={obj.bbox.height:.3f}"
                )

    logger.info("\nExample completed!")


if __name__ == "__main__":
    main()
