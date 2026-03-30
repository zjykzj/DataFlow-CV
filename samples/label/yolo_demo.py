#!/usr/bin/env python3
"""
YOLO annotation format processing example

Demonstrates how to use YoloAnnotationHandler to read, process, and write YOLO format annotations.
Supports automatic detection of object detection and instance segmentation formats.
"""

import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.label import YoloAnnotationHandler
from dataflow.util import LoggingOperations


def demo_detection():
    """Object detection data demonstration"""
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("yolo_demo_det", level="INFO")

    logger.info("YOLO object detection annotation processing example")
    logger.info("=" * 50)

    # Example data path
    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    class_file = data_dir / "classes.txt"
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.info("Please ensure sample data is prepared")
        return

    # Create YOLO handler
    logger.info(f"Creating YOLO handler:")
    logger.info(f"  Label directory: {label_dir}")
    logger.info(f"  Class file: {class_file}")
    logger.info(f"  Image directory: {image_dir}")

    handler = YoloAnnotationHandler(
        label_dir=str(label_dir),
        class_file=str(class_file),
        image_dir=str(image_dir),
        strict_mode=True,
        logger=logger,
    )

    # Reading annotation data
    logger.info("\nReading YOLO annotations...")
    result = handler.read()

    if not result.success:
        logger.error(f"Read failed: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")
        return

    # Showing annotation type detection results
    logger.info(f"Annotation type detection:")
    logger.info(f"  Object detection: {handler.is_det}")
    logger.info(f"  Instance segmentation: {handler.is_seg}")

    logger.info(f"\nSuccessfully read annotations for {len(result.data.images)} images")
    logger.info(f"Number of categories: {len(result.data.categories)}")

    # Display first image details
    if result.data.images:
        img = result.data.images[0]
        logger.info(f"\nFirst image information:")
        logger.info(f"  Image ID: {img.image_id}")
        logger.info(f"  Path: {img.image_path}")
        logger.info(f"  Dimensions: {img.width}x{img.height}")
        logger.info(f"  Number of objects: {len(img.objects)}")

        for i, obj in enumerate(img.objects[:3]):  # Display first 3 objects
            logger.info(f"  Object {i+1}: {obj.class_name} (ID: {obj.class_id})")
            if obj.bbox:
                logger.info(
                    f"    Bounding box: x={obj.bbox.x:.3f}, y={obj.bbox.y:.3f}, "
                    f"w={obj.bbox.width:.3f}, h={obj.bbox.height:.3f}"
                )

    # Format conversion example: YOLO → unified format → new YOLO
    logger.info("\nPerforming format conversion test...")
    output_dir = data_dir / "output"
    output_dir.mkdir(exist_ok=True)

    write_result = handler.write(result.data, str(output_dir))

    if write_result.success:
        logger.info(f"Successfully written to: {output_dir}")
        logger.info(f"Generated file count: {len(list(output_dir.glob('*.txt')))}")
    else:
        logger.error(f"Write failed: {write_result.message}")

    logger.info("\nObject detection example completed!\n")


def demo_segmentation():
    """Instance segmentation data demonstration"""
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("yolo_demo_seg", level="INFO")

    logger.info("YOLO instance segmentation annotation processing example")
    logger.info("=" * 50)

    # Example data path
    data_dir = project_root / "assets" / "test_data" / "seg" / "yolo"
    class_file = data_dir / "classes.txt"
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.info("Please ensure sample data is prepared")
        return

    # Create YOLO handler
    logger.info(f"Creating YOLO handler:")
    logger.info(f"  Label directory: {label_dir}")
    logger.info(f"  Class file: {class_file}")
    logger.info(f"  Image directory: {image_dir}")

    handler = YoloAnnotationHandler(
        label_dir=str(label_dir),
        class_file=str(class_file),
        image_dir=str(image_dir),
        strict_mode=True,
        logger=logger,
    )

    # Reading annotation data
    logger.info("\nReading YOLO annotations...")
    result = handler.read()

    if not result.success:
        logger.error(f"Read failed: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")
        return

    # Showing annotation type detection results
    logger.info(f"Annotation type detection:")
    logger.info(f"  Object detection: {handler.is_det}")
    logger.info(f"  Instance segmentation: {handler.is_seg}")

    logger.info(f"\nSuccessfully read annotations for {len(result.data.images)} images")
    logger.info(f"Number of categories: {len(result.data.categories)}")

    # Display first image details
    if result.data.images:
        img = result.data.images[0]
        logger.info(f"\nFirst image information:")
        logger.info(f"  Image ID: {img.image_id}")
        logger.info(f"  Path: {img.image_path}")
        logger.info(f"  Dimensions: {img.width}x{img.height}")
        logger.info(f"  Number of objects: {len(img.objects)}")

        for i, obj in enumerate(img.objects[:3]):  # Display first 3 objects
            logger.info(f"  Object {i+1}: {obj.class_name} (ID: {obj.class_id})")
            if obj.segmentation:
                logger.info(f"    Segmentation points: {len(obj.segmentation.points)}")
                # Display first 3 points
                for j, (x, y) in enumerate(obj.segmentation.points[:3]):
                    logger.info(f"      Point {j+1}: x={x:.3f}, y={y:.3f}")

    # Format conversion example: YOLO → unified format → new YOLO
    logger.info("\nPerforming format conversion test...")
    output_dir = data_dir / "output"
    output_dir.mkdir(exist_ok=True)

    write_result = handler.write(result.data, str(output_dir))

    if write_result.success:
        logger.info(f"Successfully written to: {output_dir}")
        logger.info(f"Generated file count: {len(list(output_dir.glob('*.txt')))}")
    else:
        logger.error(f"Write failed: {write_result.message}")

    logger.info("\nInstance segmentation example completed!\n")


def main():
    """Main function"""
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("yolo_demo", level="INFO")

    logger.info("=" * 60)
    logger.info("YOLO annotation format processing example")
    logger.info("=" * 60)

    # Running object detection demonstration
    demo_detection()

    # Running instance segmentation demonstration
    demo_segmentation()

    logger.info("All examples completed!")


if __name__ == "__main__":
    main()
