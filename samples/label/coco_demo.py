#!/usr/bin/env python3
"""
COCO annotation format processing example

Demonstrates how to use CocoAnnotationHandler to read, process, and write COCO format annotations.
Supports both RLE and polygon formats.
"""

import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.label import CocoAnnotationHandler
from dataflow.util import LoggingOperations


def demo_standard_format():
    """Standard COCO format (polygon) demonstration"""
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("coco_demo_std", level="INFO")

    logger.info("COCO standard format (polygon) annotation processing example")
    logger.info("=" * 50)

    # Example data path
    data_dir = project_root / "assets" / "test_data" / "det" / "coco"
    annotation_file = data_dir / "annotations.json"

    if not annotation_file.exists():
        logger.error(f"Annotation file does not exist: {annotation_file}")
        logger.info("Please ensure sample data is prepared")
        return

    # Create COCO handler
    logger.info(f"Creating COCO handler, annotation file: {annotation_file}")
    handler = CocoAnnotationHandler(
        annotation_file=str(annotation_file), strict_mode=True, logger=logger
    )

    # Read annotation data
    logger.info("Reading COCO annotations...")
    result = handler.read()

    if not result.success:
        logger.error(f"Read failed: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")
        return

    # Display RLE format detection result
    logger.info(f"RLE format detection: {handler.is_rle}")

    logger.info(f"\nSuccessfully read annotation information:")
    logger.info(f"  Number of images: {len(result.data.images)}")
    logger.info(f"  Number of annotations: {result.data.num_objects}")
    logger.info(f"  Number of categories: {len(result.data.categories)}")

    # Display dataset information
    if result.data.dataset_info:
        logger.info("\nDataset information:")
        for key, value in result.data.dataset_info.items():
            logger.info(f"  {key}: {value}")

    # Display details of the first image
    if result.data.images:
        img = result.data.images[0]
        logger.info(f"\nFirst image information:")
        logger.info(f"  Image ID: {img.image_id}")
        logger.info(f"  Path: {img.image_path}")
        logger.info(f"  Dimensions: {img.width}x{img.height}")
        logger.info(f"  Number of objects: {len(img.objects)}")

        for i, obj in enumerate(img.objects[:3]):  # Display first 3 objects
            logger.info(f"  Object{i+1}: {obj.class_name} (ID: {obj.class_id})")
            if obj.bbox:
                logger.info(
                    f"    Bounding box: x={obj.bbox.x:.3f}, y={obj.bbox.y:.3f}, "
                    f"w={obj.bbox.width:.3f}, h={obj.bbox.height:.3f}"
                )
            if obj.segmentation:
                logger.info(f"    Segmentation points count: {len(obj.segmentation.points)}")
                # Display first 3 points
                for j, (x, y) in enumerate(obj.segmentation.points[:3]):
                    logger.info(f"      Point{j+1}: x={x:.3f}, y={y:.3f}")

    # Format conversion example
    logger.info("\nPerforming format conversion test...")
    output_file = data_dir / "annotations_converted.json"

    write_result = handler.write(result.data, str(output_file))

    if write_result.success:
        logger.info(f"Successfully written to: {output_file}")

        # Verify written file
        logger.info("Verifying written file...")
        verify_handler = CocoAnnotationHandler(
            annotation_file=str(output_file), strict_mode=False  # Use non-strict mode for verification
        )
        verify_result = verify_handler.read()

        if verify_result.success:
            logger.info("Verification passed! Written file can be read correctly.")
        else:
            logger.warning(f"Verification warning: {verify_result.message}")
    else:
        logger.error(f"Write failed: {write_result.message}")

    logger.info("\nStandard format example completed!\n")


def demo_rle_format():
    """RLE format COCO annotation demonstration"""
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("coco_demo_rle", level="INFO")

    logger.info("COCO RLE format annotation processing example")
    logger.info("=" * 50)

    # Example data path
    data_dir = project_root / "assets" / "test_data" / "seg" / "coco"
    annotation_file = data_dir / "annotations-rle.json"

    if not annotation_file.exists():
        logger.error(f"Annotation file does not exist: {annotation_file}")
        logger.info("Please ensure sample data is prepared")
        return

    # Check if pycocotools is installed
    try:
        from pycocotools import mask as coco_mask

        logger.info("pycocotools is installed, RLE processing supported")
    except ImportError:
        logger.warning("pycocotools not installed, RLE functionality limited")
        logger.info("Install command: pip install pycocotools")
        # Continue demonstration, but RLE decoding will fail

    # Create COCO handler
    logger.info(f"Creating COCO handler, annotation file: {annotation_file}")
    handler = CocoAnnotationHandler(
        annotation_file=str(annotation_file),
        strict_mode=False,  # Use non-strict mode because RLE decoding may fail
        logger=logger,
    )

    # Read annotation data
    logger.info("Reading COCO RLE annotations...")
    result = handler.read()

    if not result.success:
        logger.error(f"Read failed: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")
        return

    # Display RLE format detection result
    logger.info(f"RLE format detection: {handler.is_rle}")

    logger.info(f"\nSuccessfully read annotation information:")
    logger.info(f"  Number of images: {len(result.data.images)}")
    logger.info(f"  Number of annotations: {result.data.num_objects}")
    logger.info(f"  Number of categories: {len(result.data.categories)}")

    # Display annotation type
    logger.info(f"\nAnnotation type detection:")
    logger.info(f"  Object detection: {handler.is_det}")
    logger.info(f"  Instance segmentation: {handler.is_seg}")

    # Test RLE output
    logger.info("\nTesting RLE format output...")
    output_file = data_dir / "annotations_rle_output.json"

    # Try to use RLE format output
    write_result = handler.write(result.data, str(output_file), output_rle=True)

    if write_result.success:
        logger.info(f"Successfully written RLE format to: {output_file}")

        # Check if output contains RLE
        import json

        with open(output_file, "r", encoding="utf-8") as f:
            output_data = json.load(f)

        rle_count = sum(
            1
            for ann in output_data["annotations"]
            if isinstance(ann.get("segmentation"), dict)
            and "counts" in ann.get("segmentation", {})
        )
        logger.info(f"  Output contains {rle_count} RLE format annotations")

    else:
        logger.warning(f"RLE format write failed, trying polygon format...")
        write_result = handler.write(result.data, str(output_file), output_rle=False)
        if write_result.success:
            logger.info(f"Successfully written polygon format to: {output_file}")
        else:
            logger.error(f"Write failed: {write_result.message}")

    logger.info("\nRLE format example completed!\n")


def demo_format_conversion():
    """Format conversion demonstration: polygon <-> RLE"""
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("coco_demo_conv", level="INFO")

    logger.info("COCO format conversion demonstration")
    logger.info("=" * 50)

    # Use standard format data
    data_dir = project_root / "assets" / "test_data" / "det" / "coco"
    annotation_file = data_dir / "annotations.json"

    if not annotation_file.exists():
        logger.error(f"Annotation file does not exist: {annotation_file}")
        return

    # Check pycocotools
    try:
        from pycocotools import mask as coco_mask

        has_pycocotools = True
    except ImportError:
        has_pycocotools = False
        logger.warning("pycocotools not installed, skipping RLE conversion test")
        logger.info("Install command: pip install pycocotools")

    handler = CocoAnnotationHandler(
        annotation_file=str(annotation_file), strict_mode=True, logger=logger
    )

    # Read data
    result = handler.read()
    if not result.success:
        logger.error(f"Read failed: {result.message}")
        return

    logger.info(f"Read successful: {result.data.num_objects} objects")

    # Conversion test
    test_dir = data_dir / "conversion_test"
    test_dir.mkdir(exist_ok=True)

    # 1. Original format output
    output1 = test_dir / "original.json"
    handler.write(result.data, str(output1), output_rle=False)
    logger.info(f"1. Polygon format output: {output1}")

    if has_pycocotools:
        # 2. RLE format output
        output2 = test_dir / "rle.json"
        handler.write(result.data, str(output2), output_rle=True)
        logger.info(f"2. RLE format output: {output2}")

        # 3. Read RLE output and verify
        handler2 = CocoAnnotationHandler(
            annotation_file=str(output2), strict_mode=False, logger=logger
        )
        result2 = handler2.read()
        if result2.success:
            logger.info(f"3. RLE file verification: Successfully read {result2.data.num_objects} objects")
            logger.info(f"   RLE format detection: {handler2.is_rle}")
        else:
            logger.warning(f"3. RLE file verification failed: {result2.message}")

    logger.info("\nFormat conversion demonstration completed!\n")


def main():
    """Main function"""
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("coco_demo", level="INFO")

    logger.info("=" * 60)
    logger.info("COCO annotation format processing example")
    logger.info("=" * 60)

    # Run standard format demonstration
    demo_standard_format()

    # Run RLE format demonstration
    demo_rle_format()

    # Run format conversion demonstration
    demo_format_conversion()

    logger.info("All examples completed!")


if __name__ == "__main__":
    main()
