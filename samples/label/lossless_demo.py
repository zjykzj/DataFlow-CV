#!/usr/bin/env python3
"""
Lossless annotation processing demonstration.

Demonstrates the lossless read/write capabilities of the DataFlow-CV label module. This feature ensures that reading and rewriting annotation files
produces exactly the same output, regardless of coordinate conversions or format conversions.

Key features:
1. Original data preservation: Saves complete original annotation data when reading
2. Format identification: Tracks the source format of each annotation component
3. Original data priority writing: Uses original data over converted data when writing
4. Mixed data processing: Supports annotations containing both original and newly created data
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.label import (AnnotationFormat, CocoAnnotationHandler,
                            LabelMeAnnotationHandler, OriginalData,
                            YoloAnnotationHandler, verify_lossless_roundtrip)
from dataflow.util import LoggingOperations


def demo_labelme_lossless(logger):
    """Demonstrates lossless processing of LabelMe format"""
    logger.info("=" * 60)
    logger.info("LabelMe format lossless processing demonstration")
    logger.info("=" * 60)

    # Using object detection test data
    data_dir = project_root / "assets" / "test_data" / "det" / "labelme"
    class_file = data_dir / "classes.txt"

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return False

    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        output_dir = temp_dir_path / "output_labelme"

        # 1. Reading original data
        logger.info(f"1. Reading LabelMe annotation data: {data_dir}")
        handler = LabelMeAnnotationHandler(
            label_dir=str(data_dir),
            class_file=str(class_file) if class_file.exists() else None,
            strict_mode=True,
        )

        read_result = handler.read()
        if not read_result.success:
            logger.error(f"Reading failed: {read_result.message}")
            return False

        dataset = read_result.data
        logger.info(f"   Successfully read {len(dataset.images)} images")

        # Check original data preservation status
        image_with_original = 0
        objects_with_original = 0

        for image_ann in dataset.images:
            if image_ann.has_original_data():
                image_with_original += 1
                logger.debug(f"   Image {image_ann.image_id} saved original data")

            for obj in image_ann.objects:
                if obj.has_original_data():
                    objects_with_original += 1
                    # Verify original data format
                    if obj.original_data.format == AnnotationFormat.LABELME.value:
                        logger.debug(
                            f"     Object {obj.class_name} saved LabelMe original data"
                        )

        logger.info(f"   {image_with_original} images saved original data")
        logger.info(f"   {objects_with_original} objects saved original data")

        # 2. Write data (should use original data)
        logger.info(f"2. Writing LabelMe annotations to: {output_dir}")
        write_result = handler.write(dataset, str(output_dir))
        if not write_result.success:
            logger.error(f"Write failed: {write_result.message}")
            return False

        # 3. Verify lossless
        logger.info("3. Verify lossless")
        input_files = sorted(data_dir.glob("*.json"))
        output_files = sorted(output_dir.glob("*.json"))

        if len(input_files) != len(output_files):
            logger.error(
                f"File count mismatch: input={len(input_files)}, output={len(output_files)}"
            )
            return False

        all_match = True
        for in_file, out_file in zip(input_files, output_files):
            with open(in_file, "r", encoding="utf-8") as f1:
                data1 = json.load(f1)
            with open(out_file, "r", encoding="utf-8") as f2:
                data2 = json.load(f2)

            # Remove imageData field (may be null)
            data1.pop("imageData", None)
            data2.pop("imageData", None)

            if data1 != data2:
                logger.error(f"File {in_file.name} does not match")
                all_match = False
            else:
                logger.info(f"   ✓ {in_file.name}: File exactly matches")

        if all_match:
            logger.info("   ✓ LabelMe lossless verification passed")
            return True
        else:
            logger.error("   ✗ LabelMe lossless verification failed")
            return False


def demo_yolo_lossless(logger):
    """Demonstrates lossless processing of YOLO format"""
    logger.info("\n" + "=" * 60)
    logger.info("YOLO format lossless processing demonstration")
    logger.info("=" * 60)

    # Using object detection test data
    data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    class_file = data_dir / "classes.txt"
    labels_dir = data_dir / "labels"

    if not labels_dir.exists():
        logger.error(f"Label directory does not exist: {labels_dir}")
        return False

    image_dir = data_dir / "images"
    if not image_dir.exists():
        logger.error(f"Image directory does not exist: {image_dir}")
        return False

    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        output_dir = temp_dir_path / "output_yolo"

        # 1. Reading original data
        logger.info(f"1. Reading YOLO annotation data: {labels_dir}")
        handler = YoloAnnotationHandler(
            image_dir=str(image_dir),
            label_dir=str(labels_dir),
            class_file=str(class_file) if class_file.exists() else None,
            strict_mode=True,
        )

        read_result = handler.read()
        if not read_result.success:
            logger.error(f"Reading failed: {read_result.message}")
            return False

        dataset = read_result.data
        logger.info(f"   Successfully read {len(dataset.images)} images")

        # Check original data preservation status
        objects_with_original = 0
        for image_ann in dataset.images:
            for obj in image_ann.objects:
                if obj.has_original_data():
                    objects_with_original += 1
                    if obj.original_data.format == AnnotationFormat.YOLO.value:
                        logger.debug(f"     Object {obj.class_name} saved YOLO original data")
                        # Display original line data
                        if "line" in obj.original_data.raw_data:
                            logger.debug(
                                f"       Original line: {obj.original_data.raw_data['line'].strip()}"
                            )

        logger.info(f"   {objects_with_original} objects saved original data")

        # 2. Write data
        logger.info(f"2. Writing YOLO annotations to: {output_dir}")
        write_result = handler.write(dataset, str(output_dir))
        if not write_result.success:
            logger.error(f"Write failed: {write_result.message}")
            return False

        # 3. Verify lossless
        logger.info("3. Verify lossless")
        input_files = sorted(labels_dir.glob("*.txt"))
        output_files = sorted(output_dir.glob("*.txt"))

        if len(input_files) != len(output_files):
            logger.error(
                f"File count mismatch: input={len(input_files)}, output={len(output_files)}"
            )
            return False

        all_match = True
        for in_file, out_file in zip(input_files, output_files):
            with open(in_file, "r", encoding="utf-8") as f1:
                lines1 = [line.rstrip() for line in f1.readlines()]
            with open(out_file, "r", encoding="utf-8") as f2:
                lines2 = [line.rstrip() for line in f2.readlines()]

            if lines1 != lines2:
                logger.error(f"File {in_file.name} does not match")
                logger.error(f"  Input lines: {lines1}")
                logger.error(f"  Output lines: {lines2}")
                all_match = False
            else:
                logger.info(f"   ✓ {in_file.name}: File exactly matches")

        if all_match:
            logger.info("   ✓ YOLO lossless verification passed")
            return True
        else:
            logger.error("   ✗ YOLO lossless verification failed")
            return False


def demo_coco_lossless(logger):
    """Demonstrates lossless processing of COCO format"""
    logger.info("\n" + "=" * 60)
    logger.info("COCO format lossless processing demonstration")
    logger.info("=" * 60)

    # Using object detection test data
    data_file = (
        project_root / "assets" / "test_data" / "det" / "coco" / "annotations.json"
    )

    if not data_file.exists():
        logger.error(f"COCO file does not exist: {data_file}")
        return False

    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        output_file = temp_dir_path / "output_coco.json"

        # 1. Reading original data
        logger.info(f"1. Reading COCO annotation data: {data_file}")
        handler = CocoAnnotationHandler(
            annotation_file=str(data_file), strict_mode=True
        )

        read_result = handler.read()
        if not read_result.success:
            logger.error(f"Reading failed: {read_result.message}")
            return False

        dataset = read_result.data
        logger.info(f"   Successfully read {len(dataset.images)} images")

        # Check original data preservation status
        images_with_original = 0
        objects_with_original = 0
        objects_with_rle = 0

        for image_ann in dataset.images:
            if image_ann.has_original_data():
                images_with_original += 1

            for obj in image_ann.objects:
                if obj.has_original_data():
                    objects_with_original += 1
                    if obj.original_data.format == AnnotationFormat.COCO.value:
                        logger.debug(f"     Object {obj.class_name} saved COCO original data")

                # Check RLE data preservation
                if obj.segmentation and obj.segmentation.has_rle():
                    objects_with_rle += 1
                    logger.debug(f"     Object {obj.class_name} saved RLE data")

        logger.info(f"   {images_with_original} images saved original data")
        logger.info(f"   {objects_with_original} objects saved original data")
        logger.info(f"   {objects_with_rle} objects saved RLE data")

        # 2. Write data (preserve polygon format)
        logger.info(f"2. Writing COCO annotations to: {output_file} (preserve polygon format)")
        write_result = handler.write(dataset, str(output_file), output_rle=False)
        if not write_result.success:
            logger.error(f"Write failed: {write_result.message}")
            return False

        # 3. Verify lossless
        logger.info("3. Verify lossless")
        with open(data_file, "r", encoding="utf-8") as f1:
            data1 = json.load(f1)
        with open(output_file, "r", encoding="utf-8") as f2:
            data2 = json.load(f2)

        # Remove auto-generated fields
        for data in [data1, data2]:
            data.pop("__coco_original_data__", None)
            # Remove potentially different description fields
            for field in [
                "description",
                "url",
                "version",
                "year",
                "contributor",
                "date_created",
            ]:
                data.pop(field, None)

        # Compare annotations (IDs may be regenerated, compare content)
        if len(data1["annotations"]) != len(data2["annotations"]):
            logger.error(
                f"Annotation count mismatch: input={len(data1['annotations'])}, output={len(data2['annotations'])}"
            )
            return False

        all_match = True
        for i, (ann1, ann2) in enumerate(
            zip(data1["annotations"], data2["annotations"])
        ):
            # Remove ID comparison
            ann1_copy = {k: v for k, v in ann1.items() if k != "id"}
            ann2_copy = {k: v for k, v in ann2.items() if k != "id"}

            if ann1_copy != ann2_copy:
                logger.error(f"Annotation {i} content differs")
                logger.error(f"  Input: {ann1_copy}")
                logger.error(f"  Output: {ann2_copy}")
                all_match = False

        # Compare images and categories
        if data1["images"] != data2["images"]:
            logger.error("Image information differs")
            all_match = False

        if data1["categories"] != data2["categories"]:
            logger.error("Category information differs")
            all_match = False

        if all_match:
            logger.info("   ✓ COCO lossless verification passed")
            return True
        else:
            logger.error("   ✗ COCO lossless verification failed")
            return False


def demo_utility_functions(logger):
    """Demonstrates lossless verification utility functions"""
    logger.info("\n" + "=" * 60)
    logger.info("Lossless verification utility functions demonstration")
    logger.info("=" * 60)

    # Using object detection test data
    labelme_dir = project_root / "assets" / "test_data" / "det" / "labelme"
    yolo_dir = project_root / "assets" / "test_data" / "det" / "yolo" / "labels"
    yolo_class_file = (
        project_root / "assets" / "test_data" / "det" / "yolo" / "classes.txt"
    )
    coco_file = (
        project_root / "assets" / "test_data" / "det" / "coco" / "annotations.json"
    )

    # Test LabelMe verification
    logger.info("1. Test LabelMe lossless verification")
    if labelme_dir.exists():
        # Create custom handler (demonstrates usage method)
        from dataflow.label.labelme_handler import LabelMeAnnotationHandler

        result = verify_lossless_roundtrip(
            input_path=str(labelme_dir),
            output_path="/tmp/test_output",
            handler_class=LabelMeAnnotationHandler,
        )
        logger.info(f"   LabelMe lossless verification result: {'Passed' if result else 'Failed'}")
    else:
        logger.warning("   LabelMe test data does not exist")

    # Test YOLO verification
    logger.info("2. Test YOLO lossless verification")
    if yolo_dir.exists() and yolo_class_file.exists():
        # Note: YOLO verification requires class file parameter, simplified demonstration here
        logger.info("   YOLO verification requires custom handler, skipping demonstration")
    else:
        logger.warning("   YOLO test data does not exist")

    # Test COCO verification
    logger.info("3. Test COCO lossless verification")
    if coco_file.exists():
        from dataflow.label.coco_handler import CocoAnnotationHandler

        result = verify_lossless_roundtrip(
            input_path=str(coco_file),
            output_path="/tmp/test_output_coco.json",
            handler_class=CocoAnnotationHandler,
        )
        logger.info(f"   COCO lossless verification result: {'Passed' if result else 'Failed'}")
    else:
        logger.warning("   COCO test data does not exist")

    logger.info("\nUtility functions demonstration completed")


def main():
    """Main function"""
    # Configure logging
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("lossless_demo", level="INFO")

    logger.info("=" * 60)
    logger.info("DataFlow-CV lossless annotation processing demonstration")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This demonstration shows the lossless read/write capabilities of the DataFlow-CV label module.")
    logger.info("This feature ensures that reading and rewriting annotation files produces exactly the same output.")
    logger.info("")

    # Run each demonstration
    success_count = 0
    total_demos = 3

    try:
        if demo_labelme_lossless(logger):
            success_count += 1
    except Exception as e:
        logger.error(f"LabelMe demonstration exception: {e}")

    try:
        if demo_yolo_lossless(logger):
            success_count += 1
    except Exception as e:
        logger.error(f"YOLO demonstration exception: {e}")

    try:
        if demo_coco_lossless(logger):
            success_count += 1
    except Exception as e:
        logger.error(f"COCO demonstration exception: {e}")

    # Demonstrate utility functions
    try:
        demo_utility_functions(logger)
    except Exception as e:
        logger.error(f"Utility functions demonstration exception: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Demonstration summary")
    logger.info("=" * 60)
    logger.info(f"Successful demonstrations: {success_count}/{total_demos}")

    if success_count == total_demos:
        logger.info("✓ All lossless function demonstrations successful")
        logger.info("✓ DataFlow-CV label module provides completely lossless annotation processing")
    else:
        logger.warning(f"⚠ {total_demos - success_count} demonstrations failed")
        logger.info("Please check test data or module implementation")

    logger.info("\nDemonstration completed!")


if __name__ == "__main__":
    main()
