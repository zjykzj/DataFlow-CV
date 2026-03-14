#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/10
@File    : api_yolo_to_coco.py
@Author  : DataFlow Team
@Description: Python API example for YOLO to COCO conversion

This script demonstrates how to use the DataFlow-CV Python API
for converting YOLO format to COCO JSON format.
"""

import os
import sys
import json
import tempfile
import shutil
import stat
from pathlib import Path

# Add parent directory to path to import dataflow
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import DataFlow-CV
import dataflow
from dataflow import YoloToCocoConverter
from dataflow.config import Config


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def create_test_paths():
    """创建跨平台的测试路径"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="dataflow_test_")

    # 不存在的路径（用于测试无效路径错误）
    nonexistent_path = os.path.join(temp_dir, "nonexistent_subdir", "file.txt")

    # 尝试创建只读目录（在Windows上可能失败）
    read_only_dir = os.path.join(temp_dir, "readonly_dir")
    os.makedirs(read_only_dir, exist_ok=True)

    try:
        # 尝试设置为只读
        os.chmod(read_only_dir, stat.S_IRUSR | stat.S_IXUSR)
        read_only_path = os.path.join(read_only_dir, "no_permission.txt")
    except (OSError, PermissionError):
        # Windows上无法设置只读目录，使用普通路径
        read_only_path = os.path.join(read_only_dir, "no_permission.txt")

    # 临时文件路径
    temp_file_path = os.path.join(temp_dir, "temp_file.txt")

    return {
        "temp_dir": temp_dir,
        "nonexistent_path": nonexistent_path,
        "read_only_path": read_only_path,
        "temp_file_path": temp_file_path
    }


def create_sample_yolo_data():
    """Create sample YOLO data for demonstration."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="yolo2coco_api_demo_")
    print(f"Created temporary directory: {temp_dir}")

    # Create images directory (with empty files for demonstration)
    images_dir = os.path.join(temp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Create sample images
    image_data = [
        ("image1.jpg", 800, 600),
        ("image2.jpg", 1024, 768),
        ("image3.jpg", 640, 480),
        ("image4.jpg", 1280, 720),
        ("empty.jpg", 800, 600),  # Image without annotations
    ]

    for filename, width, height in image_data:
        img_path = os.path.join(images_dir, filename)
        # Create empty file (in real usage, these would be actual images)
        with open(img_path, 'wb') as f:
            f.write(b"")
        print(f"Created sample image: {filename} ({width}x{height})")

    # Create YOLO labels directory
    labels_dir = os.path.join(temp_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    # Create label files (matching image names)
    # image1.jpg: two persons
    label1 = os.path.join(labels_dir, "image1.txt")
    with open(label1, 'w', encoding='utf-8') as f:
        # person: class 0
        f.write("0 0.3 0.4 0.2 0.3\n")   # First person
        f.write("0 0.6 0.5 0.15 0.25\n") # Second person

    # image2.jpg: one car
    label2 = os.path.join(labels_dir, "image2.txt")
    with open(label2, 'w', encoding='utf-8') as f:
        # car: class 1
        f.write("1 0.5 0.5 0.25 0.2\n")

    # image3.jpg: one person
    label3 = os.path.join(labels_dir, "image3.txt")
    with open(label3, 'w', encoding='utf-8') as f:
        # person: class 0
        f.write("0 0.4 0.4 0.3 0.3\n")

    # image4.jpg: segmentation example (polygon)
    label4 = os.path.join(labels_dir, "image4.txt")
    with open(label4, 'w', encoding='utf-8') as f:
        # car segmentation: class 1 with polygon points
        # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
        f.write("1 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6\n")

    # empty.jpg: no label file (to demonstrate handling)
    print(f"Created YOLO labels in: {labels_dir}")

    # Create YOLO classes file
    classes_file = os.path.join(temp_dir, "classes.names")
    with open(classes_file, 'w', encoding='utf-8') as f:
        f.write("person\n")
        f.write("car\n")
        f.write("bicycle\n")  # Class 2, even though not used in this example

    print(f"Created classes file: {classes_file}")

    return temp_dir, images_dir, labels_dir, classes_file


def demo_convenience_function(images_dir, labels_dir, classes_file, output_json):
    """Demonstrate using the convenience function."""
    print_header("USING CONVENIENCE FUNCTION")

    print(f"\nCode:")
    print(f"  import dataflow")
    print(f"  result = dataflow.yolo_to_coco(")
    print(f"      '{images_dir}',")
    print(f"      '{labels_dir}',")
    print(f"      '{classes_file}',")
    print(f"      '{output_json}'")
    print(f"  )")

    print(f"\nExecuting...")
    try:
        result = dataflow.yolo_to_coco(
            images_dir, labels_dir, classes_file, output_json
        )

        print(f"\n✅ Success!")
        print(f"\nResult keys: {list(result.keys())}")

        # Show important statistics
        print(f"\nConversion statistics:")
        print(f"  - Images processed: {result.get('images_processed', 0)}")
        print(f"  - Annotations processed: {result.get('annotations_processed', 0)}")
        print(f"  - Categories found: {result.get('categories_found', 0)}")
        print(f"  - Image directory: {result.get('image_dir', 'N/A')}")
        print(f"  - Label directory: {result.get('label_dir', 'N/A')}")
        print(f"  - Classes file: {result.get('classes_file', 'N/A')}")
        print(f"  - COCO JSON path: {result.get('coco_json_path', 'N/A')}")
        print(f"  - Segmentation mode: {result.get('segmentation_mode', False)}")

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def demo_converter_class(images_dir, labels_dir, classes_file, output_json):
    """Demonstrate using the converter class directly."""
    print_header("USING CONVERTER CLASS")

    print(f"\nCode:")
    print(f"  from dataflow import YoloToCocoConverter")
    print(f"  converter = YoloToCocoConverter(verbose=True)")
    print(f"  result = converter.convert(")
    print(f"      '{images_dir}',")
    print(f"      '{labels_dir}',")
    print(f"      '{classes_file}',")
    print(f"      '{output_json}'")
    print(f"  )")

    print(f"\nExecuting...")
    try:
        converter = YoloToCocoConverter(verbose=True)
        print(f"  Converter created: {converter.__class__.__name__}")
        print(f"  Verbose mode: {converter.verbose}")

        result = converter.convert(
            images_dir, labels_dir, classes_file, output_json
        )

        print(f"\n✅ Success!")
        print(f"\nResult type: {type(result)}")

        # Show additional information available through converter
        print(f"\nConverter capabilities:")
        print(f"  - Can validate paths: {hasattr(converter, 'validate_input_path')}")
        print(f"  - Has logger: {hasattr(converter, 'logger')}")
        print(f"  - Can ensure directories: {hasattr(converter, 'ensure_directory')}")

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def demo_segmentation_mode(images_dir, labels_dir, classes_file, output_json):
    """Demonstrate segmentation mode."""
    print_header("SEGMENTATION MODE")

    print(f"\nCode:")
    print(f"  import dataflow")
    print(f"  result = dataflow.yolo_to_coco(")
    print(f"      '{images_dir}',")
    print(f"      '{labels_dir}',")
    print(f"      '{classes_file}',")
    print(f"      '{output_json}',")
    print(f"      segmentation=True")
    print(f"  )")

    print(f"\nExecuting...")
    try:
        result = dataflow.yolo_to_coco(
            images_dir, labels_dir, classes_file, output_json, segmentation=True
        )

        print(f"\n✅ Success!")
        print(f"\nSegmentation mode statistics:")
        print(f"  - Segmentation mode: {result.get('segmentation_mode', False)}")
        print(f"  - Images processed: {result.get('images_processed', 0)}")
        print(f"  - Annotations processed: {result.get('annotations_processed', 0)}")

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def demo_advanced_features(images_dir, labels_dir, classes_file, output_json):
    """Demonstrate advanced features and configuration."""
    print_header("ADVANCED FEATURES")

    # Show current configuration
    print(f"\nCurrent configuration:")
    print(f"  COCO_DEFAULT_INFO: {Config.COCO_DEFAULT_INFO}")
    print(f"  VERBOSE: {Config.VERBOSE}")
    print(f"  OVERWRITE_EXISTING: {Config.OVERWRITE_EXISTING}")

    # Demonstrate custom configuration
    print(f"\nCustom configuration example:")
    print(f"  # Save original values")
    print(f"  original_verbose = Config.VERBOSE")
    print(f"  ")
    print(f"  # Configure for batch processing")
    print(f"  Config.VERBOSE = False")
    print(f"  Config.OVERWRITE_EXISTING = True")
    print(f"  ")
    print(f"  # Create converter with custom settings")
    print(f"  converter = YoloToCocoConverter(verbose=False)")
    print(f"  ")
    print(f"  # Restore configuration")
    print(f"  Config.VERBOSE = original_verbose")

    # Actually demonstrate with a different output file
    custom_output = output_json.replace(".json", "_custom.json")
    print(f"\nExecuting with custom configuration...")
    try:
        # Save original
        original_verbose = Config.VERBOSE
        original_overwrite = Config.OVERWRITE_EXISTING

        # Configure
        Config.VERBOSE = False
        Config.OVERWRITE_EXISTING = True

        # Create converter
        converter = YoloToCocoConverter(verbose=False)
        result = converter.convert(
            images_dir, labels_dir, classes_file, custom_output
        )

        print(f"\n✅ Custom conversion successful!")
        print(f"  Output file: {custom_output}")

        # Restore
        Config.VERBOSE = original_verbose
        Config.OVERWRITE_EXISTING = original_overwrite

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        # Restore anyway
        Config.VERBOSE = original_verbose
        Config.OVERWRITE_EXISTING = original_overwrite
        return None


def inspect_output(output_json):
    """Inspect the generated COCO JSON file."""
    print_header("INSPECTING OUTPUT")

    if not os.path.exists(output_json):
        print(f"Output file not found: {output_json}")
        return

    print(f"\nCOCO JSON file: {output_json}")

    try:
        with open(output_json, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        print(f"\n✓ COCO structure:")
        print(f"  - Info: {coco_data.get('info', {}).get('description', 'N/A')}")
        print(f"  - Images: {len(coco_data.get('images', []))}")
        print(f"  - Annotations: {len(coco_data.get('annotations', []))}")
        print(f"  - Categories: {len(coco_data.get('categories', []))}")

        # Show sample image
        if coco_data.get('images'):
            image = coco_data['images'][0]
            print(f"\n  Sample image:")
            print(f"    - ID: {image.get('id')}")
            print(f"    - File name: {image.get('file_name')}")
            print(f"    - Dimensions: {image.get('width')}x{image.get('height')}")

        # Show sample annotation
        if coco_data.get('annotations'):
            ann = coco_data['annotations'][0]
            print(f"\n  Sample annotation:")
            print(f"    - ID: {ann.get('id')}")
            print(f"    - Image ID: {ann.get('image_id')}")
            print(f"    - Category ID: {ann.get('category_id')}")
            print(f"    - Bbox: {ann.get('bbox')}")
            print(f"    - Area: {ann.get('area')}")

        # Show categories
        if coco_data.get('categories'):
            print(f"\n  Categories:")
            for cat in coco_data['categories']:
                print(f"    - {cat.get('id')}: {cat.get('name')} ({cat.get('supercategory', 'N/A')})")

    except Exception as e:
        print(f"❌ Error reading COCO JSON: {e}")


def demo_error_handling():
    """Demonstrate error handling."""
    print_header("ERROR HANDLING")

    # 创建测试路径
    test_paths = create_test_paths()

    try:
        print(f"\n1. Invalid image directory:")
        try:
            result = dataflow.yolo_to_coco(
                test_paths["nonexistent_path"],
                os.path.join(test_paths["temp_dir"], "labels"),
                os.path.join(test_paths["temp_dir"], "classes.names"),
                os.path.join(test_paths["temp_dir"], "output.json")
            )
            print(f"   ❌ Should have raised an error")
        except ValueError as e:
            print(f"   ✅ Caught expected error: {str(e)[:50]}...")

        print(f"\n2. Invalid label directory:")
        try:
            result = dataflow.yolo_to_coco(
                os.path.join(test_paths["temp_dir"], "images"),
                test_paths["nonexistent_path"],
                os.path.join(test_paths["temp_dir"], "classes.names"),
                os.path.join(test_paths["temp_dir"], "output.json")
            )
            print(f"   ❌ Should have raised an error")
        except ValueError as e:
            print(f"   ✅ Caught expected error: {str(e)[:50]}...")

        print(f"\n3. Invalid classes file:")
        try:
            result = dataflow.yolo_to_coco(
                os.path.join(test_paths["temp_dir"], "images"),
                os.path.join(test_paths["temp_dir"], "labels"),
                test_paths["nonexistent_path"],
                os.path.join(test_paths["temp_dir"], "output.json")
            )
            print(f"   ❌ Should have raised an error")
        except ValueError as e:
            print(f"   ✅ Caught expected error: {str(e)[:50]}...")

        print(f"\n4. Invalid output directory:")
        try:
            result = dataflow.yolo_to_coco(
                os.path.join(test_paths["temp_dir"], "images"),
                os.path.join(test_paths["temp_dir"], "labels"),
                os.path.join(test_paths["temp_dir"], "classes.names"),
                test_paths["read_only_path"]
            )
            print(f"   ❌ Should have raised an error")
        except (ValueError, PermissionError) as e:
            print(f"   ✅ Caught expected error: {str(e)[:50]}...")

        print(f"\n5. Malformed YOLO label file:")
        temp_dir = tempfile.mkdtemp(prefix="error_demo_")
        try:
            images_dir = os.path.join(temp_dir, "images")
            labels_dir = os.path.join(temp_dir, "labels")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            # Create empty image file
            with open(os.path.join(images_dir, "test.jpg"), 'wb') as f:
                f.write(b"")

            # Create malformed label file
            bad_label = os.path.join(labels_dir, "test.txt")
            with open(bad_label, 'w', encoding='utf-8') as f:
                f.write("invalid format\n")

            classes_file = os.path.join(temp_dir, "classes.names")
            with open(classes_file, 'w', encoding='utf-8') as f:
                f.write("person\n")

            try:
                result = dataflow.yolo_to_coco(
                    images_dir, labels_dir, classes_file,
                    os.path.join(test_paths["temp_dir"], "output.json")
                )
                print(f"   ❌ Should have raised an error")
            except ValueError as e:
                print(f"   ✅ Caught expected error: {str(e)[:50]}...")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    finally:
        # 清理临时文件
        shutil.rmtree(test_paths["temp_dir"], ignore_errors=True)


def main():
    """Main demonstration function."""
    print_header("YOLO TO COCO CONVERSION - PYTHON API DEMONSTRATION")

    # Create sample data
    temp_dir, images_dir, labels_dir, classes_file = create_sample_yolo_data()
    output_json = os.path.join(temp_dir, "coco_annotations.json")

    try:
        # Demo 1: Convenience function
        result1 = demo_convenience_function(images_dir, labels_dir, classes_file, output_json)

        # Demo 2: Converter class
        output_json2 = os.path.join(temp_dir, "coco_annotations2.json")
        result2 = demo_converter_class(images_dir, labels_dir, classes_file, output_json2)

        # Demo 3: Segmentation mode
        output_json3 = os.path.join(temp_dir, "coco_annotations3.json")
        demo_segmentation_mode(images_dir, labels_dir, classes_file, output_json3)

        # Demo 4: Advanced features
        demo_advanced_features(images_dir, labels_dir, classes_file, os.path.join(temp_dir, "coco_annotations4.json"))

        # Demo 5: Error handling
        demo_error_handling()

        # Inspect output
        if result1:
            inspect_output(output_json)

        print_header("SUMMARY")
        print(f"\n✅ API demonstration completed!")
        print(f"\n📁 Sample data directory: {temp_dir}")
        print(f"   - Images directory: {images_dir}")
        print(f"   - Labels directory: {labels_dir}")
        print(f"   - Classes file: {classes_file}")
        print(f"   - COCO JSON 1 (convenience function): {output_json}")
        print(f"   - COCO JSON 2 (converter class): {output_json2}")

        print(f"\n💡 Key takeaways:")
        print(f"   1. Use dataflow.yolo_to_coco() for simple conversions")
        print(f"   2. Use YoloToCocoConverter class for more control")
        print(f"   3. Use segmentation=True to enforce polygon annotations")
        print(f"   4. Configure behavior via dataflow.Config")
        print(f"   5. All methods return detailed statistics")
        print(f"   6. Error handling is built into the converters")

    finally:
        # Cleanup
        cleanup = input("\nClean up temporary files? (y/n): ").strip().lower()
        if cleanup == 'y':
            shutil.rmtree(temp_dir)
            print("✅ Temporary files cleaned up.")
        else:
            print(f"⚠️  Temporary files preserved at: {temp_dir}")
            print(f"   You may want to clean up manually: {temp_dir}")


if __name__ == "__main__":
    main()