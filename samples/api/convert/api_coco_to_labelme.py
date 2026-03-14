#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/10
@File    : api_coco_to_labelme.py
@Author  : DataFlow Team
@Description: Python API example for COCO to LabelMe conversion

This script demonstrates how to use the DataFlow-CV Python API
for converting COCO JSON format to LabelMe format.
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
from dataflow import CocoToLabelMeConverter
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


def create_sample_coco_data():
    """Create sample COCO JSON data for demonstration."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="coco2labelme_api_demo_")
    print(f"Created temporary directory: {temp_dir}")

    # Create sample COCO JSON
    coco_json = os.path.join(temp_dir, "annotations.json")
    coco_data = {
        "info": {
            "description": "Sample COCO dataset for API demonstration",
            "version": "1.0",
            "year": 2026,
            "contributor": "DataFlow-CV",
            "date_created": "2026-03-10"
        },
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": "image1.jpg",
                "width": 800,
                "height": 600
            },
            {
                "id": 2,
                "file_name": "image2.jpg",
                "width": 1024,
                "height": 768
            },
            {
                "id": 3,
                "file_name": "image3.jpg",
                "width": 640,
                "height": 480
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 150, 200, 120],
                "area": 24000,
                "segmentation": [[100, 150, 300, 150, 300, 270, 100, 270]],  # Polygon
                "iscrowd": 0
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [300, 200, 150, 100],
                "area": 15000,
                "segmentation": [[300, 200, 450, 200, 450, 300, 300, 300]],
                "iscrowd": 0
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 2,
                "bbox": [400, 300, 180, 150],
                "area": 27000,
                "segmentation": [[400, 300, 580, 300, 580, 450, 400, 450]],
                "iscrowd": 0
            },
            {
                "id": 4,
                "image_id": 3,
                "category_id": 1,
                "bbox": [200, 100, 120, 80],
                "area": 9600,
                "segmentation": [[200, 100, 320, 100, 320, 180, 200, 180]],
                "iscrowd": 0
            },
            {
                "id": 5,
                "image_id": 3,
                "category_id": 2,
                "bbox": [400, 300, 100, 120],
                "area": 12000,
                "segmentation": [[400, 300, 500, 300, 500, 420, 400, 420]],
                "iscrowd": 0
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "human"
            },
            {
                "id": 2,
                "name": "car",
                "supercategory": "vehicle"
            }
        ]
    }

    with open(coco_json, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2)

    print(f"Created sample COCO JSON: {coco_json}")
    return temp_dir, coco_json


def demo_convenience_function(coco_json, output_dir):
    """Demonstrate using the convenience function."""
    print_header("USING CONVENIENCE FUNCTION")

    print(f"\nCode:")
    print(f"  import dataflow")
    print(f"  result = dataflow.coco_to_labelme('{coco_json}', '{output_dir}')")

    print(f"\nExecuting...")
    try:
        result = dataflow.coco_to_labelme(coco_json, output_dir)

        print(f"\n✅ Success!")
        print(f"\nResult keys: {list(result.keys())}")

        # Show important statistics
        print(f"\nConversion statistics:")
        print(f"  - Images processed: {result.get('images_processed', 0)}")
        print(f"  - Annotations processed: {result.get('annotations_processed', 0)}")
        print(f"  - Categories found: {result.get('categories_found', 0)}")
        print(f"  - Output directory: {result.get('output_dir', 'N/A')}")
        print(f"  - Classes file: {result.get('classes_file', 'N/A')}")
        print(f"  - Segmentation mode: {result.get('segmentation_mode', False)}")

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def demo_converter_class(coco_json, output_dir):
    """Demonstrate using the converter class directly."""
    print_header("USING CONVERTER CLASS")

    print(f"\nCode:")
    print(f"  from dataflow import CocoToLabelMeConverter")
    print(f"  converter = CocoToLabelMeConverter(verbose=True)")
    print(f"  result = converter.convert('{coco_json}', '{output_dir}')")

    print(f"\nExecuting...")
    try:
        converter = CocoToLabelMeConverter(verbose=True)
        print(f"  Converter created: {converter.__class__.__name__}")
        print(f"  Verbose mode: {converter.verbose}")

        result = converter.convert(coco_json, output_dir)

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


def demo_segmentation_mode(coco_json, output_dir):
    """Demonstrate segmentation mode."""
    print_header("SEGMENTATION MODE")

    print(f"\nCode:")
    print(f"  import dataflow")
    print(f"  result = dataflow.coco_to_labelme('{coco_json}', '{output_dir}', segmentation=True)")

    print(f"\nExecuting...")
    try:
        result = dataflow.coco_to_labelme(coco_json, output_dir, segmentation=True)

        print(f"\n✅ Success!")
        print(f"\nSegmentation mode statistics:")
        print(f"  - Segmentation mode: {result.get('segmentation_mode', False)}")
        print(f"  - Images processed: {result.get('images_processed', 0)}")
        print(f"  - Annotations processed: {result.get('annotations_processed', 0)}")

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def demo_advanced_features(coco_json, output_dir):
    """Demonstrate advanced features and configuration."""
    print_header("ADVANCED FEATURES")

    # Show current configuration
    print(f"\nCurrent configuration:")
    print(f"  YOLO_CLASSES_FILENAME: {Config.YOLO_CLASSES_FILENAME}")
    print(f"  YOLO_LABELS_DIRNAME: {Config.YOLO_LABELS_DIRNAME}")
    print(f"  VERBOSE: {Config.VERBOSE}")
    print(f"  OVERWRITE_EXISTING: {Config.OVERWRITE_EXISTING}")
    print(f"  YOLO_SEGMENTATION: {Config.YOLO_SEGMENTATION}")

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
    print(f"  converter = CocoToLabelMeConverter(verbose=False)")
    print(f"  ")
    print(f"  # Restore configuration")
    print(f"  Config.VERBOSE = original_verbose")

    # Actually demonstrate with a different output directory
    custom_output = output_dir + "_custom"
    print(f"\nExecuting with custom configuration...")
    try:
        # Save original
        original_verbose = Config.VERBOSE
        original_overwrite = Config.OVERWRITE_EXISTING

        # Configure
        Config.VERBOSE = False
        Config.OVERWRITE_EXISTING = True

        # Create converter
        converter = CocoToLabelMeConverter(verbose=False)
        result = converter.convert(coco_json, custom_output)

        print(f"\n✅ Custom conversion successful!")
        print(f"  Output directory: {custom_output}")

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


def inspect_output(output_dir):
    """Inspect the generated output files."""
    print_header("INSPECTING OUTPUT")

    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return

    print(f"\nOutput directory: {output_dir}")

    # Check for expected files
    classes_file = os.path.join(output_dir, Config.YOLO_CLASSES_FILENAME)

    if os.path.exists(classes_file):
        print(f"\n✓ Classes file: {classes_file}")
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f if line.strip()]
        print(f"  Classes: {classes}")
    else:
        print(f"\n✗ Classes file not found!")

    # Check for LabelMe JSON files
    labelme_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    print(f"\n✓ LabelMe JSON files: {len(labelme_files)} files")

    # Show sample LabelMe JSON file
    if labelme_files:
        sample_file = os.path.join(output_dir, labelme_files[0])
        print(f"\n  Sample LabelMe file: {labelme_files[0]}")
        with open(sample_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"    - Version: {data.get('version', 'N/A')}")
            print(f"    - Image path: {data.get('imagePath', 'N/A')}")
            print(f"    - Image dimensions: {data.get('imageWidth', 'N/A')}x{data.get('imageHeight', 'N/A')}")
            print(f"    - Shapes count: {len(data.get('shapes', []))}")
            if data.get('shapes'):
                shape = data['shapes'][0]
                print(f"    - First shape: {shape.get('label')} ({shape.get('shape_type')})")


def demo_error_handling():
    """Demonstrate error handling."""
    print_header("ERROR HANDLING")

    # 创建测试路径
    test_paths = create_test_paths()

    try:
        print(f"\n1. Invalid COCO JSON path:")
        try:
            result = dataflow.coco_to_labelme(test_paths["nonexistent_path"],
                                             os.path.join(test_paths["temp_dir"], "output"))
            print(f"   ❌ Should have raised an error")
        except ValueError as e:
            print(f"   ✅ Caught expected error: {str(e)[:50]}...")

        print(f"\n2. Invalid output directory:")
        try:
            result = dataflow.coco_to_labelme(test_paths["temp_file_path"],
                                             test_paths["read_only_path"])
            print(f"   ❌ Should have raised an error")
        except (ValueError, PermissionError) as e:
            print(f"   ✅ Caught expected error: {str(e)[:50]}...")

        print(f"\n3. Malformed COCO JSON:")
        temp_dir = tempfile.mkdtemp(prefix="error_demo_")
        try:
            bad_json = os.path.join(temp_dir, "bad.json")
            with open(bad_json, 'w', encoding='utf-8') as f:
                f.write("{ invalid json }")

            try:
                result = dataflow.coco_to_labelme(bad_json, os.path.join(test_paths["temp_dir"], "output"))
                print(f"   ❌ Should have raised an error")
            except (ValueError, json.JSONDecodeError) as e:
                print(f"   ✅ Caught expected error: {str(e)[:50]}...")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    finally:
        # 清理临时文件
        shutil.rmtree(test_paths["temp_dir"], ignore_errors=True)


def main():
    """Main demonstration function."""
    print_header("COCO TO LABELME CONVERSION - PYTHON API DEMONSTRATION")

    # Create sample data
    temp_dir, coco_json = create_sample_coco_data()

    try:
        # Demo 1: Convenience function
        output_dir1 = os.path.join(temp_dir, "output1")
        result1 = demo_convenience_function(coco_json, output_dir1)

        # Demo 2: Converter class
        output_dir2 = os.path.join(temp_dir, "output2")
        result2 = demo_converter_class(coco_json, output_dir2)

        # Demo 3: Segmentation mode
        output_dir3 = os.path.join(temp_dir, "output3")
        demo_segmentation_mode(coco_json, output_dir3)

        # Demo 4: Advanced features
        demo_advanced_features(coco_json, os.path.join(temp_dir, "output4"))

        # Demo 5: Error handling
        demo_error_handling()

        # Inspect output
        if result1:
            inspect_output(output_dir1)

        print_header("SUMMARY")
        print(f"\n✅ API demonstration completed!")
        print(f"\n📁 Sample data directory: {temp_dir}")
        print(f"   - COCO JSON: {coco_json}")
        print(f"   - Output 1 (convenience function): {output_dir1}")
        print(f"   - Output 2 (converter class): {output_dir2}")
        print(f"   - Output 3 (segmentation mode): {output_dir3}")

        print(f"\n💡 Key takeaways:")
        print(f"   1. Use dataflow.coco_to_labelme() for simple conversions")
        print(f"   2. Use CocoToLabelMeConverter class for more control")
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