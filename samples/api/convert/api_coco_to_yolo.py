#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 21:25
@File    : api_coco_to_yolo.py
@Author  : zj
@Description: Python API example for COCO to YOLO conversion

This script demonstrates how to use the DataFlow-CV Python API
for converting COCO JSON format to YOLO format.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path to import dataflow
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import DataFlow-CV
import dataflow
from dataflow import CocoToYoloConverter
from dataflow.config import Config


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def create_sample_coco_data():
    """Create sample COCO JSON data for demonstration."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="coco2yolo_api_demo_")
    print(f"Created temporary directory: {temp_dir}")

    # Create sample COCO JSON
    coco_json = os.path.join(temp_dir, "annotations.json")
    coco_data = {
        "info": {
            "description": "Sample COCO dataset for API demonstration",
            "version": "1.0",
            "year": 2026,
            "contributor": "DataFlow-CV",
            "date_created": "2026-03-08"
        },
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": "person_image.jpg",
                "width": 800,
                "height": 600
            },
            {
                "id": 2,
                "file_name": "car_image.jpg",
                "width": 1024,
                "height": 768
            },
            {
                "id": 3,
                "file_name": "mixed_image.jpg",
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
                "segmentation": [],
                "iscrowd": 0
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [300, 200, 150, 100],
                "area": 15000,
                "segmentation": [],
                "iscrowd": 0
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 2,
                "bbox": [400, 300, 180, 150],
                "area": 27000,
                "segmentation": [],
                "iscrowd": 0
            },
            {
                "id": 4,
                "image_id": 3,
                "category_id": 1,
                "bbox": [200, 100, 120, 80],
                "area": 9600,
                "segmentation": [],
                "iscrowd": 0
            },
            {
                "id": 5,
                "image_id": 3,
                "category_id": 2,
                "bbox": [400, 300, 100, 120],
                "area": 12000,
                "segmentation": [],
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
    print(f"  result = dataflow.coco_to_yolo('{coco_json}', '{output_dir}')")

    print(f"\nExecuting...")
    try:
        result = dataflow.coco_to_yolo(coco_json, output_dir)

        print(f"\n✅ Success!")
        print(f"\nResult keys: {list(result.keys())}")

        # Show important statistics
        print(f"\nConversion statistics:")
        print(f"  - Images processed: {result.get('images_processed', 0)}")
        print(f"  - Images with annotations: {result.get('images_with_annotations', 0)}")
        print(f"  - Annotations processed: {result.get('annotations_processed', 0)}")
        print(f"  - Total categories: {result.get('total_categories', 0)}")
        print(f"  - Total images: {result.get('total_images', 0)}")

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def demo_converter_class(coco_json, output_dir):
    """Demonstrate using the converter class directly."""
    print_header("USING CONVERTER CLASS")

    print(f"\nCode:")
    print(f"  from dataflow import CocoToYoloConverter")
    print(f"  converter = CocoToYoloConverter(verbose=True)")
    print(f"  result = converter.convert('{coco_json}', '{output_dir}')")

    print(f"\nExecuting...")
    try:
        converter = CocoToYoloConverter(verbose=True)
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
    print(f"  converter = CocoToYoloConverter(verbose=False)")
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
        converter = CocoToYoloConverter(verbose=False)
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
    labels_dir = os.path.join(output_dir, Config.YOLO_LABELS_DIRNAME)

    if os.path.exists(classes_file):
        print(f"\n✓ Classes file: {classes_file}")
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f if line.strip()]
        print(f"  Classes: {classes}")
    else:
        print(f"\n✗ Classes file not found!")

    if os.path.exists(labels_dir):
        print(f"\n✓ Labels directory: {labels_dir}")
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        print(f"  Number of label files: {len(label_files)}")

        # Show sample label file
        if label_files:
            sample_file = os.path.join(labels_dir, label_files[0])
            print(f"\n  Sample label file: {label_files[0]}")
            with open(sample_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                for i, line in enumerate(lines[:2], 1):
                    print(f"    Line {i}: {line}")
                if len(lines) > 2:
                    print(f"    ... and {len(lines) - 2} more lines")
    else:
        print(f"\n✗ Labels directory not found!")


def demo_error_handling():
    """Demonstrate error handling."""
    print_header("ERROR HANDLING")

    print(f"\n1. Invalid COCO JSON path:")
    try:
        result = dataflow.coco_to_yolo("/invalid/path/annotations.json", "/tmp/output")
        print(f"   ❌ Should have raised an error")
    except ValueError as e:
        print(f"   ✅ Caught expected error: {str(e)[:50]}...")

    print(f"\n2. Invalid output directory:")
    try:
        result = dataflow.coco_to_yolo("/tmp/annotations.json", "/root/no_permission")
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
            result = dataflow.coco_to_yolo(bad_json, "/tmp/output")
            print(f"   ❌ Should have raised an error")
        except (ValueError, json.JSONDecodeError) as e:
            print(f"   ✅ Caught expected error: {str(e)[:50]}...")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main demonstration function."""
    print_header("COCO TO YOLO CONVERSION - PYTHON API DEMONSTRATION")

    # Create sample data
    temp_dir, coco_json = create_sample_coco_data()

    try:
        # Demo 1: Convenience function
        output_dir1 = os.path.join(temp_dir, "output1")
        result1 = demo_convenience_function(coco_json, output_dir1)

        # Demo 2: Converter class
        output_dir2 = os.path.join(temp_dir, "output2")
        result2 = demo_converter_class(coco_json, output_dir2)

        # Demo 3: Advanced features
        demo_advanced_features(coco_json, os.path.join(temp_dir, "output3"))

        # Demo 4: Error handling
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

        print(f"\n💡 Key takeaways:")
        print(f"   1. Use dataflow.coco_to_yolo() for simple conversions")
        print(f"   2. Use CocoToYoloConverter class for more control")
        print(f"   3. Configure behavior via dataflow.Config")
        print(f"   4. All methods return detailed statistics")
        print(f"   5. Error handling is built into the converters")

    finally:
        # Cleanup
        cleanup = input("\nClean up temporary files? (y/n): ").strip().lower()
        if cleanup == 'y':
            shutil.rmtree(temp_dir)
            print("✅ Temporary files cleaned up.")
        else:
            print(f"⚠️  Temporary files preserved at: {temp_dir}")
            print(f"   You may want to clean up manually: rm -rf {temp_dir}")


if __name__ == "__main__":
    main()