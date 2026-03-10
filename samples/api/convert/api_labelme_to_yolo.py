#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/10
@File    : api_labelme_to_yolo.py
@Author  : DataFlow Team
@Description: Python API example for LabelMe to YOLO conversion

This script demonstrates how to use the DataFlow-CV Python API
for converting LabelMe format to YOLO format.
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
from dataflow import LabelMeToYoloConverter
from dataflow.config import Config


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def create_sample_labelme_data():
    """Create sample LabelMe JSON data for demonstration."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="labelme2yolo_api_demo_")
    print(f"Created temporary directory: {temp_dir}")

    # Create sample LabelMe JSON files
    label_dir = os.path.join(temp_dir, "labelme_labels")
    os.makedirs(label_dir, exist_ok=True)

    # Create first LabelMe JSON file
    labelme1 = os.path.join(label_dir, "image1.json")
    data1 = {
        "version": "5.3.1",
        "flags": {},
        "shapes": [
            {
                "label": "person",
                "points": [[100.0, 150.0], [300.0, 150.0], [300.0, 270.0], [100.0, 270.0]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            },
            {
                "label": "person",
                "points": [[300.0, 200.0], [450.0, 200.0], [450.0, 300.0], [300.0, 300.0]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
        ],
        "imagePath": "image1.jpg",
        "imageData": None,
        "imageHeight": 600,
        "imageWidth": 800
    }
    with open(labelme1, 'w', encoding='utf-8') as f:
        json.dump(data1, f, indent=2)
    print(f"Created LabelMe JSON: {labelme1}")

    # Create second LabelMe JSON file
    labelme2 = os.path.join(label_dir, "image2.json")
    data2 = {
        "version": "5.3.1",
        "flags": {},
        "shapes": [
            {
                "label": "car",
                "points": [[400.0, 300.0], [580.0, 300.0], [580.0, 450.0], [400.0, 450.0]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            },
            {
                "label": "bicycle",
                "points": [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
        ],
        "imagePath": "image2.jpg",
        "imageData": None,
        "imageHeight": 768,
        "imageWidth": 1024
    }
    with open(labelme2, 'w', encoding='utf-8') as f:
        json.dump(data2, f, indent=2)
    print(f"Created LabelMe JSON: {labelme2}")

    # Create third LabelMe JSON file (empty annotations)
    labelme3 = os.path.join(label_dir, "image3.json")
    data3 = {
        "version": "5.3.1",
        "flags": {},
        "shapes": [],
        "imagePath": "image3.jpg",
        "imageData": None,
        "imageHeight": 480,
        "imageWidth": 640
    }
    with open(labelme3, 'w', encoding='utf-8') as f:
        json.dump(data3, f, indent=2)
    print(f"Created LabelMe JSON: {labelme3}")

    return temp_dir, label_dir


def demo_convenience_function(label_dir, output_dir):
    """Demonstrate using the convenience function."""
    print_header("USING CONVENIENCE FUNCTION")

    print(f"\nCode:")
    print(f"  import dataflow")
    print(f"  result = dataflow.labelme_to_yolo('{label_dir}', '{output_dir}')")

    print(f"\nExecuting...")
    try:
        result = dataflow.labelme_to_yolo(label_dir, output_dir)

        print(f"\n✅ Success!")
        print(f"\nResult keys: {list(result.keys())}")

        # Show important statistics
        print(f"\nConversion statistics:")
        print(f"  - Images processed: {result.get('images_processed', 0)}")
        print(f"  - Annotations processed: {result.get('annotations_processed', 0)}")
        print(f"  - Categories found: {result.get('categories_found', 0)}")
        print(f"  - Label directory: {result.get('label_dir', 'N/A')}")
        print(f"  - Output directory: {result.get('output_dir', 'N/A')}")
        print(f"  - Classes file: {result.get('classes_file', 'N/A')}")
        print(f"  - Labels directory: {result.get('labels_dir', 'N/A')}")
        print(f"  - Segmentation mode: {result.get('segmentation_mode', False)}")

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def demo_converter_class(label_dir, output_dir):
    """Demonstrate using the converter class directly."""
    print_header("USING CONVERTER CLASS")

    print(f"\nCode:")
    print(f"  from dataflow import LabelMeToYoloConverter")
    print(f"  converter = LabelMeToYoloConverter(verbose=True)")
    print(f"  result = converter.convert('{label_dir}', '{output_dir}')")

    print(f"\nExecuting...")
    try:
        converter = LabelMeToYoloConverter(verbose=True)
        print(f"  Converter created: {converter.__class__.__name__}")
        print(f"  Verbose mode: {converter.verbose}")

        result = converter.convert(label_dir, output_dir)

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


def demo_segmentation_mode(label_dir, output_dir):
    """Demonstrate segmentation mode."""
    print_header("SEGMENTATION MODE")

    print(f"\nCode:")
    print(f"  import dataflow")
    print(f"  result = dataflow.labelme_to_yolo('{label_dir}', '{output_dir}', segmentation=True)")

    print(f"\nExecuting...")
    try:
        result = dataflow.labelme_to_yolo(label_dir, output_dir, segmentation=True)

        print(f"\n✅ Success!")
        print(f"\nSegmentation mode statistics:")
        print(f"  - Segmentation mode: {result.get('segmentation_mode', False)}")
        print(f"  - Images processed: {result.get('images_processed', 0)}")
        print(f"  - Annotations processed: {result.get('annotations_processed', 0)}")

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def demo_advanced_features(label_dir, output_dir):
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
    print(f"  converter = LabelMeToYoloConverter(verbose=False)")
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
        converter = LabelMeToYoloConverter(verbose=False)
        result = converter.convert(label_dir, custom_output)

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

    # Check for class.names file
    classes_file = os.path.join(output_dir, Config.YOLO_CLASSES_FILENAME)
    if os.path.exists(classes_file):
        print(f"\n✓ Classes file: {classes_file}")
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f if line.strip()]
        print(f"  Classes: {classes}")
    else:
        print(f"\n✗ Classes file not found!")

    # Check for labels directory
    labels_dir = os.path.join(output_dir, Config.YOLO_LABELS_DIRNAME)
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

    # Show directory structure
    print(f"\nDirectory structure:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')


def demo_error_handling():
    """Demonstrate error handling."""
    print_header("ERROR HANDLING")

    print(f"\n1. Invalid label directory:")
    try:
        result = dataflow.labelme_to_yolo("/invalid/label/dir", "/tmp/output")
        print(f"   ❌ Should have raised an error")
    except ValueError as e:
        print(f"   ✅ Caught expected error: {str(e)[:50]}...")

    print(f"\n2. Invalid output directory:")
    try:
        result = dataflow.labelme_to_yolo("/tmp/labels", "/root/no_permission")
        print(f"   ❌ Should have raised an error")
    except (ValueError, PermissionError) as e:
        print(f"   ✅ Caught expected error: {str(e)[:50]}...")

    print(f"\n3. Malformed LabelMe JSON:")
    temp_dir = tempfile.mkdtemp(prefix="error_demo_")
    try:
        label_dir = os.path.join(temp_dir, "labels")
        os.makedirs(label_dir, exist_ok=True)

        bad_json = os.path.join(label_dir, "bad.json")
        with open(bad_json, 'w', encoding='utf-8') as f:
            f.write("{ invalid json }")

        try:
            result = dataflow.labelme_to_yolo(label_dir, "/tmp/output")
            print(f"   ❌ Should have raised an error")
        except (ValueError, json.JSONDecodeError) as e:
            print(f"   ✅ Caught expected error: {str(e)[:50]}...")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n4. Empty label directory:")
    temp_dir = tempfile.mkdtemp(prefix="error_demo_")
    try:
        label_dir = os.path.join(temp_dir, "labels")
        os.makedirs(label_dir, exist_ok=True)

        try:
            result = dataflow.labelme_to_yolo(label_dir, "/tmp/output")
            print(f"   ⚠️  Empty directory handled gracefully")
        except ValueError as e:
            print(f"   ⚠️  Error with empty directory: {str(e)[:50]}...")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main demonstration function."""
    print_header("LABELME TO YOLO CONVERSION - PYTHON API DEMONSTRATION")

    # Create sample data
    temp_dir, label_dir = create_sample_labelme_data()

    try:
        # Demo 1: Convenience function
        output_dir1 = os.path.join(temp_dir, "output1")
        result1 = demo_convenience_function(label_dir, output_dir1)

        # Demo 2: Converter class
        output_dir2 = os.path.join(temp_dir, "output2")
        result2 = demo_converter_class(label_dir, output_dir2)

        # Demo 3: Segmentation mode
        output_dir3 = os.path.join(temp_dir, "output3")
        demo_segmentation_mode(label_dir, output_dir3)

        # Demo 4: Advanced features
        demo_advanced_features(label_dir, os.path.join(temp_dir, "output4"))

        # Demo 5: Error handling
        demo_error_handling()

        # Inspect output
        if result1:
            inspect_output(output_dir1)

        print_header("SUMMARY")
        print(f"\n✅ API demonstration completed!")
        print(f"\n📁 Sample data directory: {temp_dir}")
        print(f"   - Label directory: {label_dir}")
        print(f"   - Output 1 (convenience function): {output_dir1}")
        print(f"   - Output 2 (converter class): {output_dir2}")
        print(f"   - Output 3 (segmentation mode): {output_dir3}")

        print(f"\n💡 Key takeaways:")
        print(f"   1. Use dataflow.labelme_to_yolo() for simple conversions")
        print(f"   2. Use LabelMeToYoloConverter class for more control")
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
            print(f"   You may want to clean up manually: rm -rf {temp_dir}")


if __name__ == "__main__":
    main()