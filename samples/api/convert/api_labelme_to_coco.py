#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/10
@File    : api_labelme_to_coco.py
@Author  : DataFlow Team
@Description: Python API example for LabelMe to COCO conversion

This script demonstrates how to use the DataFlow-CV Python API
for converting LabelMe format to COCO JSON format.
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
from dataflow import LabelMeToCocoConverter
from dataflow.config import Config


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def create_sample_labelme_data():
    """Create sample LabelMe JSON data for demonstration."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="labelme2coco_api_demo_")
    print(f"Created temporary directory: {temp_dir}")

    # Create classes file
    classes_file = os.path.join(temp_dir, "class.names")
    classes = ["person", "car", "bicycle"]
    with open(classes_file, 'w', encoding='utf-8') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"Created classes file: {classes_file}")

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

    return temp_dir, label_dir, classes_file


def demo_convenience_function(label_dir, classes_file, output_json):
    """Demonstrate using the convenience function."""
    print_header("USING CONVENIENCE FUNCTION")

    print(f"\nCode:")
    print(f"  import dataflow")
    print(f"  result = dataflow.labelme_to_coco('{label_dir}', '{classes_file}', '{output_json}')")

    print(f"\nExecuting...")
    try:
        result = dataflow.labelme_to_coco(label_dir, classes_file, output_json)

        print(f"\n✅ Success!")
        print(f"\nResult keys: {list(result.keys())}")

        # Show important statistics
        print(f"\nConversion statistics:")
        print(f"  - Images processed: {result.get('images_processed', 0)}")
        print(f"  - Annotations processed: {result.get('annotations_processed', 0)}")
        print(f"  - Categories found: {result.get('categories_found', 0)}")
        print(f"  - Label directory: {result.get('label_dir', 'N/A')}")
        print(f"  - Classes file: {result.get('classes_file', 'N/A')}")
        print(f"  - COCO JSON path: {result.get('coco_json_path', 'N/A')}")
        print(f"  - Segmentation mode: {result.get('segmentation_mode', False)}")

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def demo_converter_class(label_dir, classes_file, output_json):
    """Demonstrate using the converter class directly."""
    print_header("USING CONVERTER CLASS")

    print(f"\nCode:")
    print(f"  from dataflow import LabelMeToCocoConverter")
    print(f"  converter = LabelMeToCocoConverter(verbose=True)")
    print(f"  result = converter.convert('{label_dir}', '{classes_file}', '{output_json}')")

    print(f"\nExecuting...")
    try:
        converter = LabelMeToCocoConverter(verbose=True)
        print(f"  Converter created: {converter.__class__.__name__}")
        print(f"  Verbose mode: {converter.verbose}")

        result = converter.convert(label_dir, classes_file, output_json)

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


def demo_segmentation_mode(label_dir, classes_file, output_json):
    """Demonstrate segmentation mode."""
    print_header("SEGMENTATION MODE")

    print(f"\nCode:")
    print(f"  import dataflow")
    print(f"  result = dataflow.labelme_to_coco('{label_dir}', '{classes_file}', '{output_json}', segmentation=True)")

    print(f"\nExecuting...")
    try:
        result = dataflow.labelme_to_coco(label_dir, classes_file, output_json, segmentation=True)

        print(f"\n✅ Success!")
        print(f"\nSegmentation mode statistics:")
        print(f"  - Segmentation mode: {result.get('segmentation_mode', False)}")
        print(f"  - Images processed: {result.get('images_processed', 0)}")
        print(f"  - Annotations processed: {result.get('annotations_processed', 0)}")

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def demo_advanced_features(label_dir, classes_file, output_json):
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
    print(f"  converter = LabelMeToCocoConverter(verbose=False)")
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
        converter = LabelMeToCocoConverter(verbose=False)
        result = converter.convert(label_dir, classes_file, custom_output)

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

    print(f"\n1. Invalid label directory:")
    try:
        result = dataflow.labelme_to_coco("/invalid/label/dir", "/tmp/classes.names", "/tmp/output.json")
        print(f"   ❌ Should have raised an error")
    except ValueError as e:
        print(f"   ✅ Caught expected error: {str(e)[:50]}...")

    print(f"\n2. Invalid classes file:")
    try:
        result = dataflow.labelme_to_coco("/tmp/labels", "/invalid/classes.names", "/tmp/output.json")
        print(f"   ❌ Should have raised an error")
    except ValueError as e:
        print(f"   ✅ Caught expected error: {str(e)[:50]}...")

    print(f"\n3. Invalid output directory:")
    try:
        result = dataflow.labelme_to_coco("/tmp/labels", "/tmp/classes.names", "/root/no_permission/output.json")
        print(f"   ❌ Should have raised an error")
    except (ValueError, PermissionError) as e:
        print(f"   ✅ Caught expected error: {str(e)[:50]}...")

    print(f"\n4. Malformed LabelMe JSON:")
    temp_dir = tempfile.mkdtemp(prefix="error_demo_")
    try:
        label_dir = os.path.join(temp_dir, "labels")
        os.makedirs(label_dir, exist_ok=True)

        bad_json = os.path.join(label_dir, "bad.json")
        with open(bad_json, 'w', encoding='utf-8') as f:
            f.write("{ invalid json }")

        classes_file = os.path.join(temp_dir, "classes.names")
        with open(classes_file, 'w', encoding='utf-8') as f:
            f.write("person\ncar\n")

        try:
            result = dataflow.labelme_to_coco(label_dir, classes_file, "/tmp/output.json")
            print(f"   ❌ Should have raised an error")
        except (ValueError, json.JSONDecodeError) as e:
            print(f"   ✅ Caught expected error: {str(e)[:50]}...")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main demonstration function."""
    print_header("LABELME TO COCO CONVERSION - PYTHON API DEMONSTRATION")

    # Create sample data
    temp_dir, label_dir, classes_file = create_sample_labelme_data()
    output_json = os.path.join(temp_dir, "coco_annotations.json")

    try:
        # Demo 1: Convenience function
        result1 = demo_convenience_function(label_dir, classes_file, output_json)

        # Demo 2: Converter class
        output_json2 = os.path.join(temp_dir, "coco_annotations2.json")
        result2 = demo_converter_class(label_dir, classes_file, output_json2)

        # Demo 3: Segmentation mode
        output_json3 = os.path.join(temp_dir, "coco_annotations3.json")
        demo_segmentation_mode(label_dir, classes_file, output_json3)

        # Demo 4: Advanced features
        demo_advanced_features(label_dir, classes_file, os.path.join(temp_dir, "coco_annotations4.json"))

        # Demo 5: Error handling
        demo_error_handling()

        # Inspect output
        if result1:
            inspect_output(output_json)

        print_header("SUMMARY")
        print(f"\n✅ API demonstration completed!")
        print(f"\n📁 Sample data directory: {temp_dir}")
        print(f"   - Label directory: {label_dir}")
        print(f"   - Classes file: {classes_file}")
        print(f"   - COCO JSON 1 (convenience function): {output_json}")
        print(f"   - COCO JSON 2 (converter class): {output_json2}")

        print(f"\n💡 Key takeaways:")
        print(f"   1. Use dataflow.labelme_to_coco() for simple conversions")
        print(f"   2. Use LabelMeToCocoConverter class for more control")
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