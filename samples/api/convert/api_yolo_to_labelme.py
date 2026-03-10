#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/10
@File    : api_yolo_to_labelme.py
@Author  : DataFlow Team
@Description: Python API example for YOLO to LabelMe conversion

This script demonstrates how to use the DataFlow-CV Python API
for converting YOLO format to LabelMe format.
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
from dataflow import YoloToLabelMeConverter
from dataflow.config import Config


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def create_sample_yolo_data():
    """Create sample YOLO data for demonstration."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="yolo2labelme_api_demo_")
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


def demo_convenience_function(images_dir, labels_dir, classes_file, output_dir):
    """Demonstrate using the convenience function."""
    print_header("USING CONVENIENCE FUNCTION")

    print(f"\nCode:")
    print(f"  import dataflow")
    print(f"  result = dataflow.yolo_to_labelme(")
    print(f"      '{images_dir}',")
    print(f"      '{labels_dir}',")
    print(f"      '{classes_file}',")
    print(f"      '{output_dir}'")
    print(f"  )")

    print(f"\nExecuting...")
    try:
        result = dataflow.yolo_to_labelme(
            images_dir, labels_dir, classes_file, output_dir
        )

        print(f"\n✅ Success!")
        print(f"\nResult keys: {list(result.keys())}")

        # Show important statistics
        print(f"\nConversion statistics:")
        print(f"  - Images processed: {result.get('images_processed', 0)}")
        print(f"  - Annotations processed: {result.get('annotations_processed', 0)}")
        print(f"  - Image directory: {result.get('image_dir', 'N/A')}")
        print(f"  - Label directory: {result.get('label_dir', 'N/A')}")
        print(f"  - Classes file: {result.get('classes_file', 'N/A')}")
        print(f"  - Output directory: {result.get('output_dir', 'N/A')}")
        print(f"  - Segmentation mode: {result.get('segmentation_mode', False)}")

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def demo_converter_class(images_dir, labels_dir, classes_file, output_dir):
    """Demonstrate using the converter class directly."""
    print_header("USING CONVERTER CLASS")

    print(f"\nCode:")
    print(f"  from dataflow import YoloToLabelMeConverter")
    print(f"  converter = YoloToLabelMeConverter(verbose=True)")
    print(f"  result = converter.convert(")
    print(f"      '{images_dir}',")
    print(f"      '{labels_dir}',")
    print(f"      '{classes_file}',")
    print(f"      '{output_dir}'")
    print(f"  )")

    print(f"\nExecuting...")
    try:
        converter = YoloToLabelMeConverter(verbose=True)
        print(f"  Converter created: {converter.__class__.__name__}")
        print(f"  Verbose mode: {converter.verbose}")

        result = converter.convert(
            images_dir, labels_dir, classes_file, output_dir
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


def demo_segmentation_mode(images_dir, labels_dir, classes_file, output_dir):
    """Demonstrate segmentation mode."""
    print_header("SEGMENTATION MODE")

    print(f"\nCode:")
    print(f"  import dataflow")
    print(f"  result = dataflow.yolo_to_labelme(")
    print(f"      '{images_dir}',")
    print(f"      '{labels_dir}',")
    print(f"      '{classes_file}',")
    print(f"      '{output_dir}',")
    print(f"      segmentation=True")
    print(f"  )")

    print(f"\nExecuting...")
    try:
        result = dataflow.yolo_to_labelme(
            images_dir, labels_dir, classes_file, output_dir, segmentation=True
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


def demo_advanced_features(images_dir, labels_dir, classes_file, output_dir):
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
    print(f"  converter = YoloToLabelMeConverter(verbose=False)")
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
        converter = YoloToLabelMeConverter(verbose=False)
        result = converter.convert(
            images_dir, labels_dir, classes_file, custom_output
        )

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
                print(f"      Points: {shape.get('points', [])[:4]}...")

    # List all LabelMe files
    print(f"\n  All LabelMe files:")
    for i, fname in enumerate(sorted(labelme_files)[:5], 1):
        print(f"    {i}. {fname}")
    if len(labelme_files) > 5:
        print(f"    ... and {len(labelme_files) - 5} more")


def demo_error_handling():
    """Demonstrate error handling."""
    print_header("ERROR HANDLING")

    print(f"\n1. Invalid image directory:")
    try:
        result = dataflow.yolo_to_labelme(
            "/invalid/image/dir", "/tmp/labels", "/tmp/classes.names", "/tmp/output"
        )
        print(f"   ❌ Should have raised an error")
    except ValueError as e:
        print(f"   ✅ Caught expected error: {str(e)[:50]}...")

    print(f"\n2. Invalid label directory:")
    try:
        result = dataflow.yolo_to_labelme(
            "/tmp/images", "/invalid/label/dir", "/tmp/classes.names", "/tmp/output"
        )
        print(f"   ❌ Should have raised an error")
    except ValueError as e:
        print(f"   ✅ Caught expected error: {str(e)[:50]}...")

    print(f"\n3. Invalid classes file:")
    try:
        result = dataflow.yolo_to_labelme(
            "/tmp/images", "/tmp/labels", "/invalid/classes.names", "/tmp/output"
        )
        print(f"   ❌ Should have raised an error")
    except ValueError as e:
        print(f"   ✅ Caught expected error: {str(e)[:50]}...")

    print(f"\n4. Invalid output directory:")
    try:
        result = dataflow.yolo_to_labelme(
            "/tmp/images", "/tmp/labels", "/tmp/classes.names", "/root/no_permission"
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
            result = dataflow.yolo_to_labelme(
                images_dir, labels_dir, classes_file, "/tmp/output"
            )
            print(f"   ❌ Should have raised an error")
        except ValueError as e:
            print(f"   ✅ Caught expected error: {str(e)[:50]}...")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main demonstration function."""
    print_header("YOLO TO LABELME CONVERSION - PYTHON API DEMONSTRATION")

    # Create sample data
    temp_dir, images_dir, labels_dir, classes_file = create_sample_yolo_data()

    try:
        # Demo 1: Convenience function
        output_dir1 = os.path.join(temp_dir, "output1")
        result1 = demo_convenience_function(images_dir, labels_dir, classes_file, output_dir1)

        # Demo 2: Converter class
        output_dir2 = os.path.join(temp_dir, "output2")
        result2 = demo_converter_class(images_dir, labels_dir, classes_file, output_dir2)

        # Demo 3: Segmentation mode
        output_dir3 = os.path.join(temp_dir, "output3")
        demo_segmentation_mode(images_dir, labels_dir, classes_file, output_dir3)

        # Demo 4: Advanced features
        demo_advanced_features(images_dir, labels_dir, classes_file, os.path.join(temp_dir, "output4"))

        # Demo 5: Error handling
        demo_error_handling()

        # Inspect output
        if result1:
            inspect_output(output_dir1)

        print_header("SUMMARY")
        print(f"\n✅ API demonstration completed!")
        print(f"\n📁 Sample data directory: {temp_dir}")
        print(f"   - Images directory: {images_dir}")
        print(f"   - Labels directory: {labels_dir}")
        print(f"   - Classes file: {classes_file}")
        print(f"   - Output 1 (convenience function): {output_dir1}")
        print(f"   - Output 2 (converter class): {output_dir2}")
        print(f"   - Output 3 (segmentation mode): {output_dir3}")

        print(f"\n💡 Key takeaways:")
        print(f"   1. Use dataflow.yolo_to_labelme() for simple conversions")
        print(f"   2. Use YoloToLabelMeConverter class for more control")
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