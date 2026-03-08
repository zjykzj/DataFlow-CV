#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 21:30
@File    : example_usage.py
@Author  : zj
@Description: Quick usage example for DataFlow-CV

This script provides a quick demonstration of DataFlow-CV functionality.
For detailed examples, see the samples/ directory.
"""

import os
import sys
import tempfile
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import dataflow
    from dataflow.config import Config
    print("✅ DataFlow-CV imported successfully")
    print(f"   Version: {dataflow.__version__}")
    print(f"   Description: {dataflow.__description__}")
except ImportError as e:
    print(f"❌ Failed to import DataFlow-CV: {e}")
    print("   Make sure to install it first: pip install -e .")
    sys.exit(1)


def demonstrate_coco_to_yolo():
    """Demonstrate COCO to YOLO conversion."""
    print("\n" + "="*60)
    print("COCO to YOLO Conversion Demo")
    print("="*60)

    # Create a temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a minimal COCO JSON
        coco_json = os.path.join(temp_dir, "demo_coco.json")
        coco_data = {
            "info": {"description": "Demo dataset"},
            "images": [
                {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "image2.jpg", "width": 800, "height": 600}
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [100, 150, 200, 120], "area": 24000, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 2,
                 "bbox": [300, 200, 150, 100], "area": 15000, "iscrowd": 0}
            ],
            "categories": [
                {"id": 1, "name": "person", "supercategory": "human"},
                {"id": 2, "name": "car", "supercategory": "vehicle"}
            ]
        }

        with open(coco_json, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"\nCreated demo COCO JSON: {coco_json}")
        print(f"  - Images: {len(coco_data['images'])}")
        print(f"  - Annotations: {len(coco_data['annotations'])}")
        print(f"  - Categories: {len(coco_data['categories'])}")

        # Convert to YOLO format
        output_dir = os.path.join(temp_dir, "yolo_output")
        print(f"\nConverting to YOLO format in: {output_dir}")

        try:
            result = dataflow.coco_to_yolo(coco_json, output_dir)
            print("✅ Conversion successful!")
            print(f"\nStatistics:")
            print(f"  - Images processed: {result.get('images_processed', 0)}")
            print(f"  - Annotations processed: {result.get('annotations_processed', 0)}")
            print(f"  - Images with annotations: {result.get('images_with_annotations', 0)}")

            # Show generated files
            print(f"\nGenerated files:")
            classes_file = os.path.join(output_dir, Config.YOLO_CLASSES_FILENAME)
            labels_dir = os.path.join(output_dir, Config.YOLO_LABELS_DIRNAME)

            if os.path.exists(classes_file):
                with open(classes_file, 'r') as f:
                    classes = [line.strip() for line in f]
                print(f"  - {Config.YOLO_CLASSES_FILENAME}: {classes}")

            if os.path.exists(labels_dir):
                label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
                print(f"  - {Config.YOLO_LABELS_DIRNAME}/: {len(label_files)} label files")

        except Exception as e:
            print(f"❌ Conversion failed: {e}")


def demonstrate_yolo_to_coco():
    """Demonstrate YOLO to COCO conversion."""
    print("\n" + "="*60)
    print("YOLO to COCO Conversion Demo")
    print("="*60)

    # Create a temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create directory structure
        images_dir = os.path.join(temp_dir, "images")
        labels_dir = os.path.join(temp_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Create dummy image files
        for i in range(2):
            img_path = os.path.join(images_dir, f"image{i+1}.jpg")
            with open(img_path, 'wb') as f:
                f.write(b"")  # Empty file for demo

        # Create label files
        label1 = os.path.join(labels_dir, "image1.txt")
        with open(label1, 'w') as f:
            f.write("0 0.3 0.4 0.2 0.3\n")  # person
            f.write("0 0.6 0.5 0.15 0.25\n")  # another person

        label2 = os.path.join(labels_dir, "image2.txt")
        with open(label2, 'w') as f:
            f.write("1 0.5 0.5 0.25 0.2\n")  # car

        # Create classes file
        classes_file = os.path.join(temp_dir, "classes.names")
        with open(classes_file, 'w') as f:
            f.write("person\n")
            f.write("car\n")

        print(f"\nCreated demo YOLO data in: {temp_dir}")
        print(f"  - Images: {images_dir} (2 dummy files)")
        print(f"  - Labels: {labels_dir} (2 label files)")
        print(f"  - Classes: {classes_file}")

        # Convert to COCO format
        coco_output = os.path.join(temp_dir, "annotations.json")
        print(f"\nConverting to COCO format: {coco_output}")

        try:
            result = dataflow.yolo_to_coco(images_dir, labels_dir, classes_file, coco_output)
            print("✅ Conversion successful!")
            print(f"\nStatistics:")
            print(f"  - Images processed: {result.get('images_processed', 0)}")
            print(f"  - Annotations processed: {result.get('annotations_processed', 0)}")
            print(f"  - Images with annotations: {result.get('images_with_annotations', 0)}")

            # Show COCO structure
            if os.path.exists(coco_output):
                with open(coco_output, 'r') as f:
                    coco_data = json.load(f)
                print(f"\nCOCO structure:")
                print(f"  - Images: {len(coco_data.get('images', []))}")
                print(f"  - Annotations: {len(coco_data.get('annotations', []))}")
                print(f"  - Categories: {len(coco_data.get('categories', []))}")

        except Exception as e:
            print(f"❌ Conversion failed: {e}")


def show_cli_commands():
    """Show available CLI commands."""
    print("\n" + "="*60)
    print("CLI Commands Quick Reference")
    print("="*60)

    print("\nCOCO to YOLO:")
    print("  dataflow convert coco2yolo <coco_json> <output_dir>")
    print("  dataflow convert coco2yolo --help")

    print("\nYOLO to COCO:")
    print("  dataflow convert yolo2coco <images_dir> <labels_dir> <classes_file> <output_json>")
    print("  dataflow convert yolo2coco --help")

    print("\nGlobal options:")
    print("  --verbose, -v    Enable verbose output")
    print("  --overwrite      Overwrite existing files")
    print("  --help           Show help message")

    print("\nConfiguration:")
    print("  dataflow config   Show current configuration")


def main():
    """Main demonstration function."""
    print("="*60)
    print("DataFlow-CV Quick Usage Demonstration")
    print("="*60)

    # Show module info
    print(f"\n📦 Module: dataflow v{dataflow.__version__}")
    print(f"   {dataflow.__description__}")

    # Demo conversions
    demonstrate_coco_to_yolo()
    demonstrate_yolo_to_coco()

    # Show CLI commands
    show_cli_commands()

    print("\n" + "="*60)
    print("✅ Demonstration completed!")
    print("="*60)
    print("\n📚 For detailed examples, see:")
    print("   - samples/cli/convert/  for CLI usage")
    print("   - samples/api/convert/  for Python API")
    print("\n🧪 Run tests with:")
    print("   python tests/run_tests.py")


if __name__ == "__main__":
    main()