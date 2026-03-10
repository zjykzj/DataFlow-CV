#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/10
@File    : cli_labelme_to_coco.py
@Author  : DataFlow Team
@Description: CLI example for LabelMe to COCO conversion

This script demonstrates how to use the DataFlow-CV CLI tool
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


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def create_sample_labelme_data():
    """Create sample LabelMe JSON data for demonstration."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="labelme2coco_demo_")
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


def show_cli_commands(label_dir, classes_file, output_json):
    """Show available CLI commands for LabelMe to COCO conversion."""
    print_header("CLI COMMANDS FOR LABELME→COCO CONVERSION")

    print("\nBasic conversion:")
    print(f"  $ dataflow convert labelme2coco {label_dir} {classes_file} {output_json}")

    print("\nWith verbose output:")
    print(f"  $ dataflow convert labelme2coco --verbose {label_dir} {classes_file} {output_json}")
    print(f"  $ dataflow convert labelme2coco -v {label_dir} {classes_file} {output_json}")

    print("\nWith overwrite mode:")
    print(f"  $ dataflow convert labelme2coco --overwrite {label_dir} {classes_file} {output_json}")

    print("\nWith segmentation mode (for polygon annotations):")
    print(f"  $ dataflow convert labelme2coco --segmentation {label_dir} {classes_file} {output_json}")

    print("\nWith both options:")
    print(f"  $ dataflow convert labelme2coco -v --overwrite --segmentation {label_dir} {classes_file} {output_json}")

    print("\nGet help:")
    print(f"  $ dataflow convert labelme2coco --help")


def run_conversion(label_dir, classes_file, output_json, verbose=True, overwrite=False, segmentation=False):
    """Run the actual conversion using Python module."""
    print_header("RUNNING CONVERSION")

    import subprocess

    # Build command
    cmd = ["python", "-m", "dataflow.cli", "convert", "labelme2coco"]
    if verbose:
        cmd.append("--verbose")
    if overwrite:
        cmd.append("--overwrite")
    if segmentation:
        cmd.append("--segmentation")
    cmd.extend([label_dir, classes_file, output_json])

    print(f"Command: {' '.join(cmd)}")
    print("\n" + "-"*40)

    # Run command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        print("✅ Conversion successful!")
        print("\nOutput:")
        print(result.stdout)

        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)

        return True

    except subprocess.CalledProcessError as e:
        print("❌ Conversion failed!")
        print(f"\nError output:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def inspect_output(output_json):
    """Inspect the generated COCO JSON file."""
    print_header("INSPECTING OUTPUT FILES")

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


def main():
    """Main demonstration function."""
    print_header("LABELME TO COCO CONVERSION - CLI DEMONSTRATION")

    # Create sample data
    temp_dir, label_dir, classes_file = create_sample_labelme_data()
    output_json = os.path.join(temp_dir, "coco_annotations.json")

    try:
        # Show CLI commands
        show_cli_commands(label_dir, classes_file, output_json)

        # Run conversion
        success = run_conversion(label_dir, classes_file, output_json, verbose=True, overwrite=False)

        if success:
            # Inspect output
            inspect_output(output_json)

        # Demonstrate segmentation mode
        output_json_seg = os.path.join(temp_dir, "coco_annotations_seg.json")
        print_header("SEGMENTATION MODE DEMONSTRATION")
        run_conversion(label_dir, classes_file, output_json_seg, verbose=True, overwrite=False, segmentation=True)

        print_header("SUMMARY")
        print(f"\n✅ Demonstration completed!")
        print(f"\n📁 Sample data directory: {temp_dir}")
        print(f"   - LabelMe directory: {label_dir}")
        print(f"   - Classes file: {classes_file}")
        print(f"   - COCO JSON output: {output_json}")
        print(f"   - COCO JSON output (segmentation): {output_json_seg}")

        print(f"\n💡 Key points:")
        print(f"   1. LabelMe directory should contain JSON files (one per image)")
        print(f"   2. Classes file should have one class name per line")
        print(f"   3. Output is a single COCO JSON file")
        print(f"   4. Use --verbose for detailed progress information")
        print(f"   5. Use --overwrite to replace existing files")
        print(f"   6. Use --segmentation to enforce polygon annotations")

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