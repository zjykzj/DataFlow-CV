#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/10
@File    : cli_coco_to_yolo.py
@Author  : DataFlow Team
@Description: CLI example for COCO to YOLO conversion

This script demonstrates how to use the DataFlow-CV CLI tool
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


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def create_sample_coco_data():
    """Create sample COCO JSON data for demonstration."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="coco2yolo_demo_")
    print(f"Created temporary directory: {temp_dir}")

    # Create sample COCO JSON
    coco_json = os.path.join(temp_dir, "annotations.json")
    coco_data = {
        "info": {
            "description": "Sample COCO dataset for COCO→YOLO conversion demo",
            "version": "1.0",
            "year": 2026,
            "contributor": "DataFlow-CV",
            "date_created": "2026-03-10"
        },
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": "cat_image.jpg",
                "width": 800,
                "height": 600
            },
            {
                "id": 2,
                "file_name": "dog_image.jpg",
                "width": 1024,
                "height": 768
            },
            {
                "id": 3,
                "file_name": "empty_image.jpg",  # Image without annotations
                "width": 640,
                "height": 480
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 150, 200, 120],  # [x, y, width, height]
                "area": 24000,
                "segmentation": [[100, 150, 300, 150, 300, 270, 100, 270]],
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
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "cat",
                "supercategory": "animal"
            },
            {
                "id": 2,
                "name": "dog",
                "supercategory": "animal"
            }
        ]
    }

    with open(coco_json, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2)

    print(f"Created sample COCO JSON: {coco_json}")
    return temp_dir, coco_json


def show_cli_commands(coco_json, output_dir):
    """Show available CLI commands for COCO to YOLO conversion."""
    print_header("CLI COMMANDS FOR COCO→YOLO CONVERSION")

    print("\nBasic conversion:")
    print(f"  $ dataflow convert coco2yolo {coco_json} {output_dir}")

    print("\nWith verbose output:")
    print(f"  $ dataflow convert coco2yolo --verbose {coco_json} {output_dir}")
    print(f"  $ dataflow convert coco2yolo -v {coco_json} {output_dir}")


    print("\nWith segmentation mode (for polygon annotations):")
    print(f"  $ dataflow convert coco2yolo --segmentation {coco_json} {output_dir}")

    print("\nWith both options:")
    print(f"  $ dataflow convert coco2yolo -v --segmentation {coco_json} {output_dir}")

    print("\nGet help:")
    print(f"  $ dataflow convert coco2yolo --help")


def run_conversion(coco_json, output_dir, verbose=True, segmentation=False):
    """Run the actual conversion using Python module."""
    print_header("RUNNING CONVERSION")

    import subprocess

    # Build command
    cmd = ["python", "-m", "dataflow.cli", "convert", "coco2yolo"]
    if verbose:
        cmd.append("--verbose")
    if segmentation:
        cmd.append("--segmentation")
    cmd.extend([coco_json, output_dir])

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


def inspect_output(output_dir):
    """Inspect the generated output files."""
    print_header("INSPECTING OUTPUT FILES")

    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return

    print(f"\nDirectory structure of {output_dir}:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')

    # Check for class names file
    classes_file = os.path.join(output_dir, "class.names")
    if os.path.exists(classes_file):
        print(f"\nContents of {classes_file}:")
        with open(classes_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    print(f"  {i}. {line}")

    # Check for labels directory
    labels_dir = os.path.join(output_dir, "labels")
    if os.path.exists(labels_dir):
        print(f"\nLabel files in {labels_dir}:")
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        for label_file in sorted(label_files)[:3]:  # Show first 3
            label_path = os.path.join(labels_dir, label_file)
            print(f"\n  {label_file}:")
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                for i, line in enumerate(lines[:3], 1):  # Show first 3 lines
                    print(f"    Line {i}: {line}")
                if len(lines) > 3:
                    print(f"    ... and {len(lines) - 3} more lines")

        if len(label_files) > 3:
            print(f"\n  ... and {len(label_files) - 3} more label files")


def main():
    """Main demonstration function."""
    print_header("COCO TO YOLO CONVERSION - CLI DEMONSTRATION")

    # Create sample data
    temp_dir, coco_json = create_sample_coco_data()
    output_dir = os.path.join(temp_dir, "yolo_output")

    try:
        # Show CLI commands
        show_cli_commands(coco_json, output_dir)

        # Run conversion
        success = run_conversion(coco_json, output_dir, verbose=True)

        if success:
            # Inspect output
            inspect_output(output_dir)

        # Demonstrate segmentation mode
        output_dir_seg = os.path.join(temp_dir, "yolo_output_seg")
        print_header("SEGMENTATION MODE DEMONSTRATION")
        run_conversion(coco_json, output_dir_seg, verbose=True, segmentation=True)

        print_header("SUMMARY")
        print(f"\n✅ Demonstration completed!")
        print(f"\n📁 Sample data directory: {temp_dir}")
        print(f"   - COCO JSON: {coco_json}")
        print(f"   - YOLO output: {output_dir}")
        print(f"   - YOLO output (segmentation): {output_dir_seg}")

        print(f"\n💡 Key points:")
        print(f"   1. COCO JSON should contain images, annotations, and categories")
        print(f"   2. Output directory will contain 'labels/' and 'class.names'")
        print(f"   3. Each image gets a corresponding .txt file in labels/")
        print(f"   4. Use --verbose for detailed progress information")
        print(f"   5. Use --segmentation for polygon annotations")

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