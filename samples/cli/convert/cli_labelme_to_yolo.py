#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/10
@File    : cli_labelme_to_yolo.py
@Author  : DataFlow Team
@Description: CLI example for LabelMe to YOLO conversion

This script demonstrates how to use the DataFlow-CV CLI tool
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


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def create_sample_labelme_data():
    """Create sample LabelMe JSON data for demonstration."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="labelme2yolo_demo_")
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

    # Create classes.names file
    classes_file = os.path.join(temp_dir, "classes.names")
    with open(classes_file, 'w', encoding='utf-8') as f:
        f.write("person\ncar\nbicycle\n")
    print(f"Created classes.names file: {classes_file}")

    return temp_dir, label_dir, classes_file


def show_cli_commands(label_dir, classes_file, output_dir):
    """Show available CLI commands for LabelMe to YOLO conversion."""
    print_header("CLI COMMANDS FOR LABELME→YOLO CONVERSION")

    print("\nBasic conversion:")
    print(f"  $ dataflow convert labelme2yolo {label_dir} {classes_file} {output_dir}")

    print("\nWith verbose output:")
    print(f"  $ dataflow convert labelme2yolo --verbose {label_dir} {classes_file} {output_dir}")
    print(f"  $ dataflow convert labelme2yolo -v {label_dir} {classes_file} {output_dir}")

    print("\nWith overwrite mode:")
    print(f"  $ dataflow convert labelme2yolo --overwrite {label_dir} {classes_file} {output_dir}")

    print("\nWith segmentation mode (for polygon annotations):")
    print(f"  $ dataflow convert labelme2yolo --segmentation {label_dir} {classes_file} {output_dir}")

    print("\nWith both options:")
    print(f"  $ dataflow convert labelme2yolo -v --overwrite --segmentation {label_dir} {classes_file} {output_dir}")

    print("\nGet help:")
    print(f"  $ dataflow convert labelme2yolo --help")


def run_conversion(label_dir, classes_file, output_dir, verbose=True, overwrite=False, segmentation=False):
    """Run the actual conversion using Python module."""
    print_header("RUNNING CONVERSION")

    import subprocess

    # Build command
    cmd = ["python", "-m", "dataflow.cli", "convert", "labelme2yolo"]
    if verbose:
        cmd.append("--verbose")
    if overwrite:
        cmd.append("--overwrite")
    if segmentation:
        cmd.append("--segmentation")
    cmd.extend([label_dir, classes_file, output_dir])

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

    # Check for class.names file
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
    print_header("LABELME TO YOLO CONVERSION - CLI DEMONSTRATION")

    # Create sample data
    temp_dir, label_dir, classes_file = create_sample_labelme_data()
    output_dir = os.path.join(temp_dir, "yolo_output")

    try:
        # Show CLI commands
        show_cli_commands(label_dir, classes_file, output_dir)

        # Run conversion
        success = run_conversion(label_dir, classes_file, output_dir, verbose=True, overwrite=False)

        if success:
            # Inspect output
            inspect_output(output_dir)

        # Demonstrate segmentation mode
        output_dir_seg = os.path.join(temp_dir, "yolo_output_seg")
        print_header("SEGMENTATION MODE DEMONSTRATION")
        run_conversion(label_dir, classes_file, output_dir_seg, verbose=True, overwrite=False, segmentation=True)

        print_header("SUMMARY")
        print(f"\n✅ Demonstration completed!")
        print(f"\n📁 Sample data directory: {temp_dir}")
        print(f"   - LabelMe directory: {label_dir}")
        print(f"   - YOLO output: {output_dir}")
        print(f"   - YOLO output (segmentation): {output_dir_seg}")

        print(f"\n💡 Key points:")
        print(f"   1. LabelMe directory should contain JSON files (one per image)")
        print(f"   2. Output directory will contain 'labels/' and 'class.names'")
        print(f"   3. Each image gets a corresponding .txt file in labels/")
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