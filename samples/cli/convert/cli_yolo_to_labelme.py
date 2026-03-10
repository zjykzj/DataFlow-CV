#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/10
@File    : cli_yolo_to_labelme.py
@Author  : DataFlow Team
@Description: CLI example for YOLO to LabelMe conversion

This script demonstrates how to use the DataFlow-CV CLI tool
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


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def create_sample_yolo_data():
    """Create sample YOLO data for demonstration."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="yolo2labelme_demo_")
    print(f"Created temporary directory: {temp_dir}")

    # Create images directory (with empty files for demonstration)
    images_dir = os.path.join(temp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Create sample images
    image_data = [
        ("person1.jpg", 800, 600),
        ("car1.jpg", 1024, 768),
        ("person2.jpg", 640, 480),
        ("car2.jpg", 1280, 720),
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
    # person1.jpg: two persons
    label1 = os.path.join(labels_dir, "person1.txt")
    with open(label1, 'w', encoding='utf-8') as f:
        # person: class 0
        f.write("0 0.3 0.4 0.2 0.3\n")   # First person
        f.write("0 0.6 0.5 0.15 0.25\n") # Second person

    # car1.jpg: one car
    label2 = os.path.join(labels_dir, "car1.txt")
    with open(label2, 'w', encoding='utf-8') as f:
        # car: class 1
        f.write("1 0.5 0.5 0.25 0.2\n")

    # person2.jpg: one person
    label3 = os.path.join(labels_dir, "person2.txt")
    with open(label3, 'w', encoding='utf-8') as f:
        # person: class 0
        f.write("0 0.4 0.4 0.3 0.3\n")

    # car2.jpg: segmentation example (polygon)
    label4 = os.path.join(labels_dir, "car2.txt")
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


def show_cli_commands(images_dir, labels_dir, classes_file, output_dir):
    """Show available CLI commands for YOLO to LabelMe conversion."""
    print_header("CLI COMMANDS FOR YOLO→LABELME CONVERSION")

    print("\nBasic conversion:")
    print(f"  $ dataflow convert yolo2labelme {images_dir} {labels_dir} {classes_file} {output_dir}")

    print("\nWith verbose output:")
    print(f"  $ dataflow convert yolo2labelme --verbose {images_dir} {labels_dir} {classes_file} {output_dir}")
    print(f"  $ dataflow convert yolo2labelme -v {images_dir} {labels_dir} {classes_file} {output_dir}")

    print("\nWith overwrite mode:")
    print(f"  $ dataflow convert yolo2labelme --overwrite {images_dir} {labels_dir} {classes_file} {output_dir}")

    print("\nWith segmentation mode (for polygon annotations):")
    print(f"  $ dataflow convert yolo2labelme --segmentation {images_dir} {labels_dir} {classes_file} {output_dir}")

    print("\nWith both options:")
    print(f"  $ dataflow convert yolo2labelme -v --overwrite --segmentation {images_dir} {labels_dir} {classes_file} {output_dir}")

    print("\nGet help:")
    print(f"  $ dataflow convert yolo2labelme --help")


def run_conversion(images_dir, labels_dir, classes_file, output_dir, verbose=True, overwrite=False, segmentation=False):
    """Run the actual conversion using Python module."""
    print_header("RUNNING CONVERSION")

    import subprocess

    # Build command
    cmd = ["python", "-m", "dataflow.cli", "convert", "yolo2labelme"]
    if verbose:
        cmd.append("--verbose")
    if overwrite:
        cmd.append("--overwrite")
    if segmentation:
        cmd.append("--segmentation")
    cmd.extend([images_dir, labels_dir, classes_file, output_dir])

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

    # Check for LabelMe JSON files
    labelme_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    print(f"\nLabelMe JSON files: {len(labelme_files)} files")

    # Show sample LabelMe JSON file
    if labelme_files:
        sample_file = os.path.join(output_dir, labelme_files[0])
        print(f"\nSample LabelMe file: {labelme_files[0]}")
        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"  - Version: {data.get('version', 'N/A')}")
                print(f"  - Image path: {data.get('imagePath', 'N/A')}")
                print(f"  - Image dimensions: {data.get('imageWidth', 'N/A')}x{data.get('imageHeight', 'N/A')}")
                print(f"  - Shapes count: {len(data.get('shapes', []))}")
                if data.get('shapes'):
                    shape = data['shapes'][0]
                    print(f"  - First shape: {shape.get('label')} ({shape.get('shape_type')})")
        except Exception as e:
            print(f"  Error reading LabelMe JSON: {e}")


def main():
    """Main demonstration function."""
    print_header("YOLO TO LABELME CONVERSION - CLI DEMONSTRATION")

    # Create sample data
    temp_dir, images_dir, labels_dir, classes_file = create_sample_yolo_data()
    output_dir = os.path.join(temp_dir, "labelme_output")

    try:
        # Show CLI commands
        show_cli_commands(images_dir, labels_dir, classes_file, output_dir)

        # Run conversion
        success = run_conversion(images_dir, labels_dir, classes_file, output_dir, verbose=True, overwrite=False)

        if success:
            # Inspect output
            inspect_output(output_dir)

        # Demonstrate segmentation mode
        output_dir_seg = os.path.join(temp_dir, "labelme_output_seg")
        print_header("SEGMENTATION MODE DEMONSTRATION")
        run_conversion(images_dir, labels_dir, classes_file, output_dir_seg, verbose=True, overwrite=False, segmentation=True)

        print_header("SUMMARY")
        print(f"\n✅ Demonstration completed!")
        print(f"\n📁 Sample data directory: {temp_dir}")
        print(f"   - Images directory: {images_dir}")
        print(f"   - Labels directory: {labels_dir}")
        print(f"   - Classes file: {classes_file}")
        print(f"   - LabelMe output: {output_dir}")
        print(f"   - LabelMe output (segmentation): {output_dir_seg}")

        print(f"\n💡 Key points:")
        print(f"   1. YOLO format requires images, labels, and class names")
        print(f"   2. Each image should have a corresponding .txt file in labels/")
        print(f"   3. Output directory will contain LabelMe JSON files (one per image)")
        print(f"   4. Use --verbose for detailed progress information")
        print(f"   5. Use --overwrite to replace existing files")
        print(f"   6. Use --segmentation for polygon annotations")

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