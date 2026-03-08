#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 21:25
@File    : cli_yolo_to_coco.py
@Author  : zj
@Description: CLI example for YOLO to COCO conversion

This script demonstrates how to use the DataFlow-CV CLI tool
for converting YOLO format to COCO JSON format.
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
    temp_dir = tempfile.mkdtemp(prefix="yolo2coco_demo_")
    print(f"Created temporary directory: {temp_dir}")

    # Create images directory
    images_dir = os.path.join(temp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Create sample image files (empty for demonstration)
    image_files = ["cat_image.jpg", "dog_image.jpg", "bird_image.jpg"]
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        with open(img_path, 'wb') as f:
            f.write(b"")  # Empty file for demonstration
        print(f"Created sample image: {img_path}")

    # Create YOLO labels directory
    labels_dir = os.path.join(temp_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    # Create label files
    # cat_image.jpg: one cat (class 0)
    label1 = os.path.join(labels_dir, "cat_image.txt")
    with open(label1, 'w', encoding='utf-8') as f:
        # cat: class 0, normalized bbox (x_center, y_center, width, height)
        f.write("0 0.3 0.4 0.2 0.3\n")
        # Another cat in same image
        f.write("0 0.6 0.5 0.15 0.25\n")

    # dog_image.jpg: one dog (class 1)
    label2 = os.path.join(labels_dir, "dog_image.txt")
    with open(label2, 'w', encoding='utf-8') as f:
        # dog: class 1
        f.write("1 0.5 0.5 0.25 0.2\n")

    # bird_image.jpg: no annotations (empty or missing file)
    # We'll leave it without a label file to demonstrate handling

    print(f"Created YOLO labels in: {labels_dir}")

    # Create YOLO classes file
    classes_file = os.path.join(temp_dir, "classes.names")
    with open(classes_file, 'w', encoding='utf-8') as f:
        f.write("cat\n")
        f.write("dog\n")
        f.write("bird\n")  # Class 2, even though we don't have bird annotations

    print(f"Created classes file: {classes_file}")

    # Output COCO JSON path
    coco_output = os.path.join(temp_dir, "annotations.json")

    return temp_dir, images_dir, labels_dir, classes_file, coco_output


def show_cli_commands(images_dir, labels_dir, classes_file, coco_output):
    """Show available CLI commands for YOLO to COCO conversion."""
    print_header("CLI COMMANDS FOR YOLO→COCO CONVERSION")

    print("\nBasic conversion:")
    print(f"  $ dataflow convert yolo2coco {images_dir} {labels_dir} {classes_file} {coco_output}")

    print("\nWith verbose output:")
    print(f"  $ dataflow convert yolo2coco --verbose {images_dir} {labels_dir} {classes_file} {coco_output}")
    print(f"  $ dataflow convert yolo2coco -v {images_dir} {labels_dir} {classes_file} {coco_output}")

    print("\nWith overwrite mode:")
    print(f"  $ dataflow convert yolo2coco --overwrite {images_dir} {labels_dir} {classes_file} {coco_output}")

    print("\nWith both options:")
    print(f"  $ dataflow convert yolo2coco -v --overwrite {images_dir} {labels_dir} {classes_file} {coco_output}")

    print("\nGet help:")
    print(f"  $ dataflow convert yolo2coco --help")


def run_conversion(images_dir, labels_dir, classes_file, coco_output, verbose=True, overwrite=False):
    """Run the actual conversion using Python module."""
    print_header("RUNNING CONVERSION")

    import subprocess

    # Build command
    cmd = ["python", "-m", "dataflow.cli", "convert", "yolo2coco"]
    if verbose:
        cmd.append("--verbose")
    if overwrite:
        cmd.append("--overwrite")
    cmd.extend([images_dir, labels_dir, classes_file, coco_output])

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


def inspect_coco_output(coco_output):
    """Inspect the generated COCO JSON file."""
    print_header("INSPECTING COCO OUTPUT")

    if not os.path.exists(coco_output):
        print(f"COCO JSON file not found: {coco_output}")
        return

    print(f"\nCOCO JSON file: {coco_output}")
    print(f"File size: {os.path.getsize(coco_output)} bytes")

    # Load and display structure
    try:
        with open(coco_output, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        print("\nCOCO structure:")
        print(f"  - Info: {coco_data.get('info', {}).get('description', 'N/A')}")
        print(f"  - Images: {len(coco_data.get('images', []))}")
        print(f"  - Annotations: {len(coco_data.get('annotations', []))}")
        print(f"  - Categories: {len(coco_data.get('categories', []))}")

        # Show categories
        categories = coco_data.get('categories', [])
        if categories:
            print(f"\nCategories:")
            for cat in categories:
                print(f"  - {cat.get('name')} (id: {cat.get('id')})")

        # Show images
        images = coco_data.get('images', [])
        if images:
            print(f"\nFirst 2 images:")
            for img in images[:2]:
                print(f"  - {img.get('file_name')}: {img.get('width')}x{img.get('height')}")

        # Show annotations
        annotations = coco_data.get('annotations', [])
        if annotations:
            print(f"\nFirst 2 annotations:")
            for ann in annotations[:2]:
                print(f"  - Image {ann.get('image_id')}: "
                      f"category {ann.get('category_id')}, "
                      f"bbox {ann.get('bbox')}")

        # Show statistics about images without annotations
        image_ids_with_anns = set(ann.get('image_id') for ann in annotations)
        images_without_anns = [img for img in images if img.get('id') not in image_ids_with_anns]
        if images_without_anns:
            print(f"\nImages without annotations: {len(images_without_anns)}")
            for img in images_without_anns[:2]:
                print(f"  - {img.get('file_name')}")

    except Exception as e:
        print(f"Error reading COCO JSON: {e}")


def main():
    """Main demonstration function."""
    print_header("YOLO TO COCO CONVERSION - CLI DEMONSTRATION")

    # Create sample data
    temp_dir, images_dir, labels_dir, classes_file, coco_output = create_sample_yolo_data()

    try:
        # Show CLI commands
        show_cli_commands(images_dir, labels_dir, classes_file, coco_output)

        # Run conversion
        success = run_conversion(images_dir, labels_dir, classes_file, coco_output, verbose=True, overwrite=False)

        if success:
            # Inspect output
            inspect_coco_output(coco_output)

        print_header("SUMMARY")
        print(f"\n✅ Demonstration completed!")
        print(f"\n📁 Sample data directory: {temp_dir}")
        print(f"   - Images: {images_dir}")
        print(f"   - YOLO labels: {labels_dir}")
        print(f"   - Classes file: {classes_file}")
        print(f"   - COCO output: {coco_output}")

        print(f"\n💡 Key points:")
        print(f"   1. Image directory should contain actual image files")
        print(f"   2. Labels directory should have .txt files matching image names")
        print(f"   3. Classes file should have one class name per line")
        print(f"   4. Images without label files will be included without annotations")
        print(f"   5. Use --verbose for detailed progress information")
        print(f"   6. Use --overwrite to replace existing COCO JSON file")

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