#!/usr/bin/env python3
"""
Batch conversion examples for DataFlow convert module.
Demonstrates batch conversion functions for processing directories.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataflow import convert


def create_test_images(tmpdir, num_images=3):
    """Create multiple test images with annotations."""
    import cv2
    import numpy as np

    images = []
    annotations = []

    for i in range(num_images):
        # Create test image with colored rectangle
        image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        color = (0, 0, 255) if i % 2 == 0 else (0, 255, 0)  # Red or green
        x, y = 50 + i * 100, 50 + i * 50
        cv2.rectangle(image, (x, y), (x + 100, y + 100), color, -1)

        # Save image
        image_path = str(tmpdir / f"test_image_{i+1:03d}.jpg")
        cv2.imwrite(image_path, image)
        images.append(image_path)

        # Create corresponding annotation files in different formats
        annotations.append({
            'image_path': image_path,
            'idx': i,
            'color': 'red' if i % 2 == 0 else 'green'
        })

    return images, annotations


def example_batch_yolo_to_coco():
    """Example: Batch convert YOLO annotations to COCO format."""
    print("Example: Batch YOLO to COCO conversion")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test images and annotations
        images, ann_info = create_test_images(tmpdir, 3)
        class_names = ["red_box", "green_box"]

        # Create YOLO annotation files
        yolo_dir = tmpdir / "yolo_labels"
        yolo_dir.mkdir()

        for i, (image_path, info) in enumerate(zip(images, ann_info)):
            # Create YOLO annotation
            yolo_path = yolo_dir / f"test_image_{i+1:03d}.txt"

            # Calculate normalized coordinates
            img_width, img_height = 400, 300
            x = 50 + info['idx'] * 100
            y = 50 + info['idx'] * 50
            width, height = 100, 100

            xc = (x + width / 2) / img_width
            yc = (y + height / 2) / img_height
            w_norm = width / img_width
            h_norm = height / img_height

            class_idx = 0 if info['color'] == 'red' else 1

            with open(yolo_path, 'w') as f:
                f.write(f"{class_idx} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        print(f"Created {len(images)} test images and YOLO annotations")
        print(f"Image directory: {tmpdir / 'test_image_*.jpg'}")
        print(f"YOLO directory: {yolo_dir}")

        # Batch convert using CLI (simulated)
        print("\nBatch conversion (always creates single COCO file):")
        print(f"   dataflow convert yolo2coco {tmpdir} {yolo_dir} classes.txt {tmpdir / 'coco_output.json'} --batch")
        print(f"   # If output is a directory, creates coco_annotations.json inside it")

        # Actually perform batch conversion using Python API
        print("\nPerforming batch conversion with Python API...")

        # Create class names file
        classes_file = tmpdir / "classes.txt"
        with open(classes_file, 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")

        # Perform batch conversion using batch_yolo_to_coco (single COCO file)
        output_file = tmpdir / "coco_output.json"

        print(f"\nFound {len(pairs)} image-annotation pairs:")
        for img_path, ann_path in pairs:
            print(f"  {Path(img_path).name} ↔ {Path(ann_path).name}")

        # Perform batch conversion (creates single COCO file)
        try:
            convert.batch_yolo_to_coco(pairs, class_names, str(output_file))
            print(f"  ✓ Created single COCO file: {output_file.name}")

            # Show summary
            with open(output_file, 'r') as f:
                coco_data = json.load(f)
                print(f"  Contains {len(coco_data.get('images', []))} images")
                print(f"  Contains {len(coco_data.get('annotations', []))} annotations")
                print(f"  Contains {len(coco_data.get('categories', []))} categories")

        except Exception as e:
            print(f"  ✗ Error creating COCO file: {e}")
        print()


def example_batch_labelme_to_coco():
    """Example: Batch convert LabelMe annotations to COCO format."""
    print("Example: Batch LabelMe to COCO conversion")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create LabelMe annotation files (no images needed)
        labelme_dir = tmpdir / "labelme_annotations"
        labelme_dir.mkdir()

        for i in range(3):
            labelme_data = {
                "version": "5.3.1",
                "flags": {},
                "shapes": [
                    {
                        "label": f"object_{i}",
                        "points": [[50, 50], [150, 150]],
                        "shape_type": "rectangle"
                    }
                ],
                "imagePath": f"image_{i+1:03d}.jpg",
                "imageData": None,
                "imageHeight": 300,
                "imageWidth": 400
            }

            labelme_path = labelme_dir / f"image_{i+1:03d}.json"
            with open(labelme_path, 'w') as f:
                json.dump(labelme_data, f, indent=2)

        print(f"Created 3 LabelMe annotation files in {labelme_dir}")

        print("\nBatch conversion (always creates single COCO file):")
        print(f"   dataflow convert labelme2coco {labelme_dir} {tmpdir / 'coco_output.json'} --batch")
        print(f"   # If output is a directory, creates coco_annotations.json inside it")

        # Actually perform combined batch conversion using Python API
        print("\nPerforming combined batch conversion with Python API...")

        # Get all LabelMe files
        labelme_files = list(labelme_dir.glob("*.json"))
        pairs = [(str(f), str(f)) for f in labelme_files]  # Same file for input and annotation

        # Perform combined batch conversion
        combined_output = tmpdir / "combined_coco.json"

        try:
            convert.batch_labelme_to_coco(pairs, str(combined_output))
            print(f"  ✓ Created combined COCO file: {combined_output.name}")

            # Show summary
            with open(combined_output, 'r') as f:
                coco_data = json.load(f)
                print(f"  Contains {len(coco_data.get('images', []))} images")
                print(f"  Contains {len(coco_data.get('annotations', []))} annotations")
                print(f"  Contains {len(coco_data.get('categories', []))} categories")

        except Exception as e:
            print(f"  ✗ Error creating combined COCO file: {e}")

        print()


def example_batch_coco_to_yolo():
    """Example: Batch convert COCO annotations to YOLO format."""
    print("Example: Batch COCO to YOLO conversion")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test images
        images, ann_info = create_test_images(tmpdir, 2)

        # Create COCO annotation files (one per image)
        coco_dir = tmpdir / "coco_annotations"
        coco_dir.mkdir()

        class_names = ["red_box", "green_box"]

        for i, (image_path, info) in enumerate(zip(images, ann_info)):
            coco_data = {
                "images": [{
                    "id": i + 1,
                    "width": 400,
                    "height": 300,
                    "file_name": Path(image_path).name
                }],
                "annotations": [
                    {
                        "id": i + 1,
                        "image_id": i + 1,
                        "category_id": 1 if info['color'] == 'red' else 2,
                        "bbox": [50 + i * 100, 50 + i * 50, 100, 100],
                        "area": 10000,
                        "iscrowd": 0
                    }
                ],
                "categories": [
                    {"id": 1, "name": "red_box", "supercategory": "object"},
                    {"id": 2, "name": "green_box", "supercategory": "object"}
                ]
            }

            coco_path = coco_dir / f"{Path(image_path).stem}.json"
            with open(coco_path, 'w') as f:
                json.dump(coco_data, f, indent=2)

        print(f"Created {len(images)} test images and COCO annotations")
        print(f"Image directory: {tmpdir / 'test_image_*.jpg'}")
        print(f"COCO directory: {coco_dir}")

        print("\nBatch conversion using CLI:")
        print(f"   dataflow convert coco2yolo {tmpdir} {coco_dir} {tmpdir / 'yolo_output'} --batch --class-names classes.txt")

        # Create class names file
        classes_file = tmpdir / "classes.txt"
        with open(classes_file, 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")

        # Perform batch conversion using batch utility function
        print("\nPerforming batch conversion with batch_process_conversion...")

        from dataflow.convert.batch import batch_process_conversion, find_matching_conversion_pairs

        # Find matching pairs
        pairs = find_matching_conversion_pairs(str(tmpdir), str(coco_dir), '.json')

        output_dir = tmpdir / "yolo_output"

        # Use batch_process_conversion utility
        batch_process_conversion(
            pairs,
            convert.coco_to_yolo,
            str(output_dir),
            needs_image=True,
            class_names=class_names
        )

        print(f"\nYOLO files saved to: {output_dir}")
        print()


def example_batch_utility_functions():
    """Example: Using batch utility functions."""
    print("Example: Batch Utility Functions")
    print("-" * 40)

    print("\n1. find_matching_conversion_pairs()")
    print("   Finds matching image-annotation pairs between directories")
    print("""
   pairs = find_matching_conversion_pairs(
       image_dir="images/",
       annotation_dir="annotations/",
       annotation_ext=".json",
       needs_input=True  # Set to False for conversions without images
   )
   """)

    print("\n2. batch_process_conversion()")
    print("   Generic batch processing function")
    print("""
   batch_process_conversion(
       pairs,                     # List of (input_path, annotation_path) tuples
       convert_func,              # Single conversion function
       output_path,               # Output directory or file
       needs_image=True,          # Whether conversion needs image
       **kwargs                   # Additional args for conversion function
   )
   """)

    print("\n3. batch_convert_with_combined_option()")
    print("   Batch conversion with combined output support (legacy)")
    print("""
   batch_convert_with_combined_option(
       pairs,
       single_convert_func,       # Function for single file conversion
       batch_convert_func=None,   # Optional function for batch conversion
       output_path="output/",
       combined=False,            # Whether to combine outputs
       **kwargs
   )
   # Note: CLI no longer uses --combined flag for COCO outputs
   # batch mode always creates single COCO files for yolo2coco and labelme2coco
   """)

    print("\n4. Available batch conversion functions:")
    print("   - batch_coco_to_yolo()")
    print("   - batch_yolo_to_coco()")
    print("   - batch_labelme_to_coco()")
    print("   - batch_coco_to_labelme()")
    print("   - batch_labelme_to_yolo()")
    print("   - batch_yolo_to_labelme()")
    print()


def main():
    """Run all batch conversion examples."""
    print("=" * 60)
    print("DataFlow Batch Conversion Examples")
    print("=" * 60)
    print()

    examples = [
        example_batch_yolo_to_coco,
        example_batch_labelme_to_coco,
        example_batch_coco_to_yolo,
        example_batch_utility_functions
    ]

    for example_func in examples:
        example_func()

    print("=" * 60)
    print("Key Features of Batch Conversion:")
    print("=" * 60)
    print("• Progress display: Shows current file and completion percentage")
    print("• Error handling: Skips files with errors, continues processing")
    print("• Flexible output: Per-file output for YOLO/LabelMe, single COCO files for yolo2coco/labelme2coco")
    print("• Smart pairing: Automatically matches images and annotations")
    print("• Memory efficient: Processes files one at a time")
    print()
    print("Batch conversion examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()