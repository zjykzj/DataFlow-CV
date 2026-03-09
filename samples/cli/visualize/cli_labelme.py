#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/9 22:00
@File    : cli_labelme.py
@Author  : zj
@Description: Example usage of LabelMe visualization via CLI
"""

import os
import sys
import tempfile
import json
import numpy as np
import cv2

# Add parent directory to path to import dataflow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

def create_sample_labelme_dataset():
    """Create a sample LabelMe dataset for demonstration."""
    temp_dir = tempfile.mkdtemp(prefix="labelme_vis_demo_")

    # Create directories
    image_dir = os.path.join(temp_dir, "images")
    label_dir = os.path.join(temp_dir, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Create sample images with annotations
    for i in range(3):
        # Image info
        filename = f"image_{i:03d}.jpg"
        width, height = 640, 480

        # Create actual image file
        img_path = os.path.join(image_dir, filename)
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Add some visual content for better visualization
        cv2.rectangle(img, (100, 100), (300, 300), (200, 200, 200), -1)
        cv2.circle(img, (400, 200), 50, (150, 150, 150), -1)
        cv2.imwrite(img_path, img)

        # Create LabelMe JSON annotation file
        json_path = os.path.join(label_dir, f"image_{i:03d}.json")

        # Create LabelMe format data
        labelme_data = {
            "version": "5.2.1",
            "flags": {},
            "shapes": [],
            "imagePath": filename,
            "imageData": None,  # We'll store image separately
            "imageHeight": height,
            "imageWidth": width
        }

        # Add annotations based on image index
        if i == 0:
            # Image 0: Rectangle (bbox) and polygon (segmentation) annotations
            # Rectangle for person
            labelme_data["shapes"].append({
                "label": "person",
                "points": [[120, 120], [280, 280]],  # [x1,y1], [x2,y2] for rectangle
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            })

            # Polygon for car
            labelme_data["shapes"].append({
                "label": "car",
                "points": [[350, 150], [450, 150], [450, 230], [350, 230]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })

        elif i == 1:
            # Image 1: Only polygon annotation for bicycle
            labelme_data["shapes"].append({
                "label": "bicycle",
                "points": [[200, 200], [350, 200], [350, 300], [200, 300]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })

        else:
            # Image 2: Multiple rectangle annotations
            labelme_data["shapes"].append({
                "label": "dog",
                "points": [[300, 300], [420, 420]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            })

            labelme_data["shapes"].append({
                "label": "cat",
                "points": [[100, 300], [200, 380]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            })

        # Save JSON file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, indent=2)

    return temp_dir, image_dir, label_dir

def main():
    """Demonstrate LabelMe visualization via CLI."""
    print("=" * 60)
    print("LabelMe Visualization CLI Example")
    print("=" * 60)
    print()

    # Create sample dataset
    print("Creating sample LabelMe dataset...")
    temp_dir, image_dir, label_dir = create_sample_labelme_dataset()
    print(f"Created sample dataset in: {temp_dir}")
    print(f"  Images: {image_dir}")
    print(f"  Labels: {label_dir}")
    print()

    # Example 1: Display visualization (interactive)
    print("Example 1: Interactive display")
    print("Command:")
    print(f"  dataflow visualize labelme {image_dir} {label_dir}")
    print()
    print("This will open windows showing each image with annotations.")
    print("Press any key to continue to next image, ESC to exit.")
    print()

    # Example 2: Save visualization to directory
    output_dir = os.path.join(temp_dir, "visualized")
    print("Example 2: Save visualization to directory")
    print("Command:")
    print(f"  dataflow visualize labelme {image_dir} {label_dir} --save {output_dir}")
    print()
    print(f"This will save annotated images to: {output_dir}")
    print()

    # Example 3: With verbose output
    print("Example 3: With verbose output")
    print("Command:")
    print(f"  dataflow visualize labelme {image_dir} {label_dir} --save {output_dir} -v")
    print()
    print("The -v flag enables detailed progress logging.")
    print()

    # Example 4: Force segmentation mode (only polygons)
    print("Example 4: Force segmentation mode (only polygon annotations)")
    print("Command:")
    print(f"  dataflow visualize labelme {image_dir} {label_dir} --segmentation")
    print()
    print("The --segmentation flag requires all annotations to be polygons.")
    print("This will raise an error if any rectangle annotations are found.")
    print("Note: In our sample dataset, image_001.json has only polygons,")
    print("      so it would work with --segmentation flag.")
    print()

    # Example 5: Using Python API instead of CLI
    print("Example 5: Using Python API")
    print("Python code:")
    print('''  import dataflow

  result = dataflow.visualize_labelme(
      image_dir="path/to/images",
      label_dir="path/to/labels",
      save_dir="path/to/output"  # optional
  )

  print(f"Processed {result['images_processed']} images")
  print(f"Found {result['annotations_processed']} annotations")
  print(f"Classes found: {result['classes_found']}")''')
    print()

    # Example 6: LabelMe-specific features
    print("Example 6: LabelMe-specific features")
    print("The visualizer supports:")
    print("  - Rectangle shapes (bounding boxes)")
    print("  - Polygon shapes (segmentation)")
    print("  - Mixed annotation types in same image")
    print("  - Automatic detection of image files from JSON metadata")
    print("  - Strict validation in segmentation mode")
    print()

    # Cleanup note
    print("Note: To actually run these commands, install DataFlow-CV first:")
    print("  pip install -e .")
    print()
    print(f"The sample dataset will be cleaned up when this script exits.")
    print(f"Temp directory: {temp_dir}")
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()