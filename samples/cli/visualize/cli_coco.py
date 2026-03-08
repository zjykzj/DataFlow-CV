#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:40
@File    : cli_coco.py
@Author  : zj
@Description: Example usage of COCO visualization via CLI
"""

import os
import sys
import tempfile
import json
import numpy as np
import cv2

# Add parent directory to path to import dataflow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

def create_sample_coco_dataset():
    """Create a sample COCO dataset for demonstration."""
    temp_dir = tempfile.mkdtemp(prefix="coco_vis_demo_")

    # Create image directory
    image_dir = os.path.join(temp_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    # Create COCO annotation structure
    coco_data = {
        "info": {
            "year": 2026,
            "version": "1.0",
            "description": "Sample COCO dataset for visualization demo",
            "contributor": "DataFlow Team",
            "url": "https://github.com/zjykzj/DataFlow-CV",
            "date_created": "2026-03-08"
        },
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "human"},
            {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
            {"id": 3, "name": "car", "supercategory": "vehicle"},
            {"id": 4, "name": "dog", "supercategory": "animal"}
        ]
    }

    # Create sample images and annotations
    annotation_id = 1
    for i in range(3):
        # Image info
        image_id = i + 1
        filename = f"image_{i:03d}.jpg"
        width, height = 640, 480

        coco_data["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        # Create actual image file
        img_path = os.path.join(image_dir, filename)
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        cv2.imwrite(img_path, img)

        # Add random annotations
        import random
        num_annotations = random.randint(1, 4)
        for j in range(num_annotations):
            category_id = random.randint(1, 4)
            x = random.randint(50, width - 150)
            y = random.randint(50, height - 150)
            w = random.randint(50, 150)
            h = random.randint(50, 150)

            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "segmentation": [],
                "iscrowd": 0
            })
            annotation_id += 1

    # Save annotation file
    annotation_file = os.path.join(temp_dir, "annotations.json")
    with open(annotation_file, "w") as f:
        json.dump(coco_data, f, indent=2)

    return temp_dir, image_dir, annotation_file

def main():
    """Demonstrate COCO visualization via CLI."""
    print("=" * 60)
    print("COCO Visualization CLI Example")
    print("=" * 60)
    print()

    # Create sample dataset
    print("Creating sample COCO dataset...")
    temp_dir, image_dir, annotation_file = create_sample_coco_dataset()
    print(f"Created sample dataset in: {temp_dir}")
    print(f"  Images: {image_dir}")
    print(f"  Annotations: {annotation_file}")
    print()

    # Example 1: Display visualization (interactive)
    print("Example 1: Interactive display")
    print("Command:")
    print(f"  dataflow visualize coco {image_dir} {annotation_file}")
    print()
    print("This will open windows showing each image with annotations.")
    print("Press any key to continue to next image, ESC to exit.")
    print()

    # Example 2: Save visualization to directory
    output_dir = os.path.join(temp_dir, "visualized")
    print("Example 2: Save visualization to directory")
    print("Command:")
    print(f"  dataflow visualize coco {image_dir} {annotation_file} --save {output_dir}")
    print()
    print(f"This will save annotated images to: {output_dir}")
    print()

    # Example 3: With verbose output
    print("Example 3: With verbose output")
    print("Command:")
    print(f"  dataflow visualize coco {image_dir} {annotation_file} --save {output_dir} -v")
    print()
    print("The -v flag enables detailed progress logging.")
    print()

    # Example 4: Using Python API instead of CLI
    print("Example 4: Using Python API")
    print("Python code:")
    print('''  import dataflow

  result = dataflow.visualize_coco(
      image_dir="path/to/images",
      annotation_json="path/to/annotations.json",
      save_dir="path/to/output"  # optional
  )

  print(f"Processed {result['images_processed']} images")
  print(f"Found {result['annotations_processed']} annotations")
  print(f"Categories found: {result['categories_found']}")''')
    print()

    # Example 5: Handling COCO-specific features
    print("Example 5: COCO-specific features")
    print("The visualizer supports:")
    print("  - Bounding boxes (standard COCO format)")
    print("  - Segmentation polygons")
    print("  - Multiple annotations per image")
    print("  - Category names from the COCO JSON")
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