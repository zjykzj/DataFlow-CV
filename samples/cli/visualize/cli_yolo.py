#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:40
@File    : cli_yolo.py
@Author  : zj
@Description: Example usage of YOLO visualization via CLI
"""

import os
import sys
import tempfile
import numpy as np
import cv2

# Add parent directory to path to import dataflow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

def create_sample_yolo_dataset():
    """Create a sample YOLO dataset for demonstration."""
    temp_dir = tempfile.mkdtemp(prefix="yolo_vis_demo_")

    # Create directories
    image_dir = os.path.join(temp_dir, "images")
    label_dir = os.path.join(temp_dir, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Create class file
    class_file = os.path.join(temp_dir, "classes.names")
    with open(class_file, "w") as f:
        f.write("person\n")
        f.write("car\n")
        f.write("dog\n")
        f.write("cat\n")

    # Create sample images with annotations
    for i in range(3):
        # Create image
        img_path = os.path.join(image_dir, f"image_{i:03d}.jpg")
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.imwrite(img_path, img)

        # Create corresponding label file
        label_path = os.path.join(label_dir, f"image_{i:03d}.txt")
        with open(label_path, "w") as f:
            # Add random annotations
            import random
            for j in range(random.randint(1, 3)):
                class_id = random.randint(0, 3)
                x_center = random.uniform(0.2, 0.8)
                y_center = random.uniform(0.2, 0.8)
                width = random.uniform(0.1, 0.3)
                height = random.uniform(0.1, 0.3)
                confidence = random.uniform(0.5, 0.99)
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}\n")

    return temp_dir, image_dir, label_dir, class_file

def main():
    """Demonstrate YOLO visualization via CLI."""
    print("=" * 60)
    print("YOLO Visualization CLI Example")
    print("=" * 60)
    print()

    # Create sample dataset
    print("Creating sample YOLO dataset...")
    temp_dir, image_dir, label_dir, class_file = create_sample_yolo_dataset()
    print(f"Created sample dataset in: {temp_dir}")
    print(f"  Images: {image_dir}")
    print(f"  Labels: {label_dir}")
    print(f"  Classes: {class_file}")
    print()

    # Example 1: Display visualization (interactive)
    print("Example 1: Interactive display")
    print("Command:")
    print(f"  dataflow visualize yolo {image_dir} {label_dir} {class_file}")
    print()
    print("This will open windows showing each image with annotations.")
    print("Press any key to continue to next image, ESC to exit.")
    print()

    # Example 2: Save visualization to directory
    output_dir = os.path.join(temp_dir, "visualized")
    print("Example 2: Save visualization to directory")
    print("Command:")
    print(f"  dataflow visualize yolo {image_dir} {label_dir} {class_file} --save {output_dir}")
    print()
    print(f"This will save annotated images to: {output_dir}")
    print()

    # Example 3: With verbose output
    print("Example 3: With verbose output")
    print("Command:")
    print(f"  dataflow visualize yolo {image_dir} {label_dir} {class_file} --save {output_dir} -v")
    print()
    print("The -v flag enables detailed progress logging.")
    print()

    # Example 4: Using Python API instead of CLI
    print("Example 4: Using Python API")
    print("Python code:")
    print('''  import dataflow

  result = dataflow.visualize_yolo(
      image_dir="path/to/images",
      label_dir="path/to/labels",
      class_path="path/to/classes.names",
      save_dir="path/to/output"  # optional
  )

  print(f"Processed {result['images_processed']} images")
  print(f"Found {result['annotations_processed']} annotations")''')
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