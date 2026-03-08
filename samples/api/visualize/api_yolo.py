#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:40
@File    : api_yolo.py
@Author  : zj
@Description: Example usage of YOLO visualization via Python API
"""

import os
import sys
import tempfile
import numpy as np
import cv2

# Add parent directory to path to import dataflow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import dataflow

def create_sample_yolo_dataset():
    """Create a sample YOLO dataset for demonstration."""
    temp_dir = tempfile.mkdtemp(prefix="yolo_api_demo_")

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

        # Add some visual content
        cv2.rectangle(img, (100, 100), (300, 300), (200, 200, 200), -1)
        cv2.circle(img, (400, 200), 50, (150, 150, 150), -1)

        cv2.imwrite(img_path, img)

        # Create corresponding label file
        label_path = os.path.join(label_dir, f"image_{i:03d}.txt")
        with open(label_path, "w") as f:
            # Add fixed annotations for consistency
            if i == 0:
                # Image 0: person and car
                f.write("0 0.3 0.3 0.2 0.2 0.95\n")  # person
                f.write("1 0.6 0.6 0.15 0.15 0.88\n")  # car
            elif i == 1:
                # Image 1: dog
                f.write("2 0.5 0.5 0.25 0.25 0.92\n")  # dog
            else:
                # Image 2: cat and person
                f.write("3 0.4 0.4 0.1 0.1 0.85\n")  # cat
                f.write("0 0.7 0.7 0.15 0.15 0.90\n")  # person

    return temp_dir, image_dir, label_dir, class_file

def main():
    """Demonstrate YOLO visualization via Python API."""
    print("=" * 60)
    print("YOLO Visualization Python API Example")
    print("=" * 60)
    print()

    # Create sample dataset
    print("Creating sample YOLO dataset...")
    temp_dir, image_dir, label_dir, class_file = create_sample_yolo_dataset()
    print(f"Created sample dataset in: {temp_dir}")
    print()

    # Example 1: Basic visualization (display)
    print("Example 1: Basic visualization (interactive display)")
    print("Code:")
    print('''  result = dataflow.visualize_yolo(
      image_dir=image_dir,
      label_dir=label_dir,
      class_path=class_file
  )''')
    print()

    print("Running visualization...")
    print("Windows will open showing annotated images.")
    print("Press any key to continue, ESC to exit early.")
    print()

    try:
        result = dataflow.visualize_yolo(
            image_dir=image_dir,
            label_dir=label_dir,
            class_path=class_file,
            verbose=True
        )

        print("Results:")
        print(f"  Images processed: {result['images_processed']}")
        print(f"  Annotations processed: {result['annotations_processed']}")
        print(f"  Classes found: {result['classes_found']}")
        print()
    except KeyboardInterrupt:
        print("Visualization interrupted by user")
        print()
    except Exception as e:
        print(f"Error during visualization: {e}")
        print()

    # Example 2: Save to directory instead of displaying
    print("Example 2: Save visualization to directory")
    output_dir = os.path.join(temp_dir, "visualized")
    print('''  result = dataflow.visualize_yolo(
      image_dir=image_dir,
      label_dir=label_dir,
      class_path=class_file,
      save_dir=output_dir,
      verbose=True
  )''')
    print()

    print("Running visualization with save...")
    result = dataflow.visualize_yolo(
        image_dir=image_dir,
        label_dir=label_dir,
        class_path=class_file,
        save_dir=output_dir,
        verbose=True
    )

    print("Results:")
    print(f"  Images processed: {result['images_processed']}")
    print(f"  Annotations processed: {result['annotations_processed']}")
    print(f"  Images saved: {result['saved_images']}")
    print(f"  Saved to: {result['save_dir']}")
    print()

    # Show saved files
    if os.path.exists(output_dir):
        saved_files = os.listdir(output_dir)
        print(f"Saved files ({len(saved_files)}):")
        for f in sorted(saved_files)[:5]:  # Show first 5
            print(f"  {f}")
        if len(saved_files) > 5:
            print(f"  ... and {len(saved_files) - 5} more")
    print()

    # Example 3: Using the visualizer class directly
    print("Example 3: Using YoloVisualizer class directly")
    print('''  from dataflow.visualize.yolo import YoloVisualizer

  # Create visualizer with custom settings
  visualizer = YoloVisualizer(verbose=True)

  # Customize visualization parameters
  visualizer.line_thickness = 3
  visualizer.font_scale = 0.6

  # Perform visualization
  result = visualizer.visualize(
      image_dir=image_dir,
      label_dir=label_dir,
      class_path=class_file,
      save_dir=output_dir + "_custom"
  )''')
    print()

    # Example 4: Batch visualization
    print("Example 4: Batch visualization (multiple datasets)")
    print('''  from dataflow.visualize.yolo import YoloVisualizer

  visualizer = YoloVisualizer()

  # Prepare multiple datasets
  image_dirs = [image_dir, image_dir]  # Same dataset twice for demo
  label_dirs = [label_dir, label_dir]
  class_paths = [class_file, class_file]
  save_dirs = [output_dir + "_batch1", output_dir + "_batch2"]

  results = visualizer.batch_visualize(
      image_dirs=image_dirs,
      label_dirs=label_dirs,
      class_paths=class_paths,
      save_dirs=save_dirs
  )

  for i, result in enumerate(results):
      print(f"Dataset {i+1}: {result['images_processed']} images processed")''')
    print()

    # Cleanup
    print("Note: The visualizer automatically:")
    print("  - Validates input paths")
    print("  - Matches image files with label files by name")
    print("  - Handles missing label files gracefully")
    print("  - Converts normalized YOLO coordinates to pixel coordinates")
    print("  - Generates distinct colors for different classes")
    print("  - Handles confidence scores when present")
    print()
    print(f"Sample dataset location: {temp_dir}")
    print("This directory will be cleaned up when the script exits.")
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()