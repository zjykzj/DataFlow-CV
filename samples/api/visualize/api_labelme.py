#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/9 22:00
@File    : api_labelme.py
@Author  : zj
@Description: Example usage of LabelMe visualization via Python API
"""

import os
import sys
import tempfile
import json
import numpy as np
import cv2

# Add parent directory to path to import dataflow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import dataflow

def create_sample_labelme_dataset():
    """Create a sample LabelMe dataset for demonstration."""
    temp_dir = tempfile.mkdtemp(prefix="labelme_api_demo_")

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
    """Demonstrate LabelMe visualization via Python API."""
    print("=" * 60)
    print("LabelMe Visualization Python API Example")
    print("=" * 60)
    print()

    # Create sample dataset
    print("Creating sample LabelMe dataset...")
    temp_dir, image_dir, label_dir = create_sample_labelme_dataset()
    print(f"Created sample dataset in: {temp_dir}")
    print()

    # Example 1: Basic visualization (display)
    print("Example 1: Basic visualization (interactive display)")
    print("Code:")
    print('''  result = dataflow.visualize_labelme(
      image_dir=image_dir,
      label_dir=label_dir
  )''')
    print()

    print("Running visualization...")
    print("Windows will open showing annotated images.")
    print("Press any key to continue, ESC to exit early.")
    print()

    try:
        result = dataflow.visualize_labelme(
            image_dir=image_dir,
            label_dir=label_dir,
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
    print('''  result = dataflow.visualize_labelme(
      image_dir=image_dir,
      label_dir=label_dir,
      save_dir=output_dir,
      verbose=True
  )''')
    print()

    print("Running visualization with save...")
    result = dataflow.visualize_labelme(
        image_dir=image_dir,
        label_dir=label_dir,
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
    print("Example 3: Using LabelMeVisualizer class directly")
    print('''  from dataflow.visualize.labelme import LabelMeVisualizer

  # Create visualizer with custom settings
  visualizer = LabelMeVisualizer(verbose=True)

  # Customize visualization parameters
  visualizer.line_thickness = 3
  visualizer.font_scale = 0.6

  # Perform visualization
  result = visualizer.visualize(
      image_dir=image_dir,
      label_dir=label_dir,
      save_dir=output_dir + "_custom"
  )''')
    print()

    # Example 4: Force segmentation mode
    print("Example 4: Force segmentation mode")
    print("Only process polygon annotations, reject rectangle annotations")
    print('''  # Create visualizer with segmentation mode enabled
  visualizer = LabelMeVisualizer(segmentation=True, verbose=True)

  try:
      # This will fail for images with rectangle annotations
      result = visualizer.visualize(image_dir, label_dir)
  except ValueError as e:
      print(f"Segmentation mode error: {e}")''')
    print()

    print("Testing segmentation mode on image_001.json (only polygons)...")
    try:
        # Only test on the image with polygons only (image_001.json)
        test_label_dir = os.path.join(label_dir, "..", "labels_single")
        os.makedirs(test_label_dir, exist_ok=True)
        # Copy only the polygon-only JSON file
        import shutil
        shutil.copy(os.path.join(label_dir, "image_001.json"),
                    os.path.join(test_label_dir, "image_001.json"))

        visualizer = dataflow.LabelMeVisualizer(segmentation=True, verbose=False)
        result = visualizer.visualize(image_dir, test_label_dir)
        print(f"  Success! Processed {result['images_processed']} image")
    except Exception as e:
        print(f"  Error: {e}")
    print()

    # Example 5: Batch visualization
    print("Example 5: Batch visualization (multiple datasets)")
    print('''  from dataflow.visualize.labelme import LabelMeVisualizer

  visualizer = LabelMeVisualizer()

  # Prepare multiple datasets (same dataset twice for demo)
  image_dirs = [image_dir, image_dir]
  label_dirs = [label_dir, label_dir]
  save_dirs = [output_dir + "_batch1", output_dir + "_batch2"]

  results = visualizer.batch_visualize(
      image_dirs=image_dirs,
      label_dirs=label_dirs,
      save_dirs=save_dirs
  )

  for i, result in enumerate(results):
      if result:
          print(f"Dataset {i+1}: {result['images_processed']} images processed")
      else:
          print(f"Dataset {i+1}: Failed")''')
    print()

    # Example 6: Inspecting LabelMe data
    print("Example 6: Inspecting LabelMe data before visualization")
    print('''  import json
  import os

  # Load and inspect a LabelMe JSON file
  json_path = os.path.join(label_dir, "image_000.json")
  with open(json_path, "r", encoding="utf-8") as f:
      labelme_data = json.load(f)

  print(f"Image: {labelme_data['imagePath']}")
  print(f"Dimensions: {labelme_data['imageWidth']}x{labelme_data['imageHeight']}")
  print(f"Shapes found: {len(labelme_data['shapes'])}")

  for i, shape in enumerate(labelme_data['shapes']):
      print(f"  Shape {i+1}:")
      print(f"    Label: {shape['label']}")
      print(f"    Type: {shape['shape_type']}")
      print(f"    Points: {shape['points']}")''')
    print()

    # Example 7: Handling mixed annotation types
    print("Example 7: Handling mixed annotation types")
    print("LabelMe visualizer automatically handles:")
    print("  - Rectangle shapes (converted to bounding boxes)")
    print("  - Polygon shapes (drawn as segmentation masks)")
    print("  - Different annotation types in same image")
    print("  - Automatic coordinate scaling to image dimensions")
    print("  - Color assignment by class name")
    print()

    # Cleanup
    print("Note: The LabelMe visualizer automatically:")
    print("  - Parses LabelMe JSON format")
    print("  - Converts rectangle points to bounding boxes")
    print("  - Draws polygon shapes as segmentation")
    print("  - Matches JSON files with image files by name")
    print("  - Extracts class names from shape labels")
    print("  - Validates annotation format in segmentation mode")
    print()
    print(f"Sample dataset location: {temp_dir}")
    print("This directory will be cleaned up when the script exits.")
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()