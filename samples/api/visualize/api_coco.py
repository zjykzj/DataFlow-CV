#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:40
@File    : api_coco.py
@Author  : zj
@Description: Example usage of COCO visualization via Python API
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

def create_sample_coco_dataset():
    """Create a sample COCO dataset for demonstration."""
    temp_dir = tempfile.mkdtemp(prefix="coco_api_demo_")

    # Create image directory
    image_dir = os.path.join(temp_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    # Create COCO annotation structure
    coco_data = {
        "info": {
            "year": 2026,
            "version": "1.0",
            "description": "Sample COCO dataset for API demonstration",
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

        # Add some visual content
        cv2.rectangle(img, (100, 100), (300, 300), (200, 200, 200), -1)
        cv2.circle(img, (400, 200), 50, (150, 150, 150), -1)

        cv2.imwrite(img_path, img)

        # Add annotations
        if i == 0:
            # Image 0: person and car
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [120, 120, 160, 160],
                "area": 160 * 160,
                "segmentation": [[120, 120, 280, 120, 280, 280, 120, 280]],
                "iscrowd": 0
            })
            annotation_id += 1

            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 3,
                "bbox": [350, 150, 100, 80],
                "area": 100 * 80,
                "segmentation": [],
                "iscrowd": 0
            })
            annotation_id += 1

        elif i == 1:
            # Image 1: bicycle
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 2,
                "bbox": [200, 200, 150, 100],
                "area": 150 * 100,
                "segmentation": [],
                "iscrowd": 0
            })
            annotation_id += 1

        else:
            # Image 2: dog
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 4,
                "bbox": [300, 300, 120, 120],
                "area": 120 * 120,
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
    """Demonstrate COCO visualization via Python API."""
    print("=" * 60)
    print("COCO Visualization Python API Example")
    print("=" * 60)
    print()

    # Create sample dataset
    print("Creating sample COCO dataset...")
    temp_dir, image_dir, annotation_file = create_sample_coco_dataset()
    print(f"Created sample dataset in: {temp_dir}")
    print()

    # Example 1: Basic visualization (display)
    print("Example 1: Basic visualization (interactive display)")
    print("Code:")
    print('''  result = dataflow.visualize_coco(
      image_dir=image_dir,
      annotation_json=annotation_file
  )''')
    print()

    print("Running visualization...")
    print("Windows will open showing annotated images.")
    print("Press any key to continue, ESC to exit early.")
    print()

    try:
        result = dataflow.visualize_coco(
            image_dir=image_dir,
            annotation_json=annotation_file,
            verbose=True
        )

        print("Results:")
        print(f"  Images processed: {result['images_processed']}")
        print(f"  Annotations processed: {result['annotations_processed']}")
        print(f"  Categories found: {result['categories_found']}")
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
    print('''  result = dataflow.visualize_coco(
      image_dir=image_dir,
      annotation_json=annotation_file,
      save_dir=output_dir,
      verbose=True
  )''')
    print()

    print("Running visualization with save...")
    result = dataflow.visualize_coco(
        image_dir=image_dir,
        annotation_json=annotation_file,
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
    print("Example 3: Using CocoVisualizer class directly")
    print('''  from dataflow.visualize.coco import CocoVisualizer

  # Create visualizer with custom settings
  visualizer = CocoVisualizer(verbose=True)

  # Customize visualization parameters
  visualizer.line_thickness = 3
  visualizer.font_scale = 0.6

  # Perform visualization
  result = visualizer.visualize(
      image_dir=image_dir,
      annotation_json=annotation_file,
      save_dir=output_dir + "_custom"
  )''')
    print()

    # Example 4: Handling COCO-specific features
    print("Example 4: COCO-specific features")
    print("The COCO visualizer supports:")
    print("  - Standard COCO bounding box format [x, y, width, height]")
    print("  - Segmentation polygons")
    print("  - Category hierarchy (supercategory)")
    print("  - Image metadata (dimensions, filenames)")
    print("  - Flexible image file location (searches if not at exact path)")
    print()

    # Example 5: Batch visualization
    print("Example 5: Batch visualization (multiple datasets)")
    print('''  from dataflow.visualize.coco import CocoVisualizer

  visualizer = CocoVisualizer()

  # Prepare multiple datasets (same dataset twice for demo)
  image_dirs = [image_dir, image_dir]
  annotation_files = [annotation_file, annotation_file]
  save_dirs = [output_dir + "_batch1", output_dir + "_batch2"]

  results = visualizer.batch_visualize(
      image_dirs=image_dirs,
      annotation_jsons=annotation_files,
      save_dirs=save_dirs
  )

  for i, result in enumerate(results):
      print(f"Dataset {i+1}: {result['images_processed']} images processed")''')
    print()

    # Example 6: Inspecting COCO data
    print("Example 6: Inspecting COCO data before visualization")
    print('''  import json

  # Load and inspect COCO data
  with open(annotation_file, "r") as f:
      coco_data = json.load(f)

  print(f"Total images: {len(coco_data['images'])}")
  print(f"Total annotations: {len(coco_data['annotations'])}")
  print(f"Categories: {[c['name'] for c in coco_data['categories']]}")

  # Check if specific images exist
  for image_info in coco_data["images"]:
      image_path = os.path.join(image_dir, image_info["file_name"])
      if os.path.exists(image_path):
          print(f"Found image: {image_info['file_name']}")
      else:
          print(f"Missing image: {image_info['file_name']}")''')
    print()

    # Cleanup
    print("Note: The COCO visualizer automatically:")
    print("  - Parses COCO JSON format")
    print("  - Groups annotations by image_id")
    print("  - Handles missing image files (searches alternatives)")
    print("  - Converts COCO bbox format to corner coordinates")
    print("  - Draws segmentation polygons when available")
    print("  - Uses category names from the COCO data")
    print()
    print(f"Sample dataset location: {temp_dir}")
    print("This directory will be cleaned up when the script exits.")
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()