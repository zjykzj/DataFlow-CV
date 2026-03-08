#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 21:25
@File    : api_yolo_to_coco.py
@Author  : zj
@Description: Python API example for YOLO to COCO conversion

This script demonstrates how to use the DataFlow-CV Python API
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

# Import DataFlow-CV
import dataflow
from dataflow import YoloToCocoConverter
from dataflow.config import Config


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def create_sample_yolo_data():
    """Create sample YOLO data for demonstration."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="yolo2coco_api_demo_")
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

    # Output COCO JSON path
    coco_output = os.path.join(temp_dir, "annotations.json")

    return temp_dir, images_dir, labels_dir, classes_file, coco_output, image_data


def demo_convenience_function(images_dir, labels_dir, classes_file, coco_output):
    """Demonstrate using the convenience function."""
    print_header("USING CONVENIENCE FUNCTION")

    print(f"\nCode:")
    print(f"  import dataflow")
    print(f"  result = dataflow.yolo_to_coco(")
    print(f"      '{images_dir}',")
    print(f"      '{labels_dir}',")
    print(f"      '{classes_file}',")
    print(f"      '{coco_output}'")
    print(f"  )")

    print(f"\nExecuting...")
    try:
        result = dataflow.yolo_to_coco(
            images_dir,
            labels_dir,
            classes_file,
            coco_output
        )

        print(f"\n✅ Success!")
        print(f"\nResult keys: {list(result.keys())}")

        # Show important statistics
        print(f"\nConversion statistics:")
        print(f"  - Images processed: {result.get('images_processed', 0)}")
        print(f"  - Images with annotations: {result.get('images_with_annotations', 0)}")
        print(f"  - Annotations processed: {result.get('annotations_processed', 0)}")
        print(f"  - Total classes: {result.get('total_classes', 0)}")
        print(f"  - Images without labels: {result.get('images_without_labels', 0)}")
        print(f"  - Total label files: {result.get('total_label_files', 0)}")

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def demo_converter_class(images_dir, labels_dir, classes_file, coco_output):
    """Demonstrate using the converter class directly."""
    print_header("USING CONVERTER CLASS")

    print(f"\nCode:")
    print(f"  from dataflow import YoloToCocoConverter")
    print(f"  converter = YoloToCocoConverter(verbose=True)")
    print(f"  result = converter.convert(")
    print(f"      '{images_dir}',")
    print(f"      '{labels_dir}',")
    print(f"      '{classes_file}',")
    print(f"      '{coco_output}'")
    print(f"  )")

    print(f"\nExecuting...")
    try:
        converter = YoloToCocoConverter(verbose=True)
        print(f"  Converter created: {converter.__class__.__name__}")
        print(f"  Verbose mode: {converter.verbose}")

        result = converter.convert(
            images_dir,
            labels_dir,
            classes_file,
            coco_output
        )

        print(f"\n✅ Success!")
        print(f"\nResult type: {type(result)}")

        # Show additional information available through converter
        print(f"\nConverter capabilities:")
        print(f"  - Can get image files: {hasattr(converter, 'get_image_files')}")
        print(f"  - Can get label files: {hasattr(converter, 'get_label_files')}")
        print(f"  - Can read classes: {hasattr(converter, 'read_classes_file')}")
        print(f"  - Has logger: {hasattr(converter, 'logger')}")

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


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

        print("\nCOCO structure overview:")
        print(f"  - Images: {len(coco_data.get('images', []))}")
        print(f"  - Annotations: {len(coco_data.get('annotations', []))}")
        print(f"  - Categories: {len(coco_data.get('categories', []))}")

        # Show detailed information
        categories = coco_data.get('categories', [])
        if categories:
            print(f"\nCategories:")
            for cat in categories:
                print(f"  - {cat.get('name')} (id: {cat.get('id')}, "
                      f"supercategory: {cat.get('supercategory', 'none')})")

        images = coco_data.get('images', [])
        if images:
            print(f"\nImages (first 3):")
            for img in images[:3]:
                print(f"  - {img.get('file_name')}: "
                      f"{img.get('width')}x{img.get('height')} "
                      f"(id: {img.get('id')})")

        annotations = coco_data.get('annotations', [])
        if annotations:
            print(f"\nAnnotations (first 3):")
            for ann in annotations[:3]:
                bbox = ann.get('bbox', [])
                bbox_str = f"[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]" if len(bbox) == 4 else "[]"
                print(f"  - Image {ann.get('image_id')}: "
                      f"category {ann.get('category_id')}, "
                      f"bbox {bbox_str}, "
                      f"area {ann.get('area', 0):.1f}")

            # Count annotations per category
            category_counts = {}
            for ann in annotations:
                cat_id = ann.get('category_id')
                category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

            print(f"\nAnnotations per category:")
            for cat_id, count in category_counts.items():
                cat_name = next((c.get('name') for c in categories if c.get('id') == cat_id), f"unknown ({cat_id})")
                print(f"  - {cat_name}: {count} annotations")

        # Verify that segmentation annotations are preserved
        segmentation_anns = [a for a in annotations if a.get('segmentation')]
        if segmentation_anns:
            print(f"\nSegmentation annotations found: {len(segmentation_anns)}")
            for ann in segmentation_anns[:1]:  # Show first one
                seg = ann.get('segmentation', [])
                if seg and len(seg) > 0:
                    print(f"  - Polygon with {len(seg[0])//2} points")

    except Exception as e:
        print(f"Error reading COCO JSON: {e}")


def demo_advanced_usage(images_dir, labels_dir, classes_file, temp_dir):
    """Demonstrate advanced usage patterns."""
    print_header("ADVANCED USAGE PATTERNS")

    # Pattern 1: Batch processing multiple datasets
    print(f"\n1. Batch processing multiple datasets:")
    print(f"   datasets = [")
    print(f"       {{'images': 'dataset1/images', 'labels': 'dataset1/labels', 'classes': 'dataset1/classes.names'}},")
    print(f"       {{'images': 'dataset2/images', 'labels': 'dataset2/labels', 'classes': 'dataset2/classes.names'}},")
    print(f"   ]")
    print(f"   ")
    print(f"   converter = YoloToCocoConverter(verbose=False)")
    print(f"   for dataset in datasets:")
    print(f"       output = f\"{{dataset['images']}}_annotations.json\"")
    print(f"       result = converter.convert(")
    print(f"           dataset['images'],")
    print(f"           dataset['labels'],")
    print(f"           dataset['classes'],")
    print(f"           output")
    print(f"       )")

    # Pattern 2: Custom COCO info
    print(f"\n2. Custom COCO info (requires subclassing):")
    print(f"   class CustomYoloToCocoConverter(YoloToCocoConverter):")
    print(f"       def _build_coco_structure(self, class_names):")
    print(f"           coco_data = super()._build_coco_structure(class_names)")
    print(f"           coco_data['info'] = {{")
    print(f"               'description': 'Custom dataset',")
    print(f"               'version': '2.0',")
    print(f"               'year': 2026,")
    print(f"               'contributor': 'My Team',")
    print(f"               'date_created': '2026-03-08'")
    print(f"           }}")
    print(f"           return coco_data")

    # Pattern 3: Filtering specific classes
    print(f"\n3. Filtering specific classes:")
    print(f"   # Read classes file")
    print(f"   with open('{classes_file}', 'r') as f:")
    print(f"       all_classes = [line.strip() for line in f]")
    print(f"   ")
    print(f"   # Filter classes")
    print(f"   target_classes = ['person']  # Only convert person annotations")
    print(f"   # (This would require custom implementation)")

    # Actually demonstrate a simple batch processing example
    print(f"\nExecuting batch processing demo...")
    try:
        # Create second dataset
        dataset2_dir = os.path.join(temp_dir, "dataset2")
        images_dir2 = os.path.join(dataset2_dir, "images")
        labels_dir2 = os.path.join(dataset2_dir, "labels")
        os.makedirs(images_dir2, exist_ok=True)
        os.makedirs(labels_dir2, exist_ok=True)

        # Create a simple image and label
        with open(os.path.join(images_dir2, "test.jpg"), 'wb') as f:
            f.write(b"")
        with open(os.path.join(labels_dir2, "test.txt"), 'w') as f:
            f.write("0 0.5 0.5 0.3 0.3\n")

        # Copy classes file
        classes_file2 = os.path.join(dataset2_dir, "classes.names")
        shutil.copy(classes_file, classes_file2)

        # Process both datasets
        converter = YoloToCocoConverter(verbose=False)

        output1 = os.path.join(temp_dir, "output1.json")
        result1 = converter.convert(images_dir, labels_dir, classes_file, output1)

        output2 = os.path.join(temp_dir, "output2.json")
        result2 = converter.convert(images_dir2, labels_dir2, classes_file2, output2)

        print(f"\n✅ Batch processing successful!")
        print(f"  Dataset 1: {result1.get('annotations_processed', 0)} annotations")
        print(f"  Dataset 2: {result2.get('annotations_processed', 0)} annotations")

    except Exception as e:
        print(f"\n❌ Batch processing error: {e}")


def demo_error_handling():
    """Demonstrate error handling."""
    print_header("ERROR HANDLING")

    print(f"\n1. Invalid image directory:")
    try:
        result = dataflow.yolo_to_coco(
            "/invalid/image/dir",
            "/tmp/labels",
            "/tmp/classes.names",
            "/tmp/output.json"
        )
        print(f"   ❌ Should have raised an error")
    except ValueError as e:
        print(f"   ✅ Caught expected error: {str(e)[:50]}...")

    print(f"\n2. Invalid classes file:")
    try:
        result = dataflow.yolo_to_coco(
            "/tmp/images",
            "/tmp/labels",
            "/invalid/classes.names",
            "/tmp/output.json"
        )
        print(f"   ❌ Should have raised an error")
    except ValueError as e:
        print(f"   ✅ Caught expected error: {str(e)[:50]}...")

    print(f"\n3. Empty image directory:")
    empty_dir = tempfile.mkdtemp(prefix="empty_dir_")
    try:
        result = dataflow.yolo_to_coco(
            empty_dir,
            "/tmp/labels",
            "/tmp/classes.names",
            "/tmp/output.json"
        )
        print(f"   ❌ Should have raised an error")
    except ValueError as e:
        print(f"   ✅ Caught expected error: {str(e)[:50]}...")
    finally:
        shutil.rmtree(empty_dir, ignore_errors=True)


def main():
    """Main demonstration function."""
    print_header("YOLO TO COCO CONVERSION - PYTHON API DEMONSTRATION")

    # Create sample data
    temp_dir, images_dir, labels_dir, classes_file, coco_output, image_data = create_sample_yolo_data()

    try:
        # Demo 1: Convenience function
        result1 = demo_convenience_function(images_dir, labels_dir, classes_file, coco_output)

        # Demo 2: Converter class
        coco_output2 = os.path.join(temp_dir, "annotations2.json")
        result2 = demo_converter_class(images_dir, labels_dir, classes_file, coco_output2)

        # Demo 3: Advanced usage
        demo_advanced_usage(images_dir, labels_dir, classes_file, temp_dir)

        # Demo 4: Error handling
        demo_error_handling()

        # Inspect output
        if result1:
            inspect_coco_output(coco_output)

        print_header("SUMMARY")
        print(f"\n✅ API demonstration completed!")
        print(f"\n📁 Sample data directory: {temp_dir}")
        print(f"   - Images: {images_dir} ({len(image_data)} images)")
        print(f"   - YOLO labels: {labels_dir}")
        print(f"   - Classes file: {classes_file}")
        print(f"   - COCO output 1: {coco_output}")
        print(f"   - COCO output 2: {coco_output2}")

        print(f"\n💡 Key takeaways:")
        print(f"   1. Use dataflow.yolo_to_coco() for simple conversions")
        print(f"   2. Use YoloToCocoConverter class for more control")
        print(f"   3. Supports both bounding boxes and segmentation polygons")
        print(f"   4. Handles images without label files gracefully")
        print(f"   5. Returns detailed statistics for debugging")

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