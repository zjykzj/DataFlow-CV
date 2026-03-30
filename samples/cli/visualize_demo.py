#!/usr/bin/env python3
"""
CLI visualization function example script

Demonstrates how to use the dataflow-cv visualize command to visualize three label formats:
1. YOLO format visualization
2. COCO format visualization
3. LabelMe format visualization

Usage:
1. Run in project root directory: python samples/cli/visualize_demo.py
2. Or run directly: ./samples/cli/visualize_demo.py
"""

import subprocess
import sys
from pathlib import Path

# Add project root directory to PATH to import dataflow module
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_cli_command(cmd_args):
    """Run CLI command and return result"""
    print(f"Executing command: dataflow-cv {' '.join(cmd_args)}")
    try:
        result = subprocess.run(
            ["dataflow-cv"] + cmd_args,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=30,
        )
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print(f"Standard output:\n{result.stdout}")
        if result.stderr:
            print(f"Standard error:\n{result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Error: Command execution timeout")
        return False
    except FileNotFoundError:
        print("Error: dataflow-cv command not found, please install package first: pip install -e .")
        return False
    except Exception as e:
        print(f"Error: Exception occurred while executing command: {e}")
        return False

def demo_yolo_visualization():
    """Demonstrate YOLO format visualization"""
    print("\n" + "="*60)
    print("Demonstrating YOLO format visualization")
    print("="*60)

    yolo_dir = project_root / "assets" / "test_data" / "seg" / "yolo"
    image_dir = yolo_dir / "images"
    label_dir = yolo_dir / "labels"
    class_file = yolo_dir / "classes.txt"
    output_dir = project_root / "temp_output" / "visualize" / "yolo"

    # Clean up old output directory
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    cmd = [
        "visualize", "yolo",
        str(image_dir), str(label_dir), str(class_file),
        "--save", str(output_dir),
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ YOLO visualization successful, result saved at: {output_dir}")
        # Count generated files
        image_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        print(f"  Generated {len(image_files)} visualization images")
    else:
        print("✗ YOLO visualization failed")
    return success

def demo_coco_visualization():
    """Demonstrate COCO format visualization"""
    print("\n" + "="*60)
    print("Demonstrating COCO format visualization")
    print("="*60)

    coco_file = project_root / "assets" / "test_data" / "seg" / "coco" / "annotations.json"
    image_dir = project_root / "assets" / "test_data" / "seg" / "coco" / "images"
    output_dir = project_root / "temp_output" / "visualize" / "coco"

    # Clean up old output directory
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    cmd = [
        "visualize", "coco",
        str(image_dir), str(coco_file),
        "--save", str(output_dir),
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ COCO visualization successful, result saved at: {output_dir}")
        image_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        print(f"  Generated {len(image_files)} visualization images")
    else:
        print("✗ COCO visualization failed")
    return success

def demo_labelme_visualization():
    """Demonstrate LabelMe format visualization"""
    print("\n" + "="*60)
    print("Demonstrating LabelMe format visualization")
    print("="*60)

    labelme_dir = project_root / "assets" / "test_data" / "seg" / "labelme"
    output_dir = project_root / "temp_output" / "visualize" / "labelme"

    # Clean up old output directory
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    # LabelMe format: images and JSON files in the same directory
    cmd = [
        "visualize", "labelme",
        str(labelme_dir), str(labelme_dir),
        "--save", str(output_dir),
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ LabelMe visualization successful, result saved at: {output_dir}")
        image_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        print(f"  Generated {len(image_files)} visualization images")
    else:
        print("✗ LabelMe visualization failed")
    return success

def main():
    """Main function"""
    print("DataFlow-CV CLI visualization function demonstration")
    print("="*60)

    # Check if test data exists
    test_data_root = project_root / "assets" / "test_data" / "seg"
    if not test_data_root.exists():
        print(f"Error: Test data directory does not exist: {test_data_root}")
        print("Please ensure running this script in the project root directory")
        return 1

    # Create temporary output directory
    temp_root = project_root / "temp_output" / "visualize"
    temp_root.mkdir(parents=True, exist_ok=True)

    successes = []

    # Demonstrate three format visualizations
    successes.append(demo_yolo_visualization())
    successes.append(demo_coco_visualization())
    successes.append(demo_labelme_visualization())

    # Summary
    print("\n" + "="*60)
    print("Demonstration summary")
    print("="*60)
    total = len(successes)
    passed = sum(successes)
    print(f"Total demonstrated {total} format visualizations")
    print(f"Successful: {passed}, Failed: {total - passed}")

    if all(successes):
        print("✓ All demonstrations completed successfully!")
        print(f"Visualization results saved at: {project_root / 'temp_output' / 'visualize'}")
        print("\nYou can manually test with the following commands:")
        print("  dataflow-cv visualize yolo assets/test_data/seg/yolo/images assets/test_data/seg/yolo/labels assets/test_data/seg/yolo/classes.txt --save ./output")
        print("  dataflow-cv visualize coco assets/test_data/seg/coco/images assets/test_data/seg/coco/annotations.json --save ./output")
        print("  dataflow-cv visualize labelme assets/test_data/seg/labelme assets/test_data/seg/labelme --save ./output")
        return 0
    else:
        print("✗ Some demonstrations failed, please check error messages")
        return 1

if __name__ == "__main__":
    sys.exit(main())