#!/usr/bin/env python3
"""
CLI format conversion function example script

Demonstrates how to use the dataflow-cv convert command for six label format conversions:
1. YOLO → COCO
2. YOLO → LabelMe
3. COCO → YOLO
4. COCO → LabelMe
5. LabelMe → YOLO
6. LabelMe → COCO

Usage:
1. Run in project root directory: python samples/cli/convert_demo.py
2. Or run directly: ./samples/cli/convert_demo.py
"""

import subprocess
import sys
import json
from pathlib import Path

# Add project root directory to PATH to import dataflow module
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_cli_command(cmd_args):
    """Run CLI command and return result"""
    import sys
    cmd = [sys.executable, "-m", "dataflow.cli.main"] + cmd_args
    print(f"Executing command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
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
        print("Error: Python interpreter not found")
        return False
    except Exception as e:
        print(f"Error: Exception occurred while executing command: {e}")
        return False

def demo_yolo_to_coco():
    """Demonstrate YOLO to COCO format conversion"""
    print("\n" + "="*60)
    print("Demonstrating YOLO to COCO format conversion")
    print("="*60)

    yolo_dir = project_root / "assets" / "test_data" / "seg" / "yolo"
    label_dir = yolo_dir / "labels"
    class_file = yolo_dir / "classes.txt"
    image_dir = yolo_dir / "images"
    output_file = project_root / "temp_output" / "convert" / "yolo_to_coco.json"

    # Clean up old output file
    if output_file.exists():
        output_file.unlink()

    cmd = [
        "convert", "yolo2coco",
        str(image_dir),     # IMAGE_DIR (positional)
        str(label_dir),     # LABEL_DIR (positional)
        str(class_file),    # CLASS_FILE (positional)
        str(output_file),   # OUTPUT_FILE (positional)
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ YOLO→COCO conversion successful, result saved at: {output_file}")
        # Verify output file
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                image_count = len(data.get("images", []))
                annotation_count = len(data.get("annotations", []))
                category_count = len(data.get("categories", []))
                print(f"  Contains {image_count} images, {annotation_count} annotations, {category_count} categories")
        return True
    else:
        print("✗ YOLO→COCO conversion failed")
        return False

def demo_yolo_to_labelme():
    """Demonstrate YOLO to LabelMe format conversion"""
    print("\n" + "="*60)
    print("Demonstrating YOLO to LabelMe format conversion")
    print("="*60)

    yolo_dir = project_root / "assets" / "test_data" / "seg" / "yolo"
    label_dir = yolo_dir / "labels"
    class_file = yolo_dir / "classes.txt"
    image_dir = yolo_dir / "images"
    output_dir = project_root / "temp_output" / "convert" / "yolo_to_labelme"

    # Clean up old output directory
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    cmd = [
        "convert", "yolo2labelme",
        str(image_dir),     # IMAGE_DIR (positional)
        str(label_dir),     # LABEL_DIR (positional)
        str(class_file),    # CLASS_FILE (positional)
        str(output_dir),    # OUTPUT_DIR (positional)
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ YOLO→LabelMe conversion successful, result saved at: {output_dir}")
        # Count generated files
        json_files = list(output_dir.glob("*.json"))
        print(f"  Generated {len(json_files)} LabelMe annotation files")
        return True
    else:
        print("✗ YOLO→LabelMe conversion failed")
        return False

def demo_coco_to_yolo():
    """Demonstrate COCO to YOLO format conversion"""
    print("\n" + "="*60)
    print("Demonstrating COCO to YOLO format conversion")
    print("="*60)

    coco_file = project_root / "assets" / "test_data" / "seg" / "coco" / "annotations.json"
    image_dir = project_root / "assets" / "test_data" / "seg" / "coco" / "images"
    output_dir = project_root / "temp_output" / "convert" / "coco_to_yolo"

    # Clean up old output directory
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    cmd = [
        "convert", "coco2yolo",
        str(coco_file),    # COCO_FILE (positional)
        str(output_dir),   # OUTPUT_DIR (positional)
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ COCO→YOLO conversion successful, result saved at: {output_dir}")
        # Count generated files
        txt_files = list(output_dir.glob("*.txt"))
        print(f"  Generated {len(txt_files)} YOLO annotation files")
        return True
    else:
        print("✗ COCO→YOLO conversion failed")
        return False

def demo_coco_to_labelme():
    """Demonstrate COCO to LabelMe format conversion"""
    print("\n" + "="*60)
    print("Demonstrating COCO to LabelMe format conversion")
    print("="*60)

    coco_file = project_root / "assets" / "test_data" / "seg" / "coco" / "annotations.json"
    image_dir = project_root / "assets" / "test_data" / "seg" / "coco" / "images"
    output_dir = project_root / "temp_output" / "convert" / "coco_to_labelme"

    # Clean up old output directory
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    cmd = [
        "convert", "coco2labelme",
        str(coco_file),    # COCO_FILE (positional)
        str(output_dir),   # OUTPUT_DIR (positional)
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ COCO→LabelMe conversion successful, result saved at: {output_dir}")
        # Count generated files
        json_files = list(output_dir.glob("*.json"))
        print(f"  Generated {len(json_files)} LabelMe annotation files")
        return True
    else:
        print("✗ COCO→LabelMe conversion failed")
        return False

def demo_labelme_to_yolo():
    """Demonstrate LabelMe to YOLO format conversion"""
    print("\n" + "="*60)
    print("Demonstrating LabelMe to YOLO format conversion")
    print("="*60)

    labelme_dir = project_root / "assets" / "test_data" / "seg" / "labelme"
    class_file = labelme_dir / "classes.txt"
    output_dir = project_root / "temp_output" / "convert" / "labelme_to_yolo"

    # Clean up old output directory
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    cmd = [
        "convert", "labelme2yolo",
        str(labelme_dir),  # LABELME_DIR (positional)
        str(class_file),   # CLASS_FILE (positional)
        str(output_dir),   # OUTPUT_DIR (positional)
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ LabelMe→YOLO conversion successful, result saved at: {output_dir}")
        # Count generated files
        txt_files = list(output_dir.glob("*.txt"))
        print(f"  Generated {len(txt_files)} YOLO annotation files")
        return True
    else:
        print("✗ LabelMe→YOLO conversion failed")
        return False

def demo_labelme_to_coco():
    """Demonstrate LabelMe to COCO format conversion"""
    print("\n" + "="*60)
    print("Demonstrating LabelMe to COCO format conversion")
    print("="*60)

    labelme_dir = project_root / "assets" / "test_data" / "seg" / "labelme"
    class_file = labelme_dir / "classes.txt"
    output_file = project_root / "temp_output" / "convert" / "labelme_to_coco.json"

    # Clean up old output file
    if output_file.exists():
        output_file.unlink()

    cmd = [
        "convert", "labelme2coco",
        str(labelme_dir),  # LABELME_DIR (positional)
        str(class_file),   # CLASS_FILE (positional)
        str(output_file),  # OUTPUT_FILE (positional)
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        print(f"✓ LabelMe→COCO conversion successful, result saved at: {output_file}")
        # Verify output file
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                image_count = len(data.get("images", []))
                annotation_count = len(data.get("annotations", []))
                category_count = len(data.get("categories", []))
                print(f"  Contains {image_count} images, {annotation_count} annotations, {category_count} categories")
        return True
    else:
        print("✗ LabelMe→COCO conversion failed")
        return False

def main():
    """Main function"""
    print("DataFlow-CV CLI format conversion function demonstration")
    print("="*60)

    # Check if test data exists
    test_data_root = project_root / "assets" / "test_data" / "seg"
    if not test_data_root.exists():
        print(f"Error: Test data directory does not exist: {test_data_root}")
        print("Please ensure running this script in the project root directory")
        return 1

    # Create temporary output directory
    temp_root = project_root / "temp_output" / "convert"
    temp_root.mkdir(parents=True, exist_ok=True)

    successes = []

    # Demonstrate six format conversions
    successes.append(demo_yolo_to_coco())
    successes.append(demo_yolo_to_labelme())
    successes.append(demo_coco_to_yolo())
    successes.append(demo_coco_to_labelme())
    successes.append(demo_labelme_to_yolo())
    successes.append(demo_labelme_to_coco())

    # Summary
    print("\n" + "="*60)
    print("Demonstration summary")
    print("="*60)
    total = len(successes)
    passed = sum(successes)
    print(f"Total demonstrated {total} format conversions")
    print(f"Successful: {passed}, Failed: {total - passed}")

    if all(successes):
        print("✓ All demonstrations completed successfully!")
        print(f"Conversion results saved at: {project_root / 'temp_output' / 'convert'}")
        print("\nYou can manually test with the following commands:")
        print("  dataflow-cv convert yolo2coco assets/test_data/seg/yolo/images assets/test_data/seg/yolo/labels assets/test_data/seg/yolo/classes.txt ./output.json")
        print("  dataflow-cv convert yolo2labelme assets/test_data/seg/yolo/images assets/test_data/seg/yolo/labels assets/test_data/seg/yolo/classes.txt ./output")
        print("  dataflow-cv convert coco2yolo assets/test_data/seg/coco/annotations.json ./output")
        print("  dataflow-cv convert coco2labelme assets/test_data/seg/coco/annotations.json ./output")
        print("  dataflow-cv convert labelme2yolo assets/test_data/seg/labelme assets/test_data/seg/labelme/classes.txt ./output")
        print("  dataflow-cv convert labelme2coco assets/test_data/seg/labelme assets/test_data/seg/labelme/classes.txt ./output.json")
        return 0
    else:
        print("✗ Some demonstrations failed, please check error messages")
        return 1

if __name__ == "__main__":
    sys.exit(main())