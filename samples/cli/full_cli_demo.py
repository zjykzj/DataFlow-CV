#!/usr/bin/env python3
"""
CLI complete workflow example script

Demonstrates a complete computer vision data processing workflow:
1. Visualize original YOLO labels
2. Convert YOLO labels to COCO format
3. Visualize converted COCO labels
4. Convert COCO labels back to YOLO format
5. Verify conversion accuracy

Usage:
1. Run in project root directory: python samples/cli/full_cli_demo.py
2. Or run directly: ./samples/cli/full_cli_demo.py
"""

import subprocess
import sys
import json
import shutil
from pathlib import Path

# Add project root directory to PATH to import dataflow module
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_cli_command(cmd_args, check=True):
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

        if check and result.returncode != 0:
            print(f"Command execution failed, exit code: {result.returncode}")
            return False
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

def cleanup_directory(dir_path):
    """Clean up directory"""
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

def step1_visualize_yolo():
    """Step 1: Visualize original YOLO labels"""
    print("\n" + "="*60)
    print("Step 1: Visualize original YOLO labels")
    print("="*60)

    yolo_dir = project_root / "assets" / "test_data" / "seg" / "yolo"
    class_file = yolo_dir / "classes.txt"
    image_dir = yolo_dir / "images"
    output_dir = project_root / "temp_output" / "full_demo" / "step1_yolo_visualization"

    cleanup_directory(output_dir)

    cmd = [
        "visualize", "yolo",
        str(image_dir),  # image_dir (positional)
        str(yolo_dir / "labels"),  # label_dir (positional)
        str(class_file),  # class_file (positional)
        "--save",
        str(output_dir),  # output directory for --save
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        image_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        print(f"✓ Step 1 completed: Generated {len(image_files)} YOLO visualization images")
        return True, output_dir
    else:
        print("✗ Step 1 failed")
        return False, None

def step2_yolo_to_coco():
    """Step 2: YOLO to COCO format conversion"""
    print("\n" + "="*60)
    print("Step 2: YOLO to COCO format conversion")
    print("="*60)

    yolo_dir = project_root / "assets" / "test_data" / "seg" / "yolo"
    class_file = yolo_dir / "classes.txt"
    image_dir = yolo_dir / "images"
    output_file = project_root / "temp_output" / "full_demo" / "step2_coco_annotations.json"

    if output_file.exists():
        output_file.unlink()

    cmd = [
        "convert", "yolo2coco",
        str(image_dir),     # IMAGE_DIR (positional)
        str(yolo_dir / "labels"),  # LABEL_DIR (positional)
        str(class_file),    # CLASS_FILE (positional)
        str(output_file),   # OUTPUT_FILE (positional)
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        # Verify output file
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                image_count = len(data.get("images", []))
                annotation_count = len(data.get("annotations", []))
                category_count = len(data.get("categories", []))
                print(f"✓ Step 2 completed: Conversion successful, contains {image_count} images, {annotation_count} annotations, {category_count} categories")
        return True, output_file
    else:
        print("✗ Step 2 failed")
        return False, None

def step3_visualize_coco(coco_file):
    """Step 3: Visualize COCO labels"""
    print("\n" + "="*60)
    print("Step 3: Visualize COCO labels")
    print("="*60)

    image_dir = project_root / "assets" / "test_data" / "seg" / "coco" / "images"
    output_dir = project_root / "temp_output" / "full_demo" / "step3_coco_visualization"

    cleanup_directory(output_dir)

    cmd = [
        "visualize", "coco",
        str(image_dir),  # image_dir (positional)
        str(coco_file),  # coco_file (positional)
        "--save",
        str(output_dir),  # output directory for --save
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        image_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        print(f"✓ Step 3 completed: Generated {len(image_files)} COCO visualization images")
        return True, output_dir
    else:
        print("✗ Step 3 failed")
        return False, None

def step4_coco_to_yolo(coco_file):
    """Step 4: COCO to YOLO format conversion"""
    print("\n" + "="*60)
    print("Step 4: COCO to YOLO format conversion")
    print("="*60)

    image_dir = project_root / "assets" / "test_data" / "seg" / "coco" / "images"
    output_dir = project_root / "temp_output" / "full_demo" / "step4_yolo_converted"

    cleanup_directory(output_dir)

    cmd = [
        "convert", "coco2yolo",
        str(coco_file),    # COCO_FILE (positional)
        str(output_dir),   # OUTPUT_DIR (positional)
        "--verbose"
    ]

    success = run_cli_command(cmd)
    if success:
        # YOLO label files may be in the labels subdirectory
        labels_dir = output_dir / "labels"
        if labels_dir.exists():
            txt_files = list(labels_dir.rglob("*.txt"))
        else:
            txt_files = list(output_dir.rglob("*.txt"))
        # Filter out classes.txt file (not an annotation file)
        label_files = [f for f in txt_files if f.name != "classes.txt"]
        print(f"✓ Step 4 completed: Generated {len(label_files)} YOLO annotation files")
        return True, output_dir
    else:
        print("✗ Step 4 failed")
        return False, None

def step5_compare_yolo_dirs(original_dir, converted_dir):
    """Step 5: Compare original YOLO and converted YOLO directories"""
    print("\n" + "="*60)
    print("Step 5: Verify conversion accuracy")
    print("="*60)

    # YOLO label files may be in the labels subdirectory
    original_labels_dir = original_dir
    converted_labels_dir = converted_dir / "labels" if (converted_dir / "labels").exists() else converted_dir

    # Find all .txt files (recursive)
    original_files = list(original_labels_dir.rglob("*.txt"))
    converted_files = list(converted_labels_dir.rglob("*.txt"))

    # Filter out classes.txt file (not an annotation file)
    original_label_files = [f for f in original_files if f.name != "classes.txt"]
    converted_label_files = [f for f in converted_files if f.name != "classes.txt"]

    print(f"Original YOLO directory: {len(original_label_files)} annotation files")
    print(f"Converted YOLO directory: {len(converted_label_files)} annotation files")

    if len(original_label_files) == len(converted_label_files):
        print("✓ File count matches, conversion is basically accurate")
        # Can add more detailed comparison, such as content comparison
        return True
    else:
        print("✗ File count does not match, conversion may have issues")
        return False

def main():
    """Main function"""
    print("DataFlow-CV CLI complete workflow demonstration")
    print("="*60)
    print("This demonstration will show a complete label processing workflow:")
    print("1. Visualize original YOLO labels")
    print("2. Convert YOLO labels to COCO format")
    print("3. Visualize converted COCO labels")
    print("4. Convert COCO labels back to YOLO format")
    print("5. Verify conversion accuracy")
    print("="*60)

    # Check if test data exists
    test_data_root = project_root / "assets" / "test_data" / "seg"
    if not test_data_root.exists():
        print(f"Error: Test data directory does not exist: {test_data_root}")
        print("Please ensure running this script in the project root directory")
        return 1

    # Create temporary output directory
    temp_root = project_root / "temp_output" / "full_demo"
    cleanup_directory(temp_root)

    all_success = True
    results = {}

    # Step 1: Visualize original YOLO labels
    success1, yolo_viz_dir = step1_visualize_yolo()
    all_success = all_success and success1
    results["step1"] = success1

    # Step 2: YOLO to COCO format conversion
    success2, coco_file = step2_yolo_to_coco()
    all_success = all_success and success2
    results["step2"] = success2

    # Step 3: Visualize COCO labels
    if success2:
        success3, coco_viz_dir = step3_visualize_coco(coco_file)
        all_success = all_success and success3
        results["step3"] = success3
    else:
        results["step3"] = False

    # Step 4: COCO to YOLO format conversion
    if success2:
        success4, yolo_converted_dir = step4_coco_to_yolo(coco_file)
        all_success = all_success and success4
        results["step4"] = success4
    else:
        results["step4"] = False

    # Step 5: Compare original YOLO and converted YOLO
    if success1 and success4:
        original_yolo_dir = project_root / "assets" / "test_data" / "seg" / "yolo" / "labels"
        success5 = step5_compare_yolo_dirs(original_yolo_dir, yolo_converted_dir)
        all_success = all_success and success5
        results["step5"] = success5
    else:
        results["step5"] = False

    # Summary
    print("\n" + "="*60)
    print("Workflow demonstration summary")
    print("="*60)
    print("Step completion status:")
    for i, (step, success) in enumerate(results.items(), 1):
        status = "✓" if success else "✗"
        step_name = {
            "step1": "Visualize original YOLO labels",
            "step2": "YOLO to COCO format conversion",
            "step3": "Visualize COCO labels",
            "step4": "COCO to YOLO format conversion",
            "step5": "Verify conversion accuracy",
        }.get(step, step)
        print(f"  Step {i}: {step_name} {status}")

    if all_success:
        print("\n✓ Complete workflow demonstration successfully completed!")
        print(f"All results saved at: {project_root / 'temp_output' / 'full_demo'}")
        print("\nYou have successfully experienced the core functionality of DataFlow-CV CLI:")
        print("  - Visualize YOLO and COCO format labels")
        print("  - Perform bidirectional conversion between YOLO and COCO formats")
        print("  - Verify conversion accuracy")
        return 0
    else:
        print("\n✗ Workflow demonstration partially failed")
        print("Please check error messages, or try running each step individually")
        return 1

if __name__ == "__main__":
    sys.exit(main())