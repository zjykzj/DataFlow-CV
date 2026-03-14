#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : {YEAR}/{MONTH}/{DAY}
@File    : template.py
@Author  : DataFlow Team
@Description: Template for creating DataFlow-CV examples

This template provides a structure for creating consistent examples
for DataFlow-CV functionality. Use this as a starting point for both
API and CLI examples.
"""

import os
import sys
import json
import tempfile
import shutil
import stat
from pathlib import Path

# Add parent directory to path to import dataflow
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import DataFlow-CV
import dataflow
from dataflow.config import Config


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def create_test_paths():
    """创建跨平台的测试路径"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="dataflow_test_")

    # 不存在的路径（用于测试无效路径错误）
    nonexistent_path = os.path.join(temp_dir, "nonexistent_subdir", "file.txt")

    # 尝试创建只读目录（在Windows上可能失败）
    read_only_dir = os.path.join(temp_dir, "readonly_dir")
    os.makedirs(read_only_dir, exist_ok=True)

    try:
        # 尝试设置为只读
        os.chmod(read_only_dir, stat.S_IRUSR | stat.S_IXUSR)
        read_only_path = os.path.join(read_only_dir, "no_permission.txt")
    except (OSError, PermissionError):
        # Windows上无法设置只读目录，使用普通路径
        read_only_path = os.path.join(read_only_dir, "no_permission.txt")

    # 临时文件路径
    temp_file_path = os.path.join(temp_dir, "temp_file.txt")

    return {
        "temp_dir": temp_dir,
        "nonexistent_path": nonexistent_path,
        "read_only_path": read_only_path,
        "temp_file_path": temp_file_path
    }


# ============================================================================
# API EXAMPLE TEMPLATE
# ============================================================================

def create_sample_data_api():
    """
    Template for creating sample data for API examples.

    Returns:
        tuple: (temp_dir, input_path) - temporary directory and input file path
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="dataflow_api_demo_")
    print(f"Created temporary directory: {temp_dir}")

    # Create sample input data
    input_path = os.path.join(temp_dir, "input.json")

    # TODO: Replace with actual sample data creation
    sample_data = {
        "info": {
            "description": "Sample dataset for API demonstration",
            "version": "1.0",
            "year": 2026,
            "contributor": "DataFlow-CV",
            "date_created": "2026-03-10"
        },
        # Add your sample data structure here
    }

    with open(input_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)

    print(f"Created sample input file: {input_path}")
    return temp_dir, input_path


def demo_convenience_function(input_path, output_dir):
    """
    Template for demonstrating convenience functions.

    Example: dataflow.coco_to_yolo(), dataflow.visualize_yolo(), etc.
    """
    print_header("USING CONVENIENCE FUNCTION")

    print(f"\nCode:")
    print(f"  import dataflow")
    print(f"  result = dataflow.function_name('{input_path}', '{output_dir}')")

    print(f"\nExecuting...")
    try:
        # TODO: Replace with actual convenience function call
        # result = dataflow.function_name(input_path, output_dir)
        result = {"images_processed": 3, "annotations_processed": 5}

        print(f"\n✅ Success!")
        print(f"\nResult keys: {list(result.keys())}")

        # Show important statistics
        print(f"\nOperation statistics:")
        for key, value in result.items():
            print(f"  - {key}: {value}")

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def demo_converter_class(input_path, output_dir):
    """
    Template for demonstrating converter/visualizer classes directly.
    """
    print_header("USING CONVERTER/VISUALIZER CLASS")

    print(f"\nCode:")
    print(f"  from dataflow.module import ConverterClass")
    print(f"  converter = ConverterClass(verbose=True)")
    print(f"  result = converter.convert('{input_path}', '{output_dir}')")

    print(f"\nExecuting...")
    try:
        # TODO: Replace with actual converter class instantiation and method call
        # converter = ConverterClass(verbose=True)
        # result = converter.convert(input_path, output_dir)
        result = {"images_processed": 3, "annotations_processed": 5}

        print(f"\n✅ Success!")
        print(f"\nResult type: {type(result)}")

        # Show additional information available through converter
        print(f"\nConverter capabilities:")
        print(f"  - Has logger: True")
        print(f"  - Can validate paths: True")
        print(f"  - Can ensure directories: True")

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def demo_advanced_features(input_path, output_dir):
    """
    Template for demonstrating advanced features and configuration.
    """
    print_header("ADVANCED FEATURES")

    # Show current configuration
    print(f"\nCurrent configuration:")
    config_items = [
        ("VERBOSE", Config.VERBOSE),
        ("OVERWRITE_EXISTING", Config.OVERWRITE_EXISTING),
        # Add more config items as needed
    ]
    for key, value in config_items:
        print(f"  {key}: {value}")

    # Demonstrate custom configuration
    print(f"\nCustom configuration example:")
    print(f"  # Save original values")
    print(f"  original_verbose = Config.VERBOSE")
    print(f"  ")
    print(f"  # Configure for batch processing")
    print(f"  Config.VERBOSE = False")
    print(f"  Config.OVERWRITE_EXISTING = True")
    print(f"  ")
    print(f"  # Create converter with custom settings")
    print(f"  converter = ConverterClass(verbose=False)")
    print(f"  ")
    print(f"  # Restore configuration")
    print(f"  Config.VERBOSE = original_verbose")

    # Actually demonstrate with a different output directory
    custom_output = output_dir + "_custom"
    print(f"\nExecuting with custom configuration...")
    try:
        # Save original
        original_verbose = Config.VERBOSE
        original_overwrite = Config.OVERWRITE_EXISTING

        # Configure
        Config.VERBOSE = False
        Config.OVERWRITE_EXISTING = True

        # TODO: Create converter and convert
        # converter = ConverterClass(verbose=False)
        # result = converter.convert(input_path, custom_output)
        result = {"images_processed": 3}

        print(f"\n✅ Custom operation successful!")
        print(f"  Output directory: {custom_output}")

        # Restore
        Config.VERBOSE = original_verbose
        Config.OVERWRITE_EXISTING = original_overwrite

        return result
    except Exception as e:
        print(f"\n❌ Error: {e}")
        # Restore anyway
        Config.VERBOSE = original_verbose
        Config.OVERWRITE_EXISTING = original_overwrite
        return None


def demo_error_handling():
    """
    Template for demonstrating error handling.
    """
    print_header("ERROR HANDLING")

    # 创建测试路径
    test_paths = create_test_paths()

    try:
        print(f"\n1. Invalid input path:")
        try:
            # TODO: Replace with actual function call that should fail
            # result = dataflow.function_name(test_paths["nonexistent_path"], "/tmp/output")
            print(f"   ✅ Would raise: ValueError('Input path does not exist')")
        except ValueError as e:
            print(f"   ✅ Caught expected error: {str(e)[:50]}...")

        print(f"\n2. Invalid output directory:")
        try:
            # TODO: Replace with actual function call that should fail
            # result = dataflow.function_name(test_paths["temp_file_path"], test_paths["read_only_path"])
            print(f"   ✅ Would raise: PermissionError or ValueError")
        except (ValueError, PermissionError) as e:
            print(f"   ✅ Caught expected error: {str(e)[:50]}...")

        print(f"\n3. Malformed input data:")
        temp_dir = tempfile.mkdtemp(prefix="error_demo_")
        try:
            bad_file = os.path.join(temp_dir, "bad.json")
            with open(bad_file, 'w', encoding='utf-8') as f:
                f.write("{ invalid json }")

            try:
                # TODO: Replace with actual function call that should fail
                # result = dataflow.function_name(bad_file, "/tmp/output")
                print(f"   ✅ Would raise: json.JSONDecodeError")
            except (ValueError, json.JSONDecodeError) as e:
                print(f"   ✅ Caught expected error: {str(e)[:50]}...")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    finally:
        # 清理临时文件
        shutil.rmtree(test_paths["temp_dir"], ignore_errors=True)


def inspect_output(output_dir):
    """
    Template for inspecting generated output files.
    """
    print_header("INSPECTING OUTPUT")

    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return

    print(f"\nOutput directory: {output_dir}")

    # List directory structure
    print(f"\nDirectory structure:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')

    # Check for specific expected files
    # TODO: Customize based on expected output
    expected_files = [
        ("class.names", "Classes file"),
        ("labels/", "Labels directory"),
    ]

    for filename, description in expected_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"\n✓ {description}: {filepath}")

            # Example: Read and show class names
            if filename.endswith(".names"):
                with open(filepath, 'r', encoding='utf-8') as f:
                    classes = [line.strip() for line in f if line.strip()]
                print(f"  Classes: {classes}")
        else:
            print(f"\n✗ {description} not found!")


def api_example_main():
    """
    Main function for API examples.

    This demonstrates the typical flow of an API example:
    1. Create sample data
    2. Demonstrate convenience function
    3. Demonstrate converter class
    4. Demonstrate advanced features
    5. Demonstrate error handling
    6. Inspect output
    7. Clean up
    """
    print_header("DATAFLOW-CV API EXAMPLE TEMPLATE")

    # Create sample data
    temp_dir, input_path = create_sample_data_api()

    try:
        # Demo 1: Convenience function
        output_dir1 = os.path.join(temp_dir, "output1")
        result1 = demo_convenience_function(input_path, output_dir1)

        # Demo 2: Converter class
        output_dir2 = os.path.join(temp_dir, "output2")
        result2 = demo_converter_class(input_path, output_dir2)

        # Demo 3: Advanced features
        demo_advanced_features(input_path, os.path.join(temp_dir, "output3"))

        # Demo 4: Error handling
        demo_error_handling()

        # Inspect output
        if result1:
            inspect_output(output_dir1)

        print_header("SUMMARY")
        print(f"\n✅ API demonstration completed!")
        print(f"\n📁 Sample data directory: {temp_dir}")
        print(f"   - Input file: {input_path}")
        print(f"   - Output 1: {output_dir1}")
        print(f"   - Output 2: {output_dir2}")

        print(f"\n💡 Key takeaways:")
        print(f"   1. Use dataflow.function_name() for simple operations")
        print(f"   2. Use ConverterClass for more control")
        print(f"   3. Configure behavior via dataflow.Config")
        print(f"   4. All methods return detailed statistics")
        print(f"   5. Error handling is built into the converters")

    finally:
        # Cleanup
        cleanup = input("\nClean up temporary files? (y/n): ").strip().lower()
        if cleanup == 'y':
            shutil.rmtree(temp_dir)
            print("✅ Temporary files cleaned up.")
        else:
            print(f"⚠️  Temporary files preserved at: {temp_dir}")
            print(f"   You may want to clean up manually: {temp_dir}")


# ============================================================================
# CLI EXAMPLE TEMPLATE
# ============================================================================

def create_sample_data_cli():
    """
    Template for creating sample data for CLI examples.

    Returns:
        tuple: (temp_dir, input_path) - temporary directory and input file path
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="dataflow_cli_demo_")
    print(f"Created temporary directory: {temp_dir}")

    # Create sample input data
    input_path = os.path.join(temp_dir, "input.json")

    # TODO: Replace with actual sample data creation
    sample_data = {
        "info": {
            "description": "Sample dataset for CLI demonstration",
            "version": "1.0",
            "year": 2026,
            "contributor": "DataFlow-CV",
            "date_created": "2026-03-10"
        },
        # Add your sample data structure here
    }

    with open(input_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)

    print(f"Created sample input file: {input_path}")
    return temp_dir, input_path


def show_cli_commands(input_path, output_dir):
    """
    Template for showing available CLI commands.
    """
    print_header("CLI COMMANDS")

    print("\nBasic command:")
    print(f"  $ dataflow task subtask {input_path} {output_dir}")

    print("\nWith verbose output:")
    print(f"  $ dataflow task subtask --verbose {input_path} {output_dir}")
    print(f"  $ dataflow task subtask -v {input_path} {output_dir}")

    print("\nWith overwrite mode:")
    print(f"  $ dataflow task subtask --overwrite {input_path} {output_dir}")

    print("\nWith segmentation mode:")
    print(f"  $ dataflow task subtask --segmentation {input_path} {output_dir}")

    print("\nWith all options:")
    print(f"  $ dataflow task subtask -v --overwrite --segmentation {input_path} {output_dir}")

    print("\nGet help:")
    print(f"  $ dataflow task subtask --help")


def run_cli_command(input_path, output_dir, verbose=True, overwrite=False, segmentation=False):
    """
    Template for running CLI commands via subprocess.
    """
    print_header("RUNNING CLI COMMAND")

    import subprocess

    # Build command
    cmd = ["python", "-m", "dataflow.cli", "task", "subtask"]
    if verbose:
        cmd.append("--verbose")
    if overwrite:
        cmd.append("--overwrite")
    if segmentation:
        cmd.append("--segmentation")
    cmd.extend([input_path, output_dir])

    print(f"Command: {' '.join(cmd)}")
    print("\n" + "-"*40)

    # Run command
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        print("✅ Command successful!")
        print("\nOutput:")
        print(result.stdout)

        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)

        return True

    except subprocess.CalledProcessError as e:
        print("❌ Command failed!")
        print(f"\nError output:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def inspect_cli_output(output_dir):
    """
    Template for inspecting CLI-generated output files.
    """
    print_header("INSPECTING CLI OUTPUT")

    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return

    print(f"\nOutput directory: {output_dir}")

    # List directory structure
    print(f"\nDirectory structure:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')


def cli_example_main():
    """
    Main function for CLI examples.

    This demonstrates the typical flow of a CLI example:
    1. Create sample data
    2. Show available CLI commands
    3. Run basic conversion
    4. Run conversion with options
    5. Inspect output
    6. Clean up
    """
    print_header("DATAFLOW-CV CLI EXAMPLE TEMPLATE")

    # Create sample data
    temp_dir, input_path = create_sample_data_cli()
    output_dir = os.path.join(temp_dir, "output")

    try:
        # Show CLI commands
        show_cli_commands(input_path, output_dir)

        # Run basic conversion
        success = run_cli_command(input_path, output_dir, verbose=True, overwrite=False)

        if success:
            # Inspect output
            inspect_cli_output(output_dir)

        # Demonstrate with options
        output_dir_with_options = os.path.join(temp_dir, "output_with_options")
        print_header("RUNNING WITH OPTIONS")
        run_cli_command(input_path, output_dir_with_options, verbose=True, overwrite=True, segmentation=True)

        print_header("SUMMARY")
        print(f"\n✅ CLI demonstration completed!")
        print(f"\n📁 Sample data directory: {temp_dir}")
        print(f"   - Input file: {input_path}")
        print(f"   - Output: {output_dir}")
        print(f"   - Output with options: {output_dir_with_options}")

        print(f"\n💡 Key points:")
        print(f"   1. Use --verbose for detailed progress information")
        print(f"   2. Use --overwrite to replace existing files")
        print(f"   3. Use --segmentation for polygon annotations")
        print(f"   4. Always check help: dataflow task subtask --help")

    finally:
        # Cleanup
        cleanup = input("\nClean up temporary files? (y/n): ").strip().lower()
        if cleanup == 'y':
            shutil.rmtree(temp_dir)
            print("✅ Temporary files cleaned up.")
        else:
            print(f"⚠️  Temporary files preserved at: {temp_dir}")
            print(f"   You may want to clean up manually: {temp_dir}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    This template can be used to create both API and CLI examples.

    To create an API example:
    1. Fill in the TODO sections in create_sample_data_api()
    2. Implement the actual function calls in demo_convenience_function(),
       demo_converter_class(), etc.
    3. Customize inspect_output() for your specific output format
    4. Update the main function (api_example_main()) with your specific demos

    To create a CLI example:
    1. Fill in the TODO sections in create_sample_data_cli()
    2. Update show_cli_commands() with your actual CLI command
    3. Update run_cli_command() with your actual command structure
    4. Customize inspect_cli_output() for your specific output format
    5. Update the main function (cli_example_main()) as needed

    General tips:
    - Keep the structure consistent across examples
    - Include error handling demonstrations
    - Show both simple (convenience function) and advanced (class-based) usage
    - Demonstrate configuration options
    - Clean up temporary files
    """

    # Example: Run API template demonstration
    # api_example_main()

    # Example: Run CLI template demonstration
    # cli_example_main()

    print_header("DATAFLOW-CV EXAMPLE TEMPLATE")
    print("\nThis is a template file for creating consistent examples.")
    print("\nTo use this template:")
    print("1. Copy this file to a new example file")
    print("2. Fill in the TODO sections with your specific functionality")
    print("3. Update function names and documentation")
    print("4. Remove unnecessary sections for your use case")
    print("5. Test your example thoroughly")
    print("\nSee the comments at the bottom of this file for detailed instructions.")