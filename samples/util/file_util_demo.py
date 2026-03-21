#!/usr/bin/env python3
"""
FileOperations usage demo.

Demonstrates how to use the FileOperations class for common file operations.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.util.file_util import FileOperations
from dataflow.util.logging_util import LoggingOperations


def main():
    """Main demonstration function."""
    # Setup logging
    log_ops = LoggingOperations()
    logger = log_ops.get_logger("file_util_demo", level="INFO")

    logger.info("=" * 50)
    logger.info("FileOperations Demonstration")
    logger.info("=" * 50)

    # Create FileOperations instance
    file_ops = FileOperations(logger=logger)

    # 1. ensure_dir - Create directory structure
    logger.info("\n1. Directory Management")
    logger.info("-" * 30)

    temp_base = Path(tempfile.mkdtemp(prefix="demo_"))
    logger.info(f"Created temporary base directory: {temp_base}")

    nested_dir = temp_base / "level1" / "level2" / "level3"
    created = file_ops.ensure_dir(nested_dir)
    logger.info(f"Created nested directory: {created}")
    logger.info(f"Directory exists: {created.exists()}")

    # 2. copy_files - Copy files with pattern matching
    logger.info("\n2. File Copying with Pattern Matching")
    logger.info("-" * 30)

    src_dir = temp_base / "source"
    src_dir.mkdir()

    # Create some test files
    files_to_create = ["document.txt", "image.jpg", "data.json", "backup.bak"]
    for filename in files_to_create:
        file_path = src_dir / filename
        file_path.write_text(f"Content of {filename}")

    dst_dir = temp_base / "destination"

    # Copy only text files
    results = file_ops.copy_files(str(src_dir / "*.txt"), dst_dir)
    logger.info(f"Copied {len(results)} text files")

    # Copy all files with overwrite
    results = file_ops.copy_files(str(src_dir / "*"), dst_dir, overwrite=True)
    logger.info(f"Copied {len([r for r in results if r[1]])} files total")

    # 3. find_files - Search for files
    logger.info("\n3. File Search")
    logger.info("-" * 30)

    # Create subdirectory with more files
    subdir = src_dir / "subfolder"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("Nested content")

    # Non-recursive search
    files = file_ops.find_files(src_dir, "*.txt", recursive=False)
    logger.info(f"Non-recursive search found {len(files)} text files")

    # Recursive search
    files = file_ops.find_files(src_dir, "*.txt", recursive=True)
    logger.info(f"Recursive search found {len(files)} text files")

    # 4. read_lines and write_lines
    logger.info("\n4. Reading and Writing Lines")
    logger.info("-" * 30)

    lines_file = temp_base / "lines.txt"
    sample_lines = [
        "First line of text",
        "Second line with some data",
        "Third line for demonstration",
        "Fourth line to show batch writing"
    ]

    success = file_ops.write_lines(lines_file, sample_lines)
    logger.info(f"Write lines success: {success}")
    logger.info(f"File size: {file_ops.get_file_size(lines_file)} bytes")

    read_lines = file_ops.read_lines(lines_file)
    logger.info(f"Read {len(read_lines)} lines")
    logger.info(f"First line: {read_lines[0]}")

    # 5. safe_remove - Safe file deletion
    logger.info("\n5. Safe File Removal")
    logger.info("-" * 30)

    file_to_remove = temp_base / "to_delete.txt"
    file_to_remove.write_text("This file will be deleted")

    logger.info(f"File exists before removal: {file_to_remove.exists()}")
    removed = file_ops.safe_remove(file_to_remove)
    logger.info(f"Removal successful: {removed}")
    logger.info(f"File exists after removal: {file_to_remove.exists()}")

    # 6. create_temp_dir - Temporary directory creation
    logger.info("\n6. Temporary Directory Creation")
    logger.info("-" * 30)

    temp_dir = file_ops.create_temp_dir(prefix="demo_temp_")
    logger.info(f"Created temporary directory: {temp_dir}")
    logger.info(f"Directory exists: {temp_dir.exists()}")

    # Clean up temporary directory
    file_ops.safe_remove(temp_dir)
    logger.info(f"Cleaned up temporary directory")

    # 7. Error handling demonstration
    logger.info("\n7. Error Handling")
    logger.info("-" * 30)

    non_existent = temp_base / "ghost" / "nonexistent.txt"
    try:
        size = file_ops.get_file_size(non_existent)
    except FileNotFoundError as e:
        logger.info(f"Expected FileNotFoundError: {e}")

    # Clean up all temporary files
    logger.info("\n" + "=" * 50)
    logger.info("Cleaning up all temporary files...")
    file_ops.safe_remove(temp_base)
    logger.info(f"Base directory removed: {not temp_base.exists()}")

    logger.info("\nDemo completed successfully!")


if __name__ == "__main__":
    main()