"""
Utility functions for the label module.
"""

import json
import hashlib
from pathlib import Path
from typing import Type, Optional, Dict, Any
import filecmp
import difflib

from .base import BaseAnnotationHandler
from .models import DatasetAnnotations, AnnotationFormat


def verify_lossless_roundtrip(input_path: str, output_path: str, handler_class: Type[BaseAnnotationHandler]) -> bool:
    """
    Verify that reading and writing produces identical output.

    Args:
        input_path: Path to input annotation file or directory
        output_path: Path to output annotation file or directory
        handler_class: Handler class to use for reading/writing

    Returns:
        bool: True if output is identical to input, False otherwise

    Note:
        This function creates temporary handler instances for verification.
        For directory-based formats (LabelMe, YOLO), it compares all files.
        For single-file formats (COCO), it compares JSON structure.
    """
    import tempfile
    import shutil

    # Create a temporary directory for the output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = Path(temp_dir) / "output"

        # Read input annotations
        handler = handler_class(input_path)
        read_result = handler.read()
        if not read_result.success:
            print(f"Failed to read input: {read_result.message}")
            return False

        dataset = read_result.data
        if not isinstance(dataset, DatasetAnnotations):
            print(f"Invalid dataset type: {type(dataset)}")
            return False

        # Write annotations to temporary location
        write_result = handler.write(dataset, str(temp_output))
        if not write_result.success:
            print(f"Failed to write output: {write_result.message}")
            return False

        # Compare input and output
        return _compare_annotation_files(input_path, str(temp_output), handler_class)


def _compare_annotation_files(input_path: str, output_path: str, handler_class: Type[BaseAnnotationHandler]) -> bool:
    """
    Compare two annotation files/directories.

    Args:
        input_path: Path to input annotation file or directory
        output_path: Path to output annotation file or directory
        handler_class: Handler class (used to determine format)

    Returns:
        bool: True if files are identical, False otherwise
    """
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)

    # Determine format based on handler class name
    handler_name = handler_class.__name__.lower()

    if 'labelme' in handler_name:
        return _compare_labelme_dirs(input_path_obj, output_path_obj)
    elif 'yolo' in handler_name:
        return _compare_yolo_dirs(input_path_obj, output_path_obj)
    elif 'coco' in handler_name:
        return _compare_coco_files(input_path_obj, output_path_obj)
    else:
        # Generic comparison
        return _generic_file_comparison(input_path_obj, output_path_obj)


def _compare_labelme_dirs(input_dir: Path, output_dir: Path) -> bool:
    """Compare LabelMe JSON directories."""
    if not input_dir.is_dir() or not output_dir.is_dir():
        print(f"One of the paths is not a directory: {input_dir}, {output_dir}")
        return False

    # Get all JSON files
    input_files = sorted(input_dir.glob("*.json"))
    output_files = sorted(output_dir.glob("*.json"))

    if len(input_files) != len(output_files):
        print(f"Different number of JSON files: input={len(input_files)}, output={len(output_files)}")
        return False

    all_match = True
    for in_file, out_file in zip(input_files, output_files):
        if not _compare_json_files(in_file, out_file, ignore_fields=["imageData"]):
            print(f"Files differ: {in_file.name} vs {out_file.name}")
            all_match = False
            # Show diff
            _show_json_diff(in_file, out_file, ignore_fields=["imageData"])

    return all_match


def _compare_yolo_dirs(input_dir: Path, output_dir: Path) -> bool:
    """Compare YOLO text file directories."""
    if not input_dir.is_dir() or not output_dir.is_dir():
        print(f"One of the paths is not a directory: {input_dir}, {output_dir}")
        return False

    # Get all text files
    input_files = sorted(input_dir.glob("*.txt"))
    output_files = sorted(output_dir.glob("*.txt"))

    if len(input_files) != len(output_files):
        print(f"Different number of text files: input={len(input_files)}, output={len(output_files)}")
        return False

    all_match = True
    for in_file, out_file in zip(input_files, output_files):
        if not _compare_text_files(in_file, out_file):
            print(f"Files differ: {in_file.name} vs {out_file.name}")
            all_match = False
            # Show diff
            _show_text_diff(in_file, out_file)

    return all_match


def _compare_coco_files(input_file: Path, output_file: Path) -> bool:
    """Compare COCO JSON files."""
    if not input_file.is_file() or not output_file.is_file():
        print(f"One of the paths is not a file: {input_file}, {output_file}")
        return False

    # Ignore fields that may be auto-generated or vary between runs
    ignore_fields = [
        "date_created", "date_captured",
        "description", "url", "version", "year", "contributor",
        "__coco_original_data__"
    ]
    return _compare_json_files(input_file, output_file, ignore_fields=ignore_fields)


def _compare_json_files(file1: Path, file2: Path, ignore_fields: Optional[list] = None) -> bool:
    """Compare two JSON files, optionally ignoring certain fields."""
    try:
        with open(file1, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        with open(file2, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)

        # Remove ignored fields
        if ignore_fields:
            data1 = _remove_fields(data1, ignore_fields)
            data2 = _remove_fields(data2, ignore_fields)

        return data1 == data2
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error comparing JSON files: {e}")
        return False


def _compare_text_files(file1: Path, file2: Path) -> bool:
    """Compare two text files line by line."""
    try:
        with open(file1, 'r', encoding='utf-8') as f1:
            lines1 = f1.readlines()
        with open(file2, 'r', encoding='utf-8') as f2:
            lines2 = f2.readlines()

        # Normalize line endings and strip whitespace
        lines1 = [line.rstrip() for line in lines1]
        lines2 = [line.rstrip() for line in lines2]

        return lines1 == lines2
    except IOError as e:
        print(f"Error comparing text files: {e}")
        return False


def _remove_fields(data: Any, fields: list) -> Any:
    """Recursively remove fields from nested data structures."""
    if isinstance(data, dict):
        # Remove fields from this dict
        result = {}
        for key, value in data.items():
            if key not in fields:
                result[key] = _remove_fields(value, fields)
        return result
    elif isinstance(data, list):
        # Process each item in list
        return [_remove_fields(item, fields) for item in data]
    else:
        # Primitive value
        return data


def _show_json_diff(file1: Path, file2: Path, ignore_fields: Optional[list] = None):
    """Show differences between two JSON files."""
    try:
        with open(file1, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        with open(file2, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)

        if ignore_fields:
            data1 = _remove_fields(data1, ignore_fields)
            data2 = _remove_fields(data2, ignore_fields)

        # Convert to pretty JSON for comparison
        json1 = json.dumps(data1, indent=2, sort_keys=True, ensure_ascii=False)
        json2 = json.dumps(data2, indent=2, sort_keys=True, ensure_ascii=False)

        lines1 = json1.splitlines()
        lines2 = json2.splitlines()

        diff = list(difflib.unified_diff(lines1, lines2, lineterm=''))
        if diff:
            print("\n".join(diff[:50]))  # Limit output
    except Exception as e:
        print(f"Error showing diff: {e}")


def _show_text_diff(file1: Path, file2: Path):
    """Show differences between two text files."""
    try:
        with open(file1, 'r', encoding='utf-8') as f1:
            lines1 = f1.readlines()
        with open(file2, 'r', encoding='utf-8') as f2:
            lines2 = f2.readlines()

        diff = list(difflib.unified_diff(lines1, lines2, lineterm=''))
        if diff:
            print("\n".join(diff[:50]))  # Limit output
    except Exception as e:
        print(f"Error showing diff: {e}")


def _generic_file_comparison(input_path: Path, output_path: Path) -> bool:
    """Generic file/directory comparison."""
    if input_path.is_file() and output_path.is_file():
        return filecmp.cmp(input_path, output_path, shallow=False)
    elif input_path.is_dir() and output_path.is_dir():
        comparison = filecmp.dircmp(input_path, output_path)
        return (not comparison.left_only and not comparison.right_only and
                not comparison.diff_files and not comparison.funny_files)
    else:
        print(f"Incompatible path types: {input_path} vs {output_path}")
        return False


def calculate_file_hash(file_path: Path, algorithm: str = 'md5') -> str:
    """Calculate hash of a file."""
    hash_func = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()