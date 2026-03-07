"""
Batch conversion utilities for DataFlow.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable, Any
import sys

from ..config import get_config
from .base import BaseConverter


def get_batch_progress(current_idx: int, total: int) -> str:
    """
    Format progress string for batch processing.

    Args:
        current_idx: Current index (0-based)
        total: Total number of items

    Returns:
        Formatted progress string
    """
    return f"[{current_idx + 1}/{total}]"


def batch_process_conversion(
    pairs: List[Tuple[str, str]],
    convert_func: Callable,
    output_path: str,
    needs_image: bool = True,
    **kwargs
) -> None:
    """
    Process a batch of file pairs for conversion.

    Args:
        pairs: List of (input_path, annotation_path) tuples
               For conversions without images, both paths are the annotation file
        convert_func: Function to convert a single pair
        output_path: Output path (file or directory)
        needs_image: Whether the conversion function needs an image path
        **kwargs: Additional arguments passed to convert_func
    """
    if not pairs:
        print("No matching file pairs found.")
        return

    print(f"Found {len(pairs)} file pairs.")

    output_path_obj = Path(output_path)
    is_output_dir = False

    # Determine if output is a directory or file
    if output_path_obj.exists():
        is_output_dir = output_path_obj.is_dir()
    else:
        # Check if it looks like a directory (no extension or ends with slash)
        output_str = str(output_path)
        has_extension = output_path_obj.suffix in {'.json', '.txt', '.yaml', '.yml', '.xml'}
        ends_with_sep = output_str.endswith('/') or output_str.endswith('\\')

        if ends_with_sep or not has_extension:
            is_output_dir = True
            output_path_obj.mkdir(parents=True, exist_ok=True)

    successful = 0
    errors = 0

    for idx, (input_path, ann_path) in enumerate(pairs):
        progress = get_batch_progress(idx, len(pairs))
        input_name = Path(input_path).name
        ann_name = Path(ann_path).name

        try:
            # Determine output file path
            if is_output_dir:
                # Generate output filename based on input name
                input_stem = Path(input_path).stem

                # Determine appropriate extension based on conversion function name
                # This is a simple heuristic - can be improved
                func_name = convert_func.__name__.lower()
                if 'coco' in func_name:
                    output_file = output_path_obj / f"{input_stem}.json"
                elif 'yolo' in func_name:
                    output_file = output_path_obj / f"{input_stem}.txt"
                elif 'labelme' in func_name:
                    output_file = output_path_obj / f"{input_stem}.json"
                else:
                    # Default to .json
                    output_file = output_path_obj / f"{input_stem}.json"
            else:
                # Single file output (combined mode)
                output_file = output_path_obj
                # Only process first pair in combined mode
                if idx > 0:
                    continue

            # Call conversion function with appropriate arguments
            if needs_image:
                convert_func(input_path, ann_path, output_file, **kwargs)
            else:
                # For conversions without images (labelme2coco, labelme2yolo)
                convert_func(ann_path, output_file, **kwargs)

            if is_output_dir:
                print(f"{progress} Converted: {input_name} → {output_file.name}")
            else:
                print(f"{progress} Processing: {ann_name}")

            successful += 1

        except Exception as e:
            print(f"{progress} Error processing {input_name}: {e}")
            print("  Skipping...")
            errors += 1
            continue

    print(f"\nBatch conversion complete.")
    print(f"  Successfully converted: {successful}")
    if errors > 0:
        print(f"  Errors: {errors}")


def batch_convert_with_combined_option(
    pairs: List[Tuple[str, str]],
    single_convert_func: Callable,
    batch_convert_func: Optional[Callable] = None,
    output_path: str = None,
    combined: bool = False,
    **kwargs
) -> None:
    """
    Batch conversion with support for combined output (for COCO format).

    Args:
        pairs: List of (input_path, annotation_path) tuples
        single_convert_func: Function to convert a single file pair
        batch_convert_func: Function for batch/combined conversion (optional)
        output_path: Output path (file or directory)
        combined: Whether to combine outputs into a single file
        **kwargs: Additional arguments passed to conversion functions
    """
    if not pairs:
        print("No matching file pairs found.")
        return

    print(f"Found {len(pairs)} file pairs.")

    output_path_obj = Path(output_path)

    if combined:
        # Combined mode: single output file
        if batch_convert_func:
            # Use batch conversion function if available
            batch_convert_func(pairs, output_path, **kwargs)
        else:
            # Fallback to single conversion for combined mode
            # This is only suitable for formats that support merging (like COCO)
            print("Note: Using single conversion for combined mode (may not merge properly)")
            # Process first pair only
            if pairs:
                input_path, ann_path = pairs[0]
                single_convert_func(input_path, ann_path, output_path, **kwargs)
                print(f"Converted first file to: {output_path}")
    else:
        # Per-file mode: multiple output files
        # Ensure output is a directory
        if output_path_obj.exists() and not output_path_obj.is_dir():
            raise ValueError(f"Output path exists but is not a directory: {output_path}")

        # Create output directory if it doesn't exist
        output_path_obj.mkdir(parents=True, exist_ok=True)

        successful = 0
        errors = 0

        for idx, (input_path, ann_path) in enumerate(pairs):
            progress = get_batch_progress(idx, len(pairs))
            input_name = Path(input_path).name

            try:
                # Generate output filename based on input name
                input_stem = Path(input_path).stem

                # Determine appropriate extension
                func_name = single_convert_func.__name__.lower()
                if 'coco' in func_name:
                    output_file = output_path_obj / f"{input_stem}.json"
                elif 'yolo' in func_name:
                    output_file = output_path_obj / f"{input_stem}.txt"
                elif 'labelme' in func_name:
                    output_file = output_path_obj / f"{input_stem}.json"
                else:
                    output_file = output_path_obj / f"{input_stem}.json"

                single_convert_func(input_path, ann_path, output_file, **kwargs)
                print(f"{progress} Converted: {input_name} → {output_file.name}")
                successful += 1

            except Exception as e:
                print(f"{progress} Error processing {input_name}: {e}")
                print("  Skipping...")
                errors += 1
                continue

        print(f"\nBatch conversion complete.")
        print(f"  Successfully converted: {successful}")
        if errors > 0:
            print(f"  Errors: {errors}")


def find_matching_conversion_pairs(
    input_dir: str,
    annotation_dir: str,
    annotation_ext: str,
    needs_input: bool = True
) -> List[Tuple[str, str]]:
    """
    Find matching file pairs for conversion.

    Args:
        input_dir: Directory containing input files (images or annotations)
        annotation_dir: Directory containing annotation files
        annotation_ext: Expected annotation file extension (e.g., '.json', '.txt')
        needs_input: Whether input files are needed (False for labelme2coco, labelme2yolo)

    Returns:
        List of (input_path, annotation_path) tuples
    """
    return BaseConverter.find_matching_pairs_for_conversion(
        input_dir, annotation_dir, annotation_ext
    )


def validate_conversion_directories(
    input_dir: str,
    annotation_dir: str,
    needs_input: bool = True
) -> None:
    """
    Validate directories for batch conversion.

    Args:
        input_dir: Input directory path
        annotation_dir: Annotation directory path
        needs_input: Whether input directory is required

    Raises:
        FileNotFoundError: If directories don't exist
        ValueError: If directories are empty or contain no relevant files
    """
    BaseConverter.validate_conversion_directories(input_dir, annotation_dir, needs_input)