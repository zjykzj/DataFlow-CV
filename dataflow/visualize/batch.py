"""
Batch visualization utilities for DataFlow.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from ..config import get_config


def find_matching_pairs(
    image_dir: str,
    annotation_dir: str,
    annotation_ext: str
) -> List[Tuple[str, str]]:
    """
    Find matching image-annotation file pairs between directories.

    Args:
        image_dir: Directory containing image files
        annotation_dir: Directory containing annotation files
        annotation_ext: Expected annotation file extension (e.g., '.json', '.txt')

    Returns:
        List of (image_path, annotation_path) tuples
    """
    config = get_config()
    image_extensions = config["conversion"]["image_extensions"]

    image_dir_path = Path(image_dir)
    annotation_dir_path = Path(annotation_dir)

    # Build mapping of basename to image path
    image_map = {}
    for ext in image_extensions:
        for img_path in image_dir_path.glob(f"*{ext}"):
            if img_path.is_file():
                image_map[img_path.stem] = str(img_path)
        # Also check uppercase extensions
        for img_path in image_dir_path.glob(f"*{ext.upper()}"):
            if img_path.is_file():
                image_map[img_path.stem] = str(img_path)

    # Find matching annotations
    pairs = []
    for ann_path in annotation_dir_path.glob(f"*{annotation_ext}"):
        if ann_path.is_file():
            basename = ann_path.stem
            if basename in image_map:
                pairs.append((image_map[basename], str(ann_path)))

    # Also check uppercase extension for annotations
    for ann_path in annotation_dir_path.glob(f"*{annotation_ext.upper()}"):
        if ann_path.is_file():
            basename = ann_path.stem
            if basename in image_map and (image_map[basename], str(ann_path)) not in pairs:
                pairs.append((image_map[basename], str(ann_path)))

    return sorted(pairs)  # Sort for consistent ordering


def validate_batch_directories(image_dir: str, annotation_dir: str) -> None:
    """
    Validate that directories exist and contain files.

    Args:
        image_dir: Image directory path
        annotation_dir: Annotation directory path

    Raises:
        FileNotFoundError: If directories don't exist
        ValueError: If directories are empty or contain no relevant files
    """
    config = get_config()
    image_extensions = config["conversion"]["image_extensions"]

    image_dir_path = Path(image_dir)
    annotation_dir_path = Path(annotation_dir)

    if not image_dir_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not annotation_dir_path.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    # Check if directories contain any files (not just empty)
    if not any(image_dir_path.iterdir()):
        raise ValueError(f"Image directory is empty: {image_dir}")
    if not any(annotation_dir_path.iterdir()):
        raise ValueError(f"Annotation directory is empty: {annotation_dir}")

    # Check for at least one image file with supported extension
    has_images = False
    for ext in image_extensions:
        if any(image_dir_path.glob(f"*{ext}")):
            has_images = True
            break
        if any(image_dir_path.glob(f"*{ext.upper()}")):
            has_images = True
            break

    if not has_images:
        raise ValueError(f"No supported image files found in {image_dir}. Supported extensions: {image_extensions}")


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


def batch_process_images(
    pairs: List[Tuple[str, str]],
    visualize_func,
    format_name: str,
    save_dir: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> None:
    """
    Process a batch of image-annotation pairs.

    Args:
        pairs: List of (image_path, annotation_path) tuples
        visualize_func: Function to visualize a single pair
        format_name: Name of the format (for display)
        save_dir: Directory to save visualizations (optional)
        show: Whether to show images interactively
        **kwargs: Additional arguments passed to visualize_func
    """
    if not pairs:
        print("No matching image-annotation pairs found.")
        return

    print(f"Found {len(pairs)} image-annotation pairs.")

    current_idx = 0
    while current_idx < len(pairs):
        img_path, ann_path = pairs[current_idx]
        progress = get_batch_progress(current_idx, len(pairs))

        print(f"{progress} Processing: {Path(img_path).name} ↔ {Path(ann_path).name}")

        try:
            result = visualize_func(img_path, ann_path, **kwargs)

            if save_dir:
                output_dir = Path(save_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{Path(img_path).stem}_vis.jpg"

                import cv2
                success = cv2.imwrite(str(output_path), result)
                if success:
                    print(f"  Saved to: {output_path}")
                else:
                    print(f"  Warning: Failed to save {output_path}")

            if show:
                from .base import BaseVisualizer
                key = BaseVisualizer.show_batch_navigation(
                    result,
                    f"{format_name} Visualization",
                    current_idx,
                    len(pairs)
                )

                if key == 'q':
                    print("Batch visualization stopped by user.")
                    break
                elif key == 'left' and current_idx > 0:
                    current_idx -= 1
                    continue
                elif key == 'right':
                    current_idx += 1
                    continue
                else:
                    # Stay on current image
                    continue
            else:
                # Non-interactive mode, move to next
                current_idx += 1

        except Exception as e:
            print(f"  Error processing {Path(img_path).name}: {e}")
            print("  Skipping to next file...")
            current_idx += 1