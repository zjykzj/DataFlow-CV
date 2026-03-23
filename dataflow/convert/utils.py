"""
Utility functions for format conversion.

Provides helper functions for category handling, path resolution, and conversion validation.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

from ..label.models import DatasetAnnotations


def extract_categories_from_annotations(annotations: DatasetAnnotations) -> Dict[int, str]:
    """Extract category mapping from DatasetAnnotations."""
    return annotations.categories.copy()


def generate_classes_file(categories: Dict[int, str], output_path: Path) -> bool:
    """
    Generate classes.txt file from category mapping.

    Args:
        categories: Dictionary mapping category IDs to names
        output_path: Path to output classes.txt file

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write categories sorted by ID
            for cat_id in sorted(categories.keys()):
                f.write(f"{categories[cat_id]}\n")
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to write classes file {output_path}: {e}")
        return False


def load_classes_file(class_file: Path) -> Dict[int, str]:
    """
    Load category mapping from classes.txt file.

    Args:
        class_file: Path to classes.txt file

    Returns:
        Dictionary mapping index (starting from 0) to category name
    """
    categories = {}
    try:
        with open(class_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:  # Skip empty lines
                    categories[i] = line
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load classes file {class_file}: {e}")
    return categories


def extract_categories_from_coco(coco_data: Dict) -> Dict[int, str]:
    """
    Extract category information from COCO data.

    Args:
        coco_data: COCO JSON data dictionary

    Returns:
        Dictionary mapping category IDs to names
    """
    categories = {}
    for cat in coco_data.get("categories", []):
        cat_id = cat.get("id")
        cat_name = cat.get("name", "")
        if cat_id is not None:
            categories[cat_id] = cat_name
    return categories


def ensure_categories_in_annotations(annotations: DatasetAnnotations,
                                    categories: Dict[int, str]) -> DatasetAnnotations:
    """
    Ensure annotations contain the specified category mapping.

    Args:
        annotations: DatasetAnnotations instance
        categories: Desired category mapping

    Returns:
        Updated DatasetAnnotations with category mapping set
    """
    # If annotations already have categories, we need to ensure consistency
    if annotations.categories:
        # Check for conflicts
        for cat_id, cat_name in categories.items():
            if cat_id in annotations.categories and annotations.categories[cat_id] != cat_name:
                logging.getLogger(__name__).warning(
                    f"Category ID {cat_id} conflict: "
                    f"existing='{annotations.categories[cat_id]}', new='{cat_name}'"
                )
    # Update categories
    annotations.categories = categories.copy()
    return annotations


def get_image_dimensions_from_handler(handler: Any, image_path: str) -> Tuple[int, int]:
    """
    Get image dimensions using handler's internal methods.

    Args:
        handler: Annotation handler instance
        image_path: Path to image file

    Returns:
        Tuple of (width, height) in pixels
    """
    # This is a placeholder - actual implementation depends on handler
    # In practice, handlers should have a method to get image dimensions
    # For now, we'll try to import OpenCV if available
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is not None:
            return img.shape[1], img.shape[0]
    except ImportError:
        pass

    # Fallback: use PIL if available
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size
    except ImportError:
        pass

    # Last resort: raise error
    raise ImportError("Cannot determine image dimensions: need OpenCV or PIL")


def normalize_path(path: str, base_dir: Path) -> Path:
    """
    Normalize path (convert relative path to absolute).

    Args:
        path: Path string (absolute or relative)
        base_dir: Base directory for resolving relative paths

    Returns:
        Normalized Path object
    """
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = base_dir / path_obj
    return path_obj.resolve()


def validate_conversion_chain(source_format: str, target_format: str,
                             allowed_chains: List[Tuple[str, str]]) -> bool:
    """
    Validate if a conversion chain is allowed.

    Args:
        source_format: Source format name
        target_format: Target format name
        allowed_chains: List of allowed (source, target) format pairs

    Returns:
        True if conversion is allowed, False otherwise
    """
    return (source_format, target_format) in allowed_chains


def create_conversion_chain(chain: List[str]) -> List[Tuple[str, str]]:
    """
    Create conversion steps list from format chain.

    Args:
        chain: List of format names, e.g., ["labelme", "yolo", "coco"]

    Returns:
        List of (source, target) format pairs for each step
    """
    steps = []
    for i in range(len(chain) - 1):
        steps.append((chain[i], chain[i + 1]))
    return steps


def resolve_image_paths(annotations: DatasetAnnotations,
                       source_dir: Path,
                       target_dir: Path) -> DatasetAnnotations:
    """
    Resolve and normalize image paths.

    Args:
        annotations: Annotation data
        source_dir: Source directory (for resolving relative paths)
        target_dir: Target directory (for generating new paths)

    Returns:
        Updated annotations with resolved image paths
    """
    updated_images = []
    for image_ann in annotations.images:
        # Resolve source path
        source_path = normalize_path(image_ann.image_path, source_dir)

        # Generate target path (preserve relative structure)
        if source_path.is_relative_to(source_dir):
            relative_path = source_path.relative_to(source_dir)
            target_path = target_dir / relative_path
        else:
            # Cannot determine relative path, use filename
            target_path = target_dir / Path(image_ann.image_path).name

        # Update image annotation
        updated_ann = type(image_ann)(
            image_id=image_ann.image_id,
            image_path=str(target_path),
            width=image_ann.width,
            height=image_ann.height,
            objects=image_ann.objects,
            original_data=image_ann.original_data
        )
        updated_images.append(updated_ann)

    return DatasetAnnotations(
        images=updated_images,
        categories=annotations.categories,
        dataset_info=annotations.dataset_info
    )