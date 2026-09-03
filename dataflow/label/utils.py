"""
Utility functions for the label module.
"""

import hashlib
from pathlib import Path
from typing import Dict, Optional


def parse_yolo_class_id(token: str) -> Optional[int]:
    """Parse a YOLO class ID token, accepting integer-valued float strings.

    YOLO format specifies ``class_id`` as an integer, but some tooling
    outputs float-formatted values like ``5.000000``.  This parser
    handles both cases gracefully and is the **canonical implementation**
    for lenient parsing of class IDs from raw YOLO text.

    For strict validation (raising on non-integer floats), use
    ``YoloAnnotationHandler._parse_class_id()`` instead.

    Args:
        token: Raw string token from a YOLO annotation line.

    Returns:
        Integer class ID, or ``None`` if *token* is not a valid
        non-negative integer or integer-valued float.

    Examples:
        >>> parse_yolo_class_id("5")
        5
        >>> parse_yolo_class_id("5.000000")
        5
        >>> parse_yolo_class_id("0.0")
        0
        >>> parse_yolo_class_id("0.5")
        None
        >>> parse_yolo_class_id("-1")
        None
        >>> parse_yolo_class_id("abc")
        None
    """
    # Fast path: try direct int conversion first
    try:
        cid = int(token)
        if cid >= 0:
            return cid
        return None
    except ValueError:
        pass
    try:
        val = float(token)
        if val.is_integer() and val >= 0:
            return int(val)
    except (ValueError, OverflowError):
        pass
    return None


def calculate_file_hash(file_path: Path, algorithm: str = "md5") -> str:
    """Calculate hash of a file."""
    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


# Category management utilities


def extract_categories_from_coco_data(coco_data: dict) -> Dict[int, str]:
    """
    Extract category mapping from COCO JSON data.

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
        with open(output_path, "w", encoding="utf-8") as f:
            for cat_id in sorted(categories.keys()):
                f.write(f"{categories[cat_id]}\n")
        return True
    except Exception as e:
        import logging

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
        with open(class_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    categories[i] = line
    except Exception as e:
        import logging

        logging.getLogger(__name__).error(f"Failed to load classes file {class_file}: {e}")
    return categories
