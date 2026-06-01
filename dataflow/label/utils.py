"""
Utility functions for the label module.
"""

import hashlib
from pathlib import Path
from typing import Dict


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

        logging.getLogger(__name__).error(
            f"Failed to write classes file {output_path}: {e}"
        )
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

        logging.getLogger(__name__).error(
            f"Failed to load classes file {class_file}: {e}"
        )
    return categories
