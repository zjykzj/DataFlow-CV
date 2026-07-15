"""
Utility functions for format conversion.

Provides helper functions for category handling, path resolution,
conversion validation, and shared coordinate transforms.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..label.models import BoundingBox, DatasetAnnotations, Segmentation


# ---------------------------------------------------------------------------
# Shared coordinate transforms
# ---------------------------------------------------------------------------

def yolo_to_absolute_pixel(
    bbox: Optional[BoundingBox],
    seg: Optional[Segmentation],
    img_width: int,
    img_height: int,
) -> Tuple[Optional[BoundingBox], Optional[Segmentation]]:
    """Convert YOLO normalized center → absolute pixel top-left.

    Bbox transformation:
        (cx_norm, cy_norm, w_norm, h_norm) → (x_tl, y_tl, w_abs, h_abs)

    Segmentation transformation:
        (x_norm, y_norm) per point → (x_abs, y_abs) per point

    This is a pure function — stateless, no side effects.

    Used by:
        - YoloAndCocoConverter (YOLO → COCO)
        - LabelMeAndYoloConverter (YOLO → LabelMe)
    """
    new_bbox = None
    new_seg = None

    if bbox:
        if img_width <= 0 or img_height <= 0:
            raise ValueError(
                f"Image dimensions must be positive, got {img_width}x{img_height}"
            )
        cx_abs = bbox.x * img_width
        cy_abs = bbox.y * img_height
        w_abs = bbox.width * img_width
        h_abs = bbox.height * img_height
        x_tl = cx_abs - w_abs / 2
        y_tl = cy_abs - h_abs / 2
        new_bbox = BoundingBox(x=x_tl, y=y_tl, width=w_abs, height=h_abs)

    if seg:
        if img_width <= 0 or img_height <= 0:
            raise ValueError(
                f"Image dimensions must be positive, got {img_width}x{img_height}"
            )
        new_points = [
            (x * img_width, y * img_height) for x, y in seg.points
        ]
        new_seg = Segmentation(points=new_points, rle=seg.rle)

    return new_bbox, new_seg


def absolute_pixel_to_yolo(
    bbox: Optional[BoundingBox],
    seg: Optional[Segmentation],
    img_width: int,
    img_height: int,
) -> Tuple[Optional[BoundingBox], Optional[Segmentation]]:
    """Convert absolute pixel top-left → YOLO normalized center.

    Bbox transformation:
        (x_tl, y_tl, w_abs, h_abs) → (cx_norm, cy_norm, w_norm, h_norm)

    Segmentation transformation:
        (x_abs, y_abs) per point → (x_norm, y_norm) per point

    This is a pure function — stateless, no side effects.

    Used by:
        - YoloAndCocoConverter (COCO → YOLO)
        - LabelMeAndYoloConverter (LabelMe → YOLO)
    """
    new_bbox = None
    new_seg = None

    if bbox:
        if img_width <= 0 or img_height <= 0:
            raise ValueError(
                f"Image dimensions must be positive, got {img_width}x{img_height}"
            )
        cx_abs = bbox.x + bbox.width / 2
        cy_abs = bbox.y + bbox.height / 2
        cx_norm = cx_abs / img_width
        cy_norm = cy_abs / img_height
        w_norm = bbox.width / img_width
        h_norm = bbox.height / img_height
        new_bbox = BoundingBox(x=cx_norm, y=cy_norm, width=w_norm, height=h_norm)

    if seg:
        if img_width <= 0 or img_height <= 0:
            raise ValueError(
                f"Image dimensions must be positive, got {img_width}x{img_height}"
            )
        new_points = [
            (x / img_width, y / img_height) for x, y in seg.points
        ]
        new_seg = Segmentation(points=new_points, rle=seg.rle)

    return new_bbox, new_seg


# ---------------------------------------------------------------------------
# COCO helpers
# ---------------------------------------------------------------------------

def ensure_coco_categories_for_streaming(
    converter: Any,
    source_handler: Any,
    source_path: str,
) -> None:
    """Pre-load COCO categories from JSON for streaming conversions.

    When the source format is COCO, the handler does not load categories
    until ``iter_images()`` runs.  This reads them directly from the JSON
    file so ``create_target_handler()`` can generate ``classes.txt``.

    Args:
        converter: The ``BaseConverter`` instance (needs
            ``_source_annotations_for_target`` and ``source_format``).
        source_handler: Source handler (unused; for compat with
            super-class call pattern).
        source_path: Path to the COCO JSON annotation file.
    """
    from ..label.models import AnnotationFormat, DatasetAnnotations

    # Try default (handler.categories)
    converter._source_annotations_for_target = None
    cats = getattr(source_handler, "categories", None)
    if isinstance(cats, dict) and cats:
        converter._source_annotations_for_target = DatasetAnnotations(
            format=AnnotationFormat.UNKNOWN,
            categories=cats.copy(),
        )

    # If still no categories and source is COCO, read from JSON
    if (
        converter.source_format == "coco"
        and (
            not converter._source_annotations_for_target
            or not converter._source_annotations_for_target.categories
        )
    ):
        categories_dict = read_coco_categories(source_path)
        if categories_dict:
            converter._source_annotations_for_target = DatasetAnnotations(
                format=AnnotationFormat.COCO,
                categories=categories_dict,
            )


def read_coco_categories(json_path: str) -> Dict[int, str]:
    """Read categories from a COCO JSON file without loading the full dataset.

    Only parses the ``"categories"`` array — does not load images or
    annotations. Returns an empty dict on any error (file not found,
    invalid JSON, missing key).

    Args:
        json_path: Path to a COCO JSON annotation file.

    Returns:
        Category mapping ``{cat_id: cat_name}``, or empty dict on error.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
        categories: Dict[int, str] = {}
        for cat in coco_data.get("categories", []):
            cat_id = cat.get("id")
            cat_name = cat.get("name", "")
            if cat_id is not None:
                categories[cat_id] = cat_name
        return categories
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Category utilities
# ---------------------------------------------------------------------------


def extract_categories_from_annotations(
    annotations: DatasetAnnotations,
) -> Dict[int, str]:
    """Extract category mapping from DatasetAnnotations."""
    return annotations.categories.copy()


def generate_classes_file(categories: Dict[int, str], output_path: Path) -> bool:
    """
    Generate classes.txt file from category mapping.

    Delegates to ``dataflow.label.utils.generate_classes_file``.

    Args:
        categories: Dictionary mapping category IDs to names
        output_path: Path to output classes.txt file

    Returns:
        True if successful, False otherwise
    """
    from dataflow.label.utils import generate_classes_file as _gen

    return _gen(categories, output_path)


def load_classes_file(class_file: Path) -> Dict[int, str]:
    """
    Load category mapping from classes.txt file.

    Delegates to ``dataflow.label.utils.load_classes_file``.

    Args:
        class_file: Path to classes.txt file

    Returns:
        Dictionary mapping index (starting from 0) to category name
    """
    from dataflow.label.utils import load_classes_file as _load

    return _load(class_file)


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


def ensure_categories_in_annotations(
    annotations: DatasetAnnotations, categories: Dict[int, str]
) -> DatasetAnnotations:
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
            if (
                cat_id in annotations.categories
                and annotations.categories[cat_id] != cat_name
            ):
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


def validate_conversion_chain(
    source_format: str, target_format: str, allowed_chains: List[Tuple[str, str]]
) -> bool:
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
        chain: List of format names, e.g., ["yolo", "labelme", "coco"]

    Returns:
        List of (source, target) format pairs for each step
    """
    steps = []
    for i in range(len(chain) - 1):
        steps.append((chain[i], chain[i + 1]))
    return steps


def resolve_image_paths(
    annotations: DatasetAnnotations, source_dir: Path, target_dir: Path
) -> DatasetAnnotations:
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
        # Path.is_relative_to() is Python 3.9+; use relative_to()
        # (available in 3.8) with try/except for compatibility.
        try:
            relative_path = source_path.relative_to(source_dir)
        except ValueError:
            # Cannot determine relative path, use filename
            target_path = target_dir / Path(image_ann.image_path).name
        else:
            target_path = target_dir / relative_path

        # Update image annotation
        updated_ann = type(image_ann)(
            image_id=image_ann.image_id,
            image_path=str(target_path),
            width=image_ann.width,
            height=image_ann.height,
            objects=image_ann.objects,
        )
        updated_images.append(updated_ann)

    return DatasetAnnotations(
        images=updated_images,
        categories=annotations.categories,
        format=annotations.format,
        dataset_info=annotations.dataset_info,
    )
