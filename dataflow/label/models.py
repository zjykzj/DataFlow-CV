"""
Data models for label annotations.

Defines the core data structures used throughout the label processing module.
Coordinates are stored in format-native representation — see DatasetAnnotations.format
to determine the coordinate semantics.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Regex for detecting path traversal characters in image_id values.
# Matches forward/backward slashes, parent-directory references, and
# leading dots (hidden files are not traversal but are rejected for
# safety — image IDs should be plain identifiers).
_UNSAFE_PATH_CHARS_RE = re.compile(r"[\\/]|\\.\\.")


class AnnotationFormat(Enum):
    """Supported annotation formats."""

    YOLO = "yolo"
    LABELME = "labelme"
    COCO = "coco"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box for object detection.

    Coordinate semantics depend on the parent DatasetAnnotations.format:

    - format=YOLO:  (x, y) = center, all values normalized [0, 1]
    - format=COCO:  (x, y) = top-left, all values in absolute pixels
    - format=LABELME: (x, y) = top-left, all values in absolute pixels
    """

    x: float  # X coordinate in native format space
    y: float  # Y coordinate in native format space
    width: float  # Width in native format space
    height: float  # Height in native format space


@dataclass
class Segmentation:
    """Segmentation polygon for instance segmentation.

    Coordinate semantics depend on the parent DatasetAnnotations.format:

    - format=YOLO:  points are (x, y) normalized [0, 1]
    - format=COCO:  points are (x, y) in absolute pixels
    - format=LABELME: points are (x, y) in absolute pixels
    """

    points: List[Tuple[float, float]]  # Polygon vertices in native coords
    rle: Optional[Dict[str, Any]] = None  # Preserved RLE data (COCO only)

    def has_rle(self) -> bool:
        return self.rle is not None


@dataclass
class ObjectAnnotation:
    """Annotation for a single object.

    All coordinate fields (bbox, segmentation) use the coordinate semantics
    of the parent DatasetAnnotations.format.
    """

    class_id: int  # Class ID
    class_name: str  # Class name
    bbox: Optional[BoundingBox] = None  # Bounding box (object detection)
    segmentation: Optional[Segmentation] = (
        None  # Segmentation polygon (instance segmentation)
    )
    confidence: float = 1.0  # Confidence score
    is_crowd: bool = False  # Whether this is a crowd annotation (COCO specific)

    def __post_init__(self):
        # Validate that at least one of bbox or segmentation is provided
        if self.bbox is None and self.segmentation is None:
            raise ValueError("At least one of bbox or segmentation must be provided")


@dataclass
class ImageAnnotation:
    """Annotations for a single image."""

    image_id: str  # Image ID (filename or unique identifier)
    image_path: str  # Path to image file
    width: int  # Image width in pixels
    height: int  # Image height in pixels
    objects: List[ObjectAnnotation] = field(
        default_factory=list
    )  # List of object annotations

    def __post_init__(self):
        # Validate image dimensions
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid image dimensions: {self.width}x{self.height}")

        # Validate image_id does not contain path traversal characters
        if not self.image_id:
            raise ValueError("image_id must not be empty")
        if "\x00" in self.image_id:
            raise ValueError(f"image_id contains null byte: {self.image_id!r}")
        if _UNSAFE_PATH_CHARS_RE.search(self.image_id):
            raise ValueError(
                f"image_id contains path traversal characters: {self.image_id!r}"
            )


@dataclass
class DatasetAnnotations:
    """Collection of annotations for a dataset.

    The `format` field defines the coordinate semantics for ALL contained
    BoundingBox and Segmentation objects:

    - YOLO:    center-based, normalized [0, 1]
    - COCO:    top-left origin, absolute pixels
    - LABELME: top-left origin, absolute pixels
    """

    images: List[ImageAnnotation] = field(
        default_factory=list
    )  # List of image annotations
    categories: Dict[int, str] = field(
        default_factory=dict
    )  # Category mapping (ID -> name)
    format: AnnotationFormat = AnnotationFormat.UNKNOWN  # Format governing coords
    dataset_info: Dict[str, Any] = field(default_factory=dict)  # Dataset metadata

    def __post_init__(self):
        # Validate categories
        for cat_id, cat_name in self.categories.items():
            if not isinstance(cat_id, int):
                raise ValueError(
                    f"Category ID must be integer, got {type(cat_id)}: {cat_id}"
                )
            if not isinstance(cat_name, str):
                raise ValueError(
                    f"Category name must be string, got {type(cat_name)}: {cat_name}"
                )

    def add_image(self, image_annotation: ImageAnnotation):
        """Add an image annotation to the dataset."""
        self.images.append(image_annotation)

    def add_category(self, cat_id: int, cat_name: str):
        """Add a category to the dataset."""
        if cat_id in self.categories and self.categories[cat_id] != cat_name:
            raise ValueError(
                f"Category ID {cat_id} already exists with name {self.categories[cat_id]}"
            )
        self.categories[cat_id] = cat_name

    def get_category_name(self, cat_id: int) -> Optional[str]:
        """Get category name by ID."""
        return self.categories.get(cat_id)

    def get_category_id(self, cat_name: str) -> Optional[int]:
        """Get category ID by name."""
        for cat_id, name in self.categories.items():
            if name == cat_name:
                return cat_id
        return None

    @property
    def num_images(self) -> int:
        """Number of images in the dataset."""
        return len(self.images)

    @property
    def num_objects(self) -> int:
        """Total number of objects in the dataset."""
        return sum(len(img.objects) for img in self.images)

    @property
    def num_categories(self) -> int:
        """Number of categories in the dataset."""
        return len(self.categories)
