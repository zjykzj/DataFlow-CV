"""
Data models for label annotations.

Defines the core data structures used throughout the label processing module.
All coordinates are normalized (0-1 range).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box for object detection."""

    x: float  # Center x coordinate (normalized)
    y: float  # Center y coordinate (normalized)
    width: float  # Width (normalized)
    height: float  # Height (normalized)

    def xywh_abs(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert to absolute pixel coordinates."""
        return (
            int(self.x * img_width),
            int(self.y * img_height),
            int(self.width * img_width),
            int(self.height * img_height)
        )

    def xyxy(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert to top-left to bottom-right coordinates."""
        x_center = self.x * img_width
        y_center = self.y * img_height
        w = self.width * img_width
        h = self.height * img_height
        return (
            int(x_center - w / 2),
            int(y_center - h / 2),
            int(x_center + w / 2),
            int(y_center + h / 2)
        )


@dataclass
class Segmentation:
    """Segmentation polygon for instance segmentation."""

    points: List[Tuple[float, float]]  # Polygon points (normalized coordinates)
    rle: Optional[Dict[str, Any]] = None  # Original RLE data if available

    def has_rle(self) -> bool:
        return self.rle is not None

    def points_abs(self, img_width: int, img_height: int) -> List[Tuple[int, int]]:
        """Convert to absolute pixel coordinates."""
        return [(int(x * img_width), int(y * img_height)) for x, y in self.points]


@dataclass
class ObjectAnnotation:
    """Annotation for a single object."""

    class_id: int  # Class ID
    class_name: str  # Class name
    bbox: Optional[BoundingBox] = None  # Bounding box (object detection)
    segmentation: Optional[Segmentation] = None  # Segmentation polygon (instance segmentation)
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
    objects: List[ObjectAnnotation] = field(default_factory=list)  # List of object annotations

    def __post_init__(self):
        # Validate image dimensions
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid image dimensions: {self.width}x{self.height}")


@dataclass
class DatasetAnnotations:
    """Collection of annotations for a dataset."""

    images: List[ImageAnnotation] = field(default_factory=list)  # List of image annotations
    categories: Dict[int, str] = field(default_factory=dict)  # Category mapping (ID -> name)
    dataset_info: Dict[str, Any] = field(default_factory=dict)  # Dataset metadata

    def __post_init__(self):
        # Validate categories
        for cat_id, cat_name in self.categories.items():
            if not isinstance(cat_id, int):
                raise ValueError(f"Category ID must be integer, got {type(cat_id)}: {cat_id}")
            if not isinstance(cat_name, str):
                raise ValueError(f"Category name must be string, got {type(cat_name)}: {cat_name}")

    def add_image(self, image_annotation: ImageAnnotation):
        """Add an image annotation to the dataset."""
        self.images.append(image_annotation)

    def add_category(self, cat_id: int, cat_name: str):
        """Add a category to the dataset."""
        if cat_id in self.categories and self.categories[cat_id] != cat_name:
            raise ValueError(f"Category ID {cat_id} already exists with name {self.categories[cat_id]}")
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