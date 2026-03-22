"""
Data models for label annotations.

Defines the core data structures used throughout the label processing module.
All coordinates are normalized (0-1 range).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class AnnotationFormat(Enum):
    """Supported annotation formats."""
    LABELME = "labelme"
    YOLO = "yolo"
    COCO = "coco"
    UNKNOWN = "unknown"


@dataclass
class OriginalData:
    """Container for original annotation data to enable lossless round-trip."""
    format: str  # "labelme", "yolo", "coco"
    raw_data: Dict[str, Any]  # Original annotation data
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def has_data(self) -> bool:
        """Check if original data exists."""
        return bool(self.raw_data)


@dataclass
class BoundingBox:
    """Bounding box for object detection."""

    x: float  # Center x coordinate (normalized)
    y: float  # Center y coordinate (normalized)
    width: float  # Width (normalized)
    height: float  # Height (normalized)
    original_data: Optional[OriginalData] = None  # Original annotation data for lossless round-trip

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

    def has_original_data(self) -> bool:
        """Check if original data is available."""
        return self.original_data is not None and self.original_data.has_data()


@dataclass
class Segmentation:
    """Segmentation polygon for instance segmentation."""

    points: List[Tuple[float, float]]  # Polygon points (normalized coordinates)
    rle: Optional[Dict[str, Any]] = None  # Original RLE data if available
    original_data: Optional[OriginalData] = None  # Original annotation data for lossless round-trip

    def has_rle(self) -> bool:
        return self.rle is not None

    def has_original_data(self) -> bool:
        """Check if original data is available."""
        return self.original_data is not None and self.original_data.has_data()

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
    original_data: Optional[OriginalData] = None  # Original annotation data for lossless round-trip

    def __post_init__(self):
        # Validate that at least one of bbox or segmentation is provided
        if self.bbox is None and self.segmentation is None:
            raise ValueError("At least one of bbox or segmentation must be provided")

    def has_original_data(self) -> bool:
        """Check if original data is available."""
        return self.original_data is not None and self.original_data.has_data()


@dataclass
class ImageAnnotation:
    """Annotations for a single image."""

    image_id: str  # Image ID (filename or unique identifier)
    image_path: str  # Path to image file
    width: int  # Image width in pixels
    height: int  # Image height in pixels
    objects: List[ObjectAnnotation] = field(default_factory=list)  # List of object annotations
    original_data: Optional[OriginalData] = None  # Original annotation data for lossless round-trip

    def __post_init__(self):
        # Validate image dimensions
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid image dimensions: {self.width}x{self.height}")

    def has_original_data(self) -> bool:
        """Check if original data is available."""
        return self.original_data is not None and self.original_data.has_data()


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


class OriginalDataManager:
    """Manager for handling original annotation data."""

    @staticmethod
    def should_use_original(obj: ObjectAnnotation, target_format: str) -> bool:
        """Determine if original data should be used for writing."""
        return (obj.has_original_data() and
                obj.original_data.format == target_format)

    @staticmethod
    def merge_original_data(existing: Optional[OriginalData], new: OriginalData) -> OriginalData:
        """Merge original data from different sources."""
        if existing is None:
            return new

        # If formats match, we need to decide how to merge
        if existing.format == new.format:
            # For lossless preservation, we should keep the existing data
            # since it's the first original source
            return existing
        else:
            # Different formats - can't merge, return the newer one
            # This might indicate a format conversion chain
            return new

    @staticmethod
    def extract_original_coordinates(obj: ObjectAnnotation, img_width: int, img_height: int) -> Tuple[Optional[List], Optional[List]]:
        """Extract original coordinates from original data if available."""
        if not obj.has_original_data():
            return None, None

        original_data = obj.original_data.raw_data
        bbox_points = None
        segmentation_points = None

        if obj.original_data.format == AnnotationFormat.LABELME.value:
            # LabelMe shape data
            if "points" in original_data and original_data.get("shape_type") == "rectangle":
                bbox_points = original_data["points"]
            elif "points" in original_data and original_data.get("shape_type") == "polygon":
                segmentation_points = original_data["points"]
        elif obj.original_data.format == AnnotationFormat.YOLO.value:
            # YOLO line data
            if "items" in original_data:
                items = original_data["items"]
                if original_data.get("is_detection"):
                    # YOLO detection: class_id x_center y_center width height
                    if len(items) >= 5:
                        _, x_center, y_center, width, height = items[:5]
                        # Convert normalized center to absolute for comparison
                        bbox_points = [
                            [(x_center - width/2) * img_width, (y_center - height/2) * img_height],
                            [(x_center + width/2) * img_width, (y_center + height/2) * img_height]
                        ]
                elif original_data.get("is_segmentation"):
                    # YOLO segmentation: class_id x1 y1 x2 y2 ...
                    if len(items) > 1:
                        coords = items[1:]
                        segmentation_points = []
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                x = coords[i] * img_width
                                y = coords[i + 1] * img_height
                                segmentation_points.append([x, y])
        elif obj.original_data.format == AnnotationFormat.COCO.value:
            # COCO annotation data
            if "bbox" in original_data and original_data["bbox"]:
                x_abs, y_abs, w_abs, h_abs = original_data["bbox"]
                bbox_points = [
                    [x_abs, y_abs],
                    [x_abs + w_abs, y_abs + h_abs]
                ]
            if "segmentation" in original_data:
                seg_data = original_data["segmentation"]
                if isinstance(seg_data, list) and seg_data:
                    # Polygon format
                    segmentation_points = []
                    for polygon in seg_data:
                        for i in range(0, len(polygon), 2):
                            if i + 1 < len(polygon):
                                segmentation_points.append([polygon[i], polygon[i + 1]])

        return bbox_points, segmentation_points