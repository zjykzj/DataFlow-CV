"""
Base annotation handler abstract class.

Defines the interface for all annotation format handlers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging

from .models import DatasetAnnotations


@dataclass
class AnnotationResult:
    """Result of an annotation processing operation."""

    success: bool
    data: Optional[Any] = None
    message: str = ""
    errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.success = False
        if not self.message:
            self.message = error
        else:
            self.message += f"; {error}"

    def add_info(self, info: str):
        """Add an informational message."""
        if not self.message:
            self.message = info
        else:
            self.message += f"; {info}"


class BaseAnnotationHandler(ABC):
    """Abstract base class for annotation format handlers."""

    def __init__(self, strict_mode: bool = True, logger: Optional[logging.Logger] = None):
        self.strict_mode = strict_mode
        self.logger = logger or logging.getLogger(__name__)
        self.is_det = False  # Whether annotations are for object detection
        self.is_seg = False  # Whether annotations are for instance segmentation
        self.is_rle = False  # Whether annotations use RLE format (COCO specific)

    @abstractmethod
    def read(self, *args, **kwargs) -> AnnotationResult:
        """Read annotation files and return DatasetAnnotations."""
        pass

    @abstractmethod
    def write(self, annotations: DatasetAnnotations, *args, **kwargs) -> AnnotationResult:
        """Write DatasetAnnotations to annotation files."""
        pass

    @abstractmethod
    def validate(self, *args, **kwargs) -> bool:
        """Validate annotation files."""
        pass

    def _log_info(self, message: str):
        """Log informational message."""
        self.logger.info(message)

    def _log_error(self, message: str):
        """Log error message and raise exception in strict mode."""
        self.logger.error(message)
        if self.strict_mode:
            raise ValueError(message)

    def _log_warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def _set_annotation_flags(self, annotations: DatasetAnnotations):
        """Set handler flags based on annotation data."""
        has_detection = any(
            obj.bbox is not None
            for img in annotations.images
            for obj in img.objects
        )
        has_segmentation = any(
            obj.segmentation is not None
            for img in annotations.images
            for obj in img.objects
        )

        self.is_det = has_detection
        self.is_seg = has_segmentation

        # Log detection results
        if self.is_det and self.is_seg:
            self._log_info("Detected mixed annotation types: object detection + instance segmentation")
        elif self.is_det:
            self._log_info("Detected object detection annotations")
        elif self.is_seg:
            self._log_info("Detected instance segmentation annotations")
        else:
            self._log_warning("No valid annotations detected (no bbox or segmentation)")

    def _validate_image_dimensions(self, width: int, height: int) -> bool:
        """Validate image dimensions are positive."""
        if width <= 0 or height <= 0:
            self._log_error(f"Invalid image dimensions: {width}x{height}")
            return False
        return True

    def _validate_normalized_coordinate(self, value: float, name: str) -> bool:
        """Validate normalized coordinate is in [0, 1] range."""
        if value < 0 or value > 1:
            self._log_error(f"Normalized {name} out of range [0, 1]: {value}")
            return False
        return True

    def _validate_bbox(self, bbox) -> bool:
        """Validate bounding box coordinates."""
        if bbox is None:
            return True

        checks = [
            self._validate_normalized_coordinate(bbox.x, "bbox.x"),
            self._validate_normalized_coordinate(bbox.y, "bbox.y"),
            self._validate_normalized_coordinate(bbox.width, "bbox.width"),
            self._validate_normalized_coordinate(bbox.height, "bbox.height"),
        ]
        return all(checks)

    def _validate_segmentation_points(self, points: List[Tuple[float, float]]) -> bool:
        """Validate segmentation polygon points."""
        if not points:
            self._log_error("Segmentation polygon has no points")
            return False

        for i, (x, y) in enumerate(points):
            if not self._validate_normalized_coordinate(x, f"point[{i}].x"):
                return False
            if not self._validate_normalized_coordinate(y, f"point[{i}].y"):
                return False

        # Check polygon has at least 3 points
        if len(points) < 3:
            self._log_error(f"Segmentation polygon needs at least 3 points, got {len(points)}")
            return False

        return True