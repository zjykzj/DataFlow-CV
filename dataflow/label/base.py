"""
Base annotation handler abstract class.

Defines the interface for all annotation format handlers.
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .models import DatasetAnnotations


class ImageError(Exception):
    """Exception raised for image-related errors that should always be treated as warnings.

    Image errors (missing files, unreadable images, invalid dimensions) should
    always skip processing regardless of strict_mode, unlike validation errors.
    """

    pass


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

    def __init__(
        self, strict_mode: bool = True, logger: Optional[logging.Logger] = None
    ):
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
    def write(
        self, annotations: DatasetAnnotations, *args, **kwargs
    ) -> AnnotationResult:
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

    def _log_debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def _set_annotation_flags(self, annotations: DatasetAnnotations):
        """Set handler flags based on annotation data."""
        has_detection = any(
            obj.bbox is not None for img in annotations.images for obj in img.objects
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
            self._log_info(
                "Detected mixed annotation types: object detection + instance segmentation"
            )
        elif self.is_det:
            self._log_info("Detected object detection annotations")
        elif self.is_seg:
            self._log_info("Detected instance segmentation annotations")
        else:
            self._log_warning("No valid annotations detected (no bbox or segmentation)")

    def _validate_image_dimensions(self, width: int, height: int) -> bool:
        """Validate image dimensions are positive integers."""
        if not isinstance(width, int) or not isinstance(height, int):
            self._log_error(
                f"Image dimensions must be integers, got {type(width).__name__}x{type(height).__name__}"
            )
            return False
        if width <= 0 or height <= 0:
            self._log_error(f"Invalid image dimensions: {width}x{height}")
            return False
        return True

    def _validate_normalized_coordinate(self, value: float, name: str) -> bool:
        """Validate normalized coordinate is a finite number in [0, 1] range."""
        if not math.isfinite(value):
            self._log_error(f"Normalized {name} is not finite: {value}")
            return False
        if value < 0 or value > 1:
            self._log_error(f"Normalized {name} out of range [0, 1]: {value}")
            return False
        return True

    def _validate_bbox(self, bbox) -> bool:
        """Validate bounding box coordinates, dimensions, and boundary containment."""
        if bbox is None:
            return True

        # Validate each coordinate is a finite number in [0, 1]
        checks = [
            self._validate_normalized_coordinate(bbox.x, "bbox.x"),
            self._validate_normalized_coordinate(bbox.y, "bbox.y"),
            self._validate_normalized_coordinate(bbox.width, "bbox.width"),
            self._validate_normalized_coordinate(bbox.height, "bbox.height"),
        ]
        if not all(checks):
            return False

        # Validate width and height are strictly positive
        if bbox.width <= 0:
            self._log_error(f"bbox.width must be > 0, got {bbox.width}")
            return False
        if bbox.height <= 0:
            self._log_error(f"bbox.height must be > 0, got {bbox.height}")
            return False

        # Validate bounding box is contained within image boundaries [0, 1]
        half_w = bbox.width / 2
        half_h = bbox.height / 2
        if bbox.x - half_w < 0:
            self._log_error(f"bbox overflows left boundary: x={bbox.x}, w={bbox.width}")
            return False
        if bbox.x + half_w > 1:
            self._log_error(f"bbox overflows right boundary: x={bbox.x}, w={bbox.width}")
            return False
        if bbox.y - half_h < 0:
            self._log_error(f"bbox overflows top boundary: y={bbox.y}, h={bbox.height}")
            return False
        if bbox.y + half_h > 1:
            self._log_error(f"bbox overflows bottom boundary: y={bbox.y}, h={bbox.height}")
            return False

        return True

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
            self._log_error(
                f"Segmentation polygon needs at least 3 points, got {len(points)}"
            )
            return False

        return True
