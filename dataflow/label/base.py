"""
Base annotation handler abstract class.

Defines the interface for all annotation format handlers.
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .models import AnnotationFormat, DatasetAnnotations, ImageAnnotation


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
    def iter_images(self) -> Iterator[ImageAnnotation]:
        """Yield ImageAnnotation objects one at a time (streaming).

        Validates directories and categories upfront
        (raises immediately if invalid). Then scans annotation files
        and yields each successfully parsed image.

        Image errors (missing file, unreadable) always skip regardless
        of strict_mode.

        Strict mode (default):
            Raises ValueError on the first invalid file or annotation
            line. The iterator stops — partial results before the error
            are available.

        Non-strict mode:
            Skips invalid files/lines, logs warnings, continues yielding
            valid images.

        Yields:
            ImageAnnotation with format-native coordinates, one per
            image file.

        Raises:
            ValueError: In strict mode, when parsing fails for any file
                or line.
        """
        pass

    @abstractmethod
    def write(
        self, annotations: DatasetAnnotations, *args, **kwargs
    ) -> AnnotationResult:
        """Write DatasetAnnotations to annotation files."""
        pass

    @abstractmethod
    def write_one(
        self, image_ann: ImageAnnotation, output_dir: Path
    ) -> AnnotationResult:
        """Write annotations for a single image (streaming write).

        Called by the Convert module's streaming pipeline
        (``stream_convert()``) to write each image immediately after
        conversion. Receives an ``ImageAnnotation`` with coordinates
        already in the target format's native space.

        Args:
            image_ann: Single ImageAnnotation with target-native
                coordinates.
            output_dir: Directory to write the output file into.

        Returns:
            AnnotationResult with success status.

        Note:
            For single-file formats (COCO JSON), this must raise
            ``NotImplementedError``. Use ``write()`` instead.
        """
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

    def _validate_absolute_coordinate(self, value: float, name: str) -> bool:
        """Validate absolute coordinate is a finite non-negative number."""
        if not math.isfinite(value):
            self._log_error(f"Absolute {name} is not finite: {value}")
            return False
        if value < 0:
            self._log_error(f"Absolute {name} is negative: {value}")
            return False
        return True

    def _clamp_abs_bbox(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        img_width: int,
        img_height: int,
    ) -> tuple:
        """Clamp absolute-pixel bbox to image boundaries.

        Tolerates minor floating-point imprecision at image edges (e.g.,
        x=-0.39 → 0). Emits a WARNING if any coordinate value is modified.
        Returns (x, y, w, h) clamped to [0, img_width] × [0, img_height].
        """
        x_orig, y_orig, w_orig, h_orig = x, y, w, h
        right_orig = x_orig + w_orig
        bottom_orig = y_orig + h_orig

        # Clamp top-left to [0, img_width] × [0, img_height]
        x = max(0.0, min(float(img_width), x))
        y = max(0.0, min(float(img_height), y))

        # Clamp width/height so bottom-right stays within image
        w = max(0.0, min(float(img_width) - x, w))
        h = max(0.0, min(float(img_height) - y, h))

        right_new = x + w
        bottom_new = y + h

        changed = (
            abs(x - x_orig) > 1e-9
            or abs(y - y_orig) > 1e-9
            or abs(right_new - right_orig) > 1e-9
            or abs(bottom_new - bottom_orig) > 1e-9
        )

        if changed:
            self._log_warning(
                f"Clamped bbox to image boundaries: "
                f"({x_orig:.2f}, {y_orig:.2f}, {w_orig:.2f}, {h_orig:.2f}) "
                f"→ ({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f})"
            )

        return x, y, w, h

    def _clamp_abs_points(
        self,
        points: list,
        img_width: int,
        img_height: int,
    ) -> list:
        """Clamp absolute-pixel polygon points to image boundaries.

        Each point (x, y) is clamped to [0, img_width] × [0, img_height].
        Emits a single WARNING if any point coordinate is modified.
        Returns list of clamped (x, y) tuples.
        """
        clamped_points = []
        changed = False

        for x, y in points:
            cx = max(0.0, min(float(img_width), float(x)))
            cy = max(0.0, min(float(img_height), float(y)))
            clamped_points.append((cx, cy))
            if abs(cx - x) > 1e-9 or abs(cy - y) > 1e-9:
                changed = True

        if changed:
            self._log_warning(
                f"Clamped polygon points to image boundaries "
                f"[0, {img_width}] × [0, {img_height}]"
            )

        return clamped_points

    def _validate_bbox(
        self, bbox, format: AnnotationFormat = AnnotationFormat.YOLO
    ) -> bool:
        """Validate bounding box based on format-specific coordinate semantics."""
        if bbox is None:
            return True

        if format == AnnotationFormat.YOLO:
            return self._validate_bbox_normalized(bbox)
        else:
            return self._validate_bbox_absolute(bbox)

    def _validate_bbox_normalized(self, bbox) -> bool:
        """Validate YOLO-style normalized bounding box."""
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
            self._log_error(
                f"bbox overflows right boundary: x={bbox.x}, w={bbox.width}"
            )
            return False
        if bbox.y - half_h < 0:
            self._log_error(f"bbox overflows top boundary: y={bbox.y}, h={bbox.height}")
            return False
        if bbox.y + half_h > 1:
            self._log_error(
                f"bbox overflows bottom boundary: y={bbox.y}, h={bbox.height}"
            )
            return False

        return True

    def _validate_bbox_absolute(self, bbox) -> bool:
        """Validate absolute-pixel bounding box (COCO/LabelMe)."""
        checks = [
            self._validate_absolute_coordinate(bbox.x, "bbox.x"),
            self._validate_absolute_coordinate(bbox.y, "bbox.y"),
        ]
        if not all(checks):
            return False

        # Width and height must be strictly positive
        if bbox.width <= 0:
            self._log_error(f"bbox.width must be > 0, got {bbox.width}")
            return False
        if bbox.height <= 0:
            self._log_error(f"bbox.height must be > 0, got {bbox.height}")
            return False

        return True

    def _validate_segmentation_points(
        self,
        points: List[Tuple[float, float]],
        format: AnnotationFormat = AnnotationFormat.YOLO,
    ) -> bool:
        """Validate segmentation polygon points based on format."""
        if not points:
            self._log_error("Segmentation polygon has no points")
            return False

        if format == AnnotationFormat.YOLO:
            validator = self._validate_normalized_coordinate
        else:
            validator = self._validate_absolute_coordinate

        for i, (x, y) in enumerate(points):
            if not validator(x, f"point[{i}].x"):
                return False
            if not validator(y, f"point[{i}].y"):
                return False

        # Check polygon has at least 3 points
        if len(points) < 3:
            self._log_error(
                f"Segmentation polygon needs at least 3 points, got {len(points)}"
            )
            return False

        return True
