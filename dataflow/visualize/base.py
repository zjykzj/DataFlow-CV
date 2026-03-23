"""
Base visualization classes for DataFlow-CV.

Defines the abstract base class for all visualizers and supporting
data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging

import cv2
import numpy as np

from dataflow.label.models import (
    DatasetAnnotations,
    ImageAnnotation,
    ObjectAnnotation,
    BoundingBox,
    Segmentation,
)
from dataflow.util import FileOperations


@dataclass
class VisualizationResult:
    """Visualization processing result."""

    success: bool
    data: Optional[Any] = None
    message: str = ""
    errors: List[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error message to the result."""
        self.errors.append(error)


class ColorManager:
    """Color manager that ensures consistent colors for the same class."""

    def __init__(self) -> None:
        # Predefined 20 high-contrast colors (BGR format)
        self.predefined_colors = [
            (0, 0, 255),  # Red
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (0, 128, 255),  # Orange
            (128, 0, 255),  # Pink
            (0, 255, 128),  # Light green
            (255, 128, 0),  # Sky blue
            (128, 255, 0),  # Lime
            (255, 0, 128),  # Rose
            (0, 128, 128),  # Olive
            (128, 0, 128),  # Purple
            (128, 128, 0),  # Teal
            (192, 192, 192),  # Silver
            (128, 128, 128),  # Gray
            (64, 64, 64),  # Dark gray
            (0, 64, 128),  # Dark blue
            (128, 64, 0),  # Brown
        ]
        self.color_cache: Dict[int, Tuple[int, int, int]] = {}

    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for a class ID."""
        if class_id in self.color_cache:
            return self.color_cache[class_id]

        # Cycle through predefined colors
        color_idx = class_id % len(self.predefined_colors)
        color = self.predefined_colors[color_idx]
        self.color_cache[class_id] = color

        return color


class BaseVisualizer(ABC):
    """Abstract base class for all visualizers."""

    def __init__(
        self,
        label_dir: Union[str, Path],
        image_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        is_show: bool = True,
        is_save: bool = False,
        strict_mode: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.label_dir = Path(label_dir)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.is_show = is_show
        self.is_save = is_save
        self.strict_mode = strict_mode
        self.logger = logger or logging.getLogger(__name__)
        self.file_ops = FileOperations(logger=self.logger)

        # Configuration parameters
        self.config = {
            "bbox_thickness": 2,  # Bounding box line width
            "seg_thickness": 1,  # Segmentation line width
            "seg_alpha": 0.3,  # Segmentation mask transparency
            "text_thickness": 1,  # Text line width
            "text_scale": 0.5,  # Text scale
            "text_padding": 5,  # Text padding
            "font": cv2.FONT_HERSHEY_SIMPLEX,  # Font
        }

        # Color manager
        self.color_manager = ColorManager()

    @abstractmethod
    def load_annotations(self) -> DatasetAnnotations:
        """Load annotation data (abstract method)."""
        pass

    def visualize(self) -> VisualizationResult:
        """Execute visualization pipeline."""
        result = VisualizationResult(success=False)

        try:
            # 1. Load annotation data
            annotations = self.load_annotations()

            # 2. Validate output directory (if save mode is enabled)
            if self.is_save:
                if not self.output_dir:
                    result.add_error("Save mode requires output_dir parameter")
                    return result
                self.file_ops.ensure_dir(self.output_dir)

            # 3. Process all images for visualization
            processed_count = 0
            for image_ann in annotations.images:
                success = self._visualize_single_image(image_ann)
                if success:
                    processed_count += 1
                elif self.strict_mode:
                    result.add_error(
                        f"Failed to visualize image: {image_ann.image_id}"
                    )
                    return result

            result.success = True
            result.message = (
                f"Successfully visualized {processed_count}/"
                f"{len(annotations.images)} images"
            )
            result.data = {"processed_count": processed_count}

        except Exception as e:
            result.add_error(f"Unexpected error during visualization: {e}")

        return result

    def _visualize_single_image(self, image_ann: ImageAnnotation) -> bool:
        """Visualize a single image."""
        try:
            # 1. Load image
            image_path = Path(image_ann.image_path)
            if not image_path.is_absolute():
                image_path = self.image_dir / image_ann.image_path

            if not image_path.exists():
                self._log_error(f"Image file not found: {image_path}")
                return False

            image = cv2.imread(str(image_path))
            if image is None:
                self._log_error(f"Failed to load image: {image_path}")
                return False

            # 2. Draw all objects
            for obj in image_ann.objects:
                self._draw_object(
                    image, obj, image_ann.width, image_ann.height
                )

            # 3. Display or save
            if self.is_show:
                window_name = f"Visualization - {image_ann.image_id}"
                cv2.imshow(window_name, image)
                key = cv2.waitKey(0)
                cv2.destroyWindow(window_name)

                # Handle keyboard input
                if key == ord("q") or key == 27:  # 'q' key or ESC
                    return False  # Stop visualization
                # Enter key or space key continue

            if self.is_save:
                output_file = (
                    self.output_dir / f"{image_ann.image_id}_visualized.jpg"
                )
                cv2.imwrite(
                    str(output_file), image, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                self._log_info(f"Saved visualization to: {output_file}")

            return True

        except Exception as e:
            self._log_error(
                f"Error visualizing image {image_ann.image_id}: {e}"
            )
            return False

    def _draw_object(
        self,
        image: np.ndarray,
        obj: ObjectAnnotation,
        img_width: int,
        img_height: int,
    ) -> None:
        """Draw a single object annotation."""
        # Get class color
        color = self.color_manager.get_color(obj.class_id)

        # Draw bounding box
        if obj.bbox is not None:
            self._draw_bbox(
                image, obj.bbox, color, obj.class_name, img_width, img_height
            )

        # Draw segmentation
        if obj.segmentation is not None:
            if obj.segmentation.has_rle():
                # RLE format (requires pycocotools)
                self._draw_rle_mask(
                    image, obj.segmentation.rle, color, img_width, img_height
                )
            else:
                # Polygon format
                self._draw_polygon(
                    image,
                    obj.segmentation,
                    color,
                    obj.class_name,
                    img_width,
                    img_height,
                )

    def _draw_bbox(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        color: Tuple[int, int, int],
        class_name: str,
        img_width: int,
        img_height: int,
    ) -> None:
        """Draw bounding box."""
        x1, y1, x2, y2 = bbox.xyxy(img_width, img_height)

        # Ensure coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Debug logging with types and image properties
        self.logger.debug(
            f"Drawing bbox: ({x1}, {y1}), ({x2}, {y2}), color: {color}, "
            f"image shape: {image.shape if image is not None else 'None'}"
        )
        if image is not None:
            self.logger.debug(
                f"Image dtype: {image.dtype}, flags: {image.flags}"
            )
            self.logger.debug(f"Image is writable: {image.flags['WRITEABLE']}")
        self.logger.debug(
            f"Coordinate types: x1={type(x1)}, y1={type(y1)}, "
            f"x2={type(x2)}, y2={type(y2)}"
        )
        self.logger.debug(f"Color type: {type(color)}, color value: {color}")

        # Draw rectangle
        try:
            pt1 = (x1, y1)
            pt2 = (x2, y2)
            self.logger.debug(
                f"Attempting cv2.rectangle with pt1={pt1}, "
                f"pt2={pt2}, color={color}"
            )
            cv2.rectangle(
                image, pt1, pt2, color, self.config["bbox_thickness"]
            )
            self.logger.debug("cv2.rectangle completed without exception")
        except BaseException as e:
            self.logger.error(f"cv2.rectangle raised exception: {e}")
            # Try with hardcoded values to see if OpenCV works at all
            try:
                self.logger.debug("Testing with hardcoded values")
                cv2.rectangle(image, (10, 10), (100, 100), (0, 255, 0), 2)
                self.logger.debug("Hardcoded rectangle succeeded")
            except BaseException as e2:
                self.logger.error(f"Hardcoded rectangle also failed: {e2}")
            raise

        # Draw class label
        self._draw_text(
            image, class_name, (x1, y1 - self.config["text_padding"]), color
        )

    def _draw_polygon(
        self,
        image: np.ndarray,
        segmentation: Segmentation,
        color: Tuple[int, int, int],
        class_name: str,
        img_width: int,
        img_height: int,
    ) -> None:
        """Draw polygon segmentation."""
        points = segmentation.points_abs(img_width, img_height)

        if len(points) < 2:
            return

        # Convert to numpy array
        points_np = np.array(points, dtype=np.int32)

        # Draw polygon fill (semi-transparent)
        overlay = image.copy()
        cv2.fillPoly(overlay, [points_np], color)
        cv2.addWeighted(
            overlay,
            self.config["seg_alpha"],
            image,
            1 - self.config["seg_alpha"],
            0,
            image,
        )

        # Draw polygon outline
        cv2.polylines(
            image, [points_np], True, color, self.config["seg_thickness"]
        )

        # Draw class label (use first point)
        if points:
            self._draw_text(image, class_name, points[0], color)

    def _draw_rle_mask(
        self,
        image: np.ndarray,
        rle: Dict,
        color: Tuple[int, int, int],
        img_width: int,
        img_height: int,
    ) -> None:
        """Draw RLE mask."""
        # Requires pycocotools support
        try:
            from pycocotools import mask as coco_mask
        except ImportError:
            self._log_error("pycocotools not installed, cannot draw RLE mask")
            return

        # Decode RLE to binary mask
        binary_mask = coco_mask.decode(rle)

        # Convert binary mask to color mask
        color_mask = np.zeros_like(image)
        for c in range(3):
            color_mask[:, :, c] = binary_mask * color[c]

        # Semi-transparent overlay
        overlay = image.copy()
        overlay = cv2.addWeighted(
            overlay,
            1 - self.config["seg_alpha"],
            color_mask,
            self.config["seg_alpha"],
            0,
        )

        # Copy overlay back to original image where mask is True
        np.copyto(image, overlay, where=binary_mask[:, :, None].astype(bool))

    def _draw_text(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        """Draw text label."""
        # Ensure position coordinates are integers
        x, y = int(position[0]), int(position[1])

        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            self.config["font"],
            self.config["text_scale"],
            self.config["text_thickness"],
        )

        # Ensure text dimensions are integers
        text_width = int(text_width)
        text_height = int(text_height)
        baseline = int(baseline)

        # Calculate background rectangle coordinates with clamping
        img_height, img_width = image.shape[:2]

        # Top-left corner
        x1 = x
        y1 = y - text_height - baseline
        # Bottom-right corner
        x2 = x + text_width
        y2 = y + baseline

        # Clamp coordinates to image boundaries
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))

        # Ensure coordinates are integers after clamping
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw background rectangle if valid
        if x1 < x2 and y1 < y2:
            bg_rect = ((x1, y1), (x2, y2))
            try:
                cv2.rectangle(
                    image, bg_rect, (0, 0, 0), -1
                )  # Black background
            except Exception as e:
                self.logger.warning(f"Failed to draw text background: {e}")
                # Continue without background

        # Draw text (adjust position if needed)
        text_y = y - baseline
        # Clamp text position
        text_y = max(baseline, min(text_y, img_height - 1))
        text_x = max(0, min(x, img_width - 1))

        cv2.putText(
            image,
            text,
            (text_x, text_y),
            self.config["font"],
            self.config["text_scale"],
            (255, 255, 255),  # White text
            self.config["text_thickness"],
            cv2.LINE_AA,
        )

    def _log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def _log_error(self, message: str) -> None:
        """Log error message and raise exception (strict mode)."""
        self.logger.error(message)
        if self.strict_mode:
            raise ValueError(message)

    def _log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
