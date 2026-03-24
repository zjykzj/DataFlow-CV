"""
Base visualization classes for DataFlow-CV.

Defines the abstract base class for all visualizers and supporting
data structures.
"""

import datetime
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from dataflow.label.models import (BoundingBox, DatasetAnnotations,
                                   ImageAnnotation, ObjectAnnotation,
                                   Segmentation)
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
    """Color manager that ensures consistent and unique colors for the same class."""

    def __init__(self, debug: bool = False) -> None:
        # Generate 1000 distinct colors using HSV space with ensured uniqueness
        self.predefined_colors = []
        self.debug = debug
        self._generate_unique_colors(1000)
        self.color_cache: Dict[int, Tuple[int, int, int]] = {}

    def _generate_unique_colors(self, num_colors: int) -> None:
        """Generate N unique colors using HSV space."""
        colors_set = set()

        # Try to generate colors with different hue, saturation, value combinations
        # We'll iterate through hue primarily, then adjust saturation/value when needed
        hue_step = max(1, 180 // (num_colors // 3))
        sat_step = max(1, 100 // (num_colors // 3))
        val_step = max(1, 100 // (num_colors // 3))

        for i in range(num_colors):
            # Primary: vary hue (0-179)
            hue = (i * hue_step) % 180

            # Secondary: vary saturation (100-200)
            saturation = 100 + ((i // 180) * sat_step) % 100

            # Tertiary: vary value (155-255)
            value = 155 + ((i // (180 * 100)) * val_step) % 100

            # Convert to BGR
            hsv_color = np.uint8([[[hue, saturation, value]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
            color = (
                int(bgr_color[0, 0, 0]),
                int(bgr_color[0, 0, 1]),
                int(bgr_color[0, 0, 2]),
            )

            # If color already exists (unlikely but possible due to rounding),
            # adjust saturation and value
            attempt = 0
            while color in colors_set and attempt < 10:
                # Try different adjustments
                saturation = (saturation + 23) % 100 + 100
                value = (value + 37) % 100 + 155
                hsv_color = np.uint8([[[hue, saturation, value]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
                color = (
                    int(bgr_color[0, 0, 0]),
                    int(bgr_color[0, 0, 1]),
                    int(bgr_color[0, 0, 2]),
                )
                attempt += 1

            if color in colors_set:
                # Last resort: skip this hue and try next
                hue = (hue + 1) % 180
                saturation = 100 + (i % 100)
                value = 155 + ((i + 13) % 100)
                hsv_color = np.uint8([[[hue, saturation, value]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
                color = (
                    int(bgr_color[0, 0, 0]),
                    int(bgr_color[0, 0, 1]),
                    int(bgr_color[0, 0, 2]),
                )

            colors_set.add(color)
            self.predefined_colors.append(color)

    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for a class ID."""
        if class_id in self.color_cache:
            return self.color_cache[class_id]

        # Use predefined colors for first N classes
        if class_id < len(self.predefined_colors):
            color = self.predefined_colors[class_id]
            if self.debug:
                print(
                    f"[ColorManager] class_id={class_id}, using predefined unique color {color}",
                    file=sys.stderr,
                )
        else:
            # For classes beyond predefined, use deterministic algorithm
            # with larger steps to avoid conflicts
            hue = (class_id * 127) % 180
            saturation = 100 + (class_id * 67) % 100  # 100-199
            value = 155 + (class_id * 37) % 100  # 155-254

            # Convert HSV to BGR
            hsv_color = np.uint8([[[hue, saturation, value]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
            color = (
                int(bgr_color[0, 0, 0]),
                int(bgr_color[0, 0, 1]),
                int(bgr_color[0, 0, 2]),
            )
            if self.debug:
                print(
                    f"[ColorManager] class_id={class_id}, generating HSV color {color}",
                    file=sys.stderr,
                )

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
        verbose: bool = False,  # New: verbose parameter
        logger: Optional[logging.Logger] = None,
    ):
        self.label_dir = Path(label_dir)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.is_show = is_show
        self.is_save = is_save
        self.strict_mode = strict_mode
        self.verbose = verbose  # Store verbose setting

        # Configure logger based on verbose
        if verbose and logger is None:
            from dataflow.util.logging_util import VerboseLoggingOperations

            logging_ops = VerboseLoggingOperations()
            self.logger = logging_ops.get_verbose_logger(
                name=f"visualize.{self.__class__.__name__.lower()}", verbose=verbose
            )
            self.progress_logger = logging_ops.create_progress_logger()
        else:
            self.logger = logger or logging.getLogger(__name__)
            self.progress_logger = None

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

        # Color manager's debug mode
        self.color_manager = ColorManager(debug=verbose)

        # Summary data collection
        self.summary_data = {
            "total_images": 0,
            "processed_images": 0,
            "failed_images": 0,
            "total_objects": 0,
            "start_time": None,
            "end_time": None,
        }

    @abstractmethod
    def load_annotations(self) -> DatasetAnnotations:
        """Load annotation data (abstract method)."""
        pass

    def visualize(self) -> VisualizationResult:
        """Execute visualization pipeline (enhanced version)."""
        start_time = datetime.datetime.now()
        self.summary_data["start_time"] = start_time

        if self.verbose:
            self.logger.debug(f"Starting visualization pipeline: {self.label_dir}")
            self.logger.debug(
                f"Configuration: show={self.is_show}, save={self.is_save}"
            )

        result = VisualizationResult(success=False)

        try:
            # 1. Load annotation data
            annotations = self.load_annotations()
            self.summary_data["total_images"] = len(annotations.images)
            self.summary_data["total_objects"] = sum(
                len(img.objects) for img in annotations.images
            )

            if self.verbose:
                self.logger.info(
                    f"Loaded annotations for {len(annotations.images)} images"
                )
                self.logger.debug(f"Category mapping: {annotations.categories}")

            # 2. Validate output directory (if save mode is enabled)
            if self.is_save:
                if not self.output_dir:
                    result.add_error("Save mode requires output_dir parameter")
                    return result
                self.file_ops.ensure_dir(self.output_dir)

            # 3. Process all images for visualization
            processed_count = 0
            for i, image_ann in enumerate(annotations.images):
                # Progress feedback
                if (
                    self.progress_logger and i % 10 == 0
                ):  # Update progress every 10 images
                    self._log_progress(
                        i, len(annotations.images), f"Processing {image_ann.image_id}"
                    )

                if self.verbose:
                    self.logger.debug(f"Processing image: {image_ann.image_id}")
                    self.logger.debug(
                        f"Image dimensions: {image_ann.width}x{image_ann.height}"
                    )
                    self.logger.debug(f"Number of objects: {len(image_ann.objects)}")

                success = self._visualize_single_image(image_ann)
                if success:
                    processed_count += 1
                    self.summary_data["processed_images"] = processed_count
                elif self.strict_mode:
                    result.add_error(f"Failed to visualize image: {image_ann.image_id}")
                    return result
                else:
                    self.summary_data["failed_images"] += 1

            result.success = True
            result.message = (
                f"Successfully visualized {processed_count}/"
                f"{len(annotations.images)} images"
            )
            result.data = {"processed_count": processed_count}

            # Record summary
            self.summary_data["end_time"] = datetime.datetime.now()
            if self.verbose:
                self._log_visualization_summary(result)

        except Exception as e:
            result.add_error(f"Unexpected error during visualization: {e}")
            if self.verbose:
                self.logger.exception("Visualization failed")

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
                self._draw_object(image, obj, image_ann.width, image_ann.height)

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
                output_file = self.output_dir / f"{image_ann.image_id}_visualized.jpg"
                cv2.imwrite(str(output_file), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                self._log_info(f"Saved visualization to: {output_file}")

            return True

        except Exception as e:
            self._log_error(f"Error visualizing image {image_ann.image_id}: {e}")
            return False

    def _draw_object(
        self,
        image: np.ndarray,
        obj: ObjectAnnotation,
        img_width: int,
        img_height: int,
    ) -> None:
        """Draw a single object annotation."""
        # Debug logging for color assignment
        self.logger.debug(
            f"Drawing object: class_id={obj.class_id}, class_name={obj.class_name}, "
            f"color={self.color_manager.get_color(obj.class_id)}"
        )
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
            self.logger.debug(f"Image dtype: {image.dtype}, flags: {image.flags}")
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
                f"Attempting cv2.rectangle with pt1={pt1}, " f"pt2={pt2}, color={color}"
            )
            cv2.rectangle(image, pt1, pt2, color, self.config["bbox_thickness"])
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
        cv2.polylines(image, [points_np], True, color, self.config["seg_thickness"])

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
            try:
                cv2.rectangle(
                    image, (x1, y1), (x2, y2), (0, 0, 0), -1
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

    def _log_visualization_summary(self, result: VisualizationResult):
        """Log visualization summary."""
        duration = self.summary_data["end_time"] - self.summary_data["start_time"]

        summary_data = {
            "Module Name": self.__class__.__name__,
            "Runtime": f"{duration.total_seconds():.2f} seconds",
            "Input Label Directory": str(self.label_dir),
            "Input Image Directory": str(self.image_dir),
            "Output Directory": str(self.output_dir) if self.output_dir else "None",
            "Image Statistics": {
                "Total": self.summary_data["total_images"],
                "Success": self.summary_data["processed_images"],
                "Failed": self.summary_data["failed_images"],
                "Success Rate": f"{(self.summary_data['processed_images']/self.summary_data['total_images']*100):.1f}%",
            },
            "Total Objects": self.summary_data["total_objects"],
            "Operation Status": "Success" if result.success else "Failed",
        }

        from dataflow.util.logging_util import VerboseLoggingOperations

        logging_ops = VerboseLoggingOperations()
        logging_ops.log_summary(
            self.logger, "Visualization Operation Summary", summary_data
        )

    def _log_progress(self, current: int, total: int, message: str = ""):
        """Log progress information."""
        if self.progress_logger and total > 0:
            percentage = (current / total) * 100
            progress_bar = self._create_progress_bar(current, total)
            self.progress_logger.info(f"{progress_bar} {percentage:.1f}% {message}")

    def _create_progress_bar(self, current: int, total: int, width: int = 40) -> str:
        """Create text progress bar."""
        if total == 0:
            return "[>······································]"

        filled = int(width * current / total)
        bar = "[" + "=" * filled + ">" + "." * (width - filled - 1) + "]"
        return bar

    def _log_color_info(self, class_id: int, color: Tuple[int, int, int]):
        """Log color assignment information (verbose mode only)."""
        if self.verbose:
            class_name = self._get_class_name(class_id)
            self.logger.debug(
                f"Color assignment - Class ID: {class_id}, Name: {class_name}, "
                f"Color (BGR): {color}"
            )

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID (to be implemented by subclasses)."""
        # Default implementation, subclasses should override
        return f"class_{class_id}"
