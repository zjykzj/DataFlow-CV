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

from dataflow.util import FileOperations
from dataflow.label.base import BaseAnnotationHandler
from dataflow.label.models import ImageAnnotation


@dataclass
class RenderAnnotation:
    """Annotation data prepared for rendering.

    All coordinates are in absolute pixel integers, ready for OpenCV drawing.
    """

    class_name: str
    class_id: int
    bbox: Optional[Tuple[int, int, int, int]] = None  # [x1, y1, x2, y2] absolute px
    polygon: Optional[List[Tuple[int, int]]] = None  # [(x, y), ...] absolute px
    rle: Optional[Dict] = None  # RLE mask data (COCO specific)


@dataclass
class RenderData:
    """Rendering data for a single image."""

    annotations: List[RenderAnnotation]
    image_width: int
    image_height: int


@dataclass
class VisualizationResult:
    """Visualization processing result."""

    success: bool
    data: Optional[Any] = None
    message: str = ""
    errors: List[str] = field(default_factory=list)
    log_file_path: Optional[str] = None  # Log file path when verbose=True

    def add_error(self, error: str) -> None:
        """Add an error message to the result."""
        self.errors.append(error)
        self.success = False


class ColorManager:
    """Color manager that ensures consistent and unique colors for the same class.

    Uses golden ratio hue spacing with high saturation for maximum visual
    distinctiveness. Adjacent class IDs are spaced ~69° apart in hue, and
    saturation/value are varied to create clearly distinguishable colors.
    """

    # Golden ratio conjugate: 1 - φ ≈ 0.382, provides optimal hue spacing
    _GOLDEN_RATIO = 0.3819660112501051

    def __init__(self, debug: bool = False) -> None:
        self.predefined_colors = []
        self.debug = debug
        self._generate_unique_colors(1000)
        self.color_cache: Dict[int, Tuple[int, int, int]] = {}

    def _generate_unique_colors(self, num_colors: int) -> None:
        """Generate N visually distinct colors using golden ratio hue spacing.

        Saturation is kept high (200-255, ≥78%) for vivid, easily distinguishable
        colors. Value varies between 180-240 to provide brightness contrast without
        becoming too dark for text readability.
        """
        colors_set = set()

        for i in range(num_colors):
            # Golden ratio spacing maximizes minimum hue distance
            hue = int((i * self._GOLDEN_RATIO * 180) % 180)
            # High saturation range [200, 255] for vivid colors
            saturation = 200 + (i * 43) % 56
            # Varied value range [180, 240] for brightness contrast
            value = 180 + (i * 67) % 61

            hsv_color = np.uint8([[[hue, saturation, value]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
            color = (
                int(bgr_color[0, 0, 0]),
                int(bgr_color[0, 0, 1]),
                int(bgr_color[0, 0, 2]),
            )

            # Resolve collisions (rare with golden ratio spacing)
            attempt = 0
            while color in colors_set and attempt < 20:
                hue = (hue + 17) % 180
                saturation = min(255, saturation + 20)
                if saturation >= 255:
                    saturation = 200
                    value = (value + 30) % 61 + 180
                hsv_color = np.uint8([[[hue, saturation, value]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
                color = (
                    int(bgr_color[0, 0, 0]),
                    int(bgr_color[0, 0, 1]),
                    int(bgr_color[0, 0, 2]),
                )
                attempt += 1

            colors_set.add(color)
            self.predefined_colors.append(color)

    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for a class ID."""
        if class_id in self.color_cache:
            return self.color_cache[class_id]

        if class_id < len(self.predefined_colors):
            color = self.predefined_colors[class_id]
            if self.debug:
                print(
                    f"[ColorManager] class_id={class_id}, using predefined unique color {color}",
                    file=sys.stderr,
                )
        else:
            # Fallback for class_id ≥ 1000: deterministic high-saturation colors
            hue = (class_id * 127) % 180
            saturation = 180 + (class_id * 43) % 76   # [180, 255]
            value = 160 + (class_id * 67) % 96        # [160, 255]

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
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
        log_file_path: Optional[str] = None,
    ):
        self.label_dir = Path(label_dir)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.is_show = is_show
        self.is_save = is_save
        self.strict_mode = strict_mode
        self.verbose = verbose

        if log_file_path is not None:
            self.log_file_path = log_file_path
            self.logger = logger or logging.getLogger(__name__)
            self.progress_logger = None
        elif verbose and logger is None:
            from dataflow.util.logging_util import VerboseLoggingOperations

            logging_ops = VerboseLoggingOperations()
            self.logger, self.log_file_path = logging_ops.get_verbose_logger(
                name=f"visualize.{self.__class__.__name__.lower()}", verbose=verbose
            )
            self.progress_logger = logging_ops.create_progress_logger()
        else:
            self.logger = logger or logging.getLogger(__name__)
            self.progress_logger = None
            self.log_file_path = None

        self.file_ops = FileOperations(logger=self.logger)

        self.config = {
            "bbox_thickness": 2,
            "seg_thickness": 1,
            "seg_alpha": 0.3,
            "text_thickness": 1,
            "text_scale": 0.5,
            "text_padding": 5,
            "font": cv2.FONT_HERSHEY_SIMPLEX,
        }

        self.color_manager = ColorManager(debug=verbose)

        self.summary_data = {
            "total_images": 0,
            "processed_images": 0,
            "failed_images": 0,
            "total_objects": 0,
            "start_time": None,
            "end_time": None,
        }

        self._window_positioned = False

    @abstractmethod
    def _create_handler(self) -> BaseAnnotationHandler:
        """Create and return the format-specific Label handler instance.

        The handler will be used to stream ``ImageAnnotation`` objects via
        its ``iter_images()`` method.
        """
        pass

    @abstractmethod
    def _convert_to_render_data(self, image_ann: ImageAnnotation) -> RenderData:
        """Convert a single ImageAnnotation to RenderData.

        This is called per-image during the streaming loop. Each concrete
        visualizer implements format-specific coordinate conversion.

        Args:
            image_ann: Single ImageAnnotation with format-native coordinates.

        Returns:
            RenderData with absolute-pixel annotations ready for drawing.
        """
        pass

    def visualize(self) -> VisualizationResult:
        """Execute visualization pipeline using streaming iteration.

        Images are loaded, converted, and displayed one at a time via
        ``handler.iter_images()``. This provides low first-image latency
        and low memory usage.
        """
        start_time = datetime.datetime.now()
        self.summary_data["start_time"] = start_time

        if self.verbose:
            self.logger.debug(
                f"Starting visualization pipeline: {self.label_dir}"
            )

        result = VisualizationResult(
            success=False, log_file_path=self.log_file_path
        )

        processed_count = 0
        image_index = 0
        total_objects = 0

        try:
            # 1. Validate output directory
            if self.is_save:
                if not self.output_dir:
                    error_msg = "Save mode requires output_dir parameter"
                    result.add_error(error_msg)
                    result.message = error_msg
                    return result
                self.file_ops.ensure_dir(self.output_dir)

            # 2. Create handler and obtain streaming iterator
            handler = self._create_handler()
            image_iter = handler.iter_images()

            # 3. Process images one at a time (streaming)
            user_interrupted = False

            for image_ann in image_iter:
                image_index += 1

                if self.progress_logger and image_index % 10 == 0:
                    self._log_progress(
                        image_index,
                        message=f"Processing {image_ann.image_path}",
                    )

                if self.verbose:
                    self.logger.debug(
                        f"Processing image: {image_ann.image_path}"
                    )

                # Convert to render data (per-image)
                render_data = self._convert_to_render_data(image_ann)
                total_objects += len(render_data.annotations)

                # Visualize
                success = self._visualize_single_image(
                    image_ann.image_path, render_data
                )
                if success is None:
                    user_interrupted = True
                    break
                elif success:
                    processed_count += 1
                    self.summary_data["processed_images"] = processed_count
                else:
                    self.summary_data["failed_images"] += 1

            # Update summary data with actual counts
            self.summary_data["total_images"] = image_index
            self.summary_data["total_objects"] = total_objects

            if user_interrupted:
                result.success = True
                result.message = (
                    f"Visualization interrupted by user after "
                    f"{processed_count} images. "
                    f"{self.summary_data['failed_images']} failed out of "
                    f"{image_index} images"
                )
                result.data = {
                    "processed_count": processed_count,
                    "interrupted": True,
                }
            else:
                result.success = True
                failed_count = self.summary_data["failed_images"]
                result.message = (
                    f"Visualization completed: {processed_count} successful, "
                    f"{failed_count} failed out of {image_index} images"
                )
                result.data = {"processed_count": processed_count}

            self.summary_data["end_time"] = datetime.datetime.now()
            if self.verbose:
                self._log_visualization_summary(result)

        except ValueError as e:
            # Handler structural errors or strict-mode parsing errors
            # during iteration. Partial results before the error are valid.
            error_msg = str(e)
            result.add_error(error_msg)
            self.summary_data["total_images"] = image_index
            self.summary_data["total_objects"] = total_objects
            if processed_count > 0:
                result.success = True
                result.message = (
                    f"Visualization completed with partial results: "
                    f"{processed_count} processed. Error: {error_msg}"
                )
                result.data = {
                    "processed_count": processed_count,
                    "partial": True,
                }
            else:
                result.message = error_msg
            self.summary_data["end_time"] = datetime.datetime.now()

        except Exception as e:
            error_msg = str(e)
            result.add_error(error_msg)
            result.message = error_msg
            if self.verbose:
                self.logger.exception("Visualization failed")
        finally:
            if self.is_show:
                try:
                    cv2.destroyWindow("DataFlow-CV Visualization")
                except Exception:
                    pass

        return result

    def _visualize_single_image(
        self, image_path_str: str, render_data: RenderData
    ) -> Optional[bool]:
        """Visualize a single image using pre-computed RenderData."""
        try:
            # 1. Resolve and load image
            image_path = Path(image_path_str)
            if not image_path.is_absolute():
                image_path = self.image_dir / image_path_str

            if not image_path.exists():
                self._log_warning(f"Image file not found: {image_path}")
                return False

            image = cv2.imread(str(image_path))
            if image is None:
                self._log_warning(f"Failed to load image: {image_path}")
                return False

            # 2. Draw all render annotations
            for render_ann in render_data.annotations:
                self._draw_render_annotation(image, render_ann)

            # 3. Display or save
            if self.is_show:
                try:
                    window_name = "DataFlow-CV Visualization"
                    window_flags = cv2.WINDOW_NORMAL
                    try:
                        window_flags |= cv2.WINDOW_KEEPRATIO
                    except AttributeError:
                        pass
                    cv2.namedWindow(window_name, window_flags)

                    if not self._window_positioned:
                        cv2.moveWindow(window_name, 100, 100)
                        self._window_positioned = True

                    h, w = image.shape[:2]
                    MAX_DISPLAY_W, MAX_DISPLAY_H = 1920, 1080
                    if w <= MAX_DISPLAY_W and h <= MAX_DISPLAY_H:
                        cv2.resizeWindow(window_name, w, h)
                    else:
                        scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h)
                        cv2.resizeWindow(window_name, int(w * scale), int(h * scale))

                    cv2.imshow(window_name, image)
                    key = cv2.waitKey(0)

                    if key == ord("q") or key == 27:
                        return None
                except Exception as e:
                    self._log_warning(f"Failed to display visualization window: {e}")

            if self.is_save:
                output_file = (
                    self.output_dir
                    / f"{Path(image_path_str).stem}_visualized.jpg"
                )
                cv2.imwrite(
                    str(output_file), image, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                self._log_info(f"Saved visualization to: {output_file}")

            return True

        except Exception as e:
            self._log_error(f"Error visualizing image {image_path_str}: {e}")
            return False

    def _draw_render_annotation(
        self, image: np.ndarray, render_ann: RenderAnnotation
    ) -> None:
        """Draw a single RenderAnnotation onto the image."""
        color = self.color_manager.get_color(render_ann.class_id)

        self.logger.debug(
            f"Drawing object: class_id={render_ann.class_id}, "
            f"class_name={render_ann.class_name}, color={color}"
        )

        # Determine label position (prefer bbox, fall back to polygon first point)
        label_pos = None
        if render_ann.bbox is not None:
            x1, y1, x2, y2 = render_ann.bbox
            label_pos = (x1, y1 - self.config["text_padding"])

        if render_ann.bbox is not None:
            self._draw_bbox(image, render_ann.bbox, color)

        if render_ann.polygon is not None:
            if label_pos is None and render_ann.polygon:
                label_pos = render_ann.polygon[0]
            self._draw_polygon(image, render_ann.polygon, color)

        if render_ann.rle is not None:
            self._draw_rle_mask(image, render_ann.rle, color)

        if label_pos is not None:
            self._draw_text(image, render_ann.class_name, label_pos, color)

    def _draw_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
    ) -> None:
        """Draw bounding box from absolute pixel coordinates [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox

        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.config["bbox_thickness"])

    def _draw_polygon(
        self,
        image: np.ndarray,
        polygon: List[Tuple[int, int]],
        color: Tuple[int, int, int],
    ) -> None:
        """Draw polygon from absolute pixel points."""
        if len(polygon) < 3:
            return

        points_np = np.array(polygon, dtype=np.int32)

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

    def _draw_rle_mask(
        self,
        image: np.ndarray,
        rle: Dict,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw RLE mask."""
        try:
            from pycocotools import mask as coco_mask
        except ImportError:
            self._log_warning("pycocotools not installed, cannot draw RLE mask")
            return

        rle_dict = dict(rle)
        if "counts" in rle_dict and isinstance(rle_dict["counts"], str):
            rle_dict["counts"] = rle_dict["counts"].encode("latin1")

        binary_mask = coco_mask.decode(rle_dict)

        color_mask = np.zeros_like(image)
        for c in range(3):
            color_mask[:, :, c] = binary_mask * color[c]

        overlay = image.copy()
        overlay = cv2.addWeighted(
            overlay,
            1 - self.config["seg_alpha"],
            color_mask,
            self.config["seg_alpha"],
            0,
        )

        np.copyto(image, overlay, where=binary_mask[:, :, None].astype(bool))

    def _draw_text(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        """Draw text label.

        Args:
            position: (x, y) baseline position for the text.
        """
        x, y = int(position[0]), int(position[1])

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            self.config["font"],
            self.config["text_scale"],
            self.config["text_thickness"],
        )

        text_width = int(text_width)
        text_height = int(text_height)
        baseline = int(baseline)

        img_height, img_width = image.shape[:2]

        # Background rectangle: covers from text top (y - text_height + baseline)
        # to text bottom (y + baseline)
        x1 = x
        y1 = y - text_height + baseline
        x2 = x + text_width
        y2 = y + baseline

        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))

        if x1 < x2 and y1 < y2:
            try:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
            except Exception as e:
                self.logger.warning(f"Failed to draw text background: {e}")

        # Text baseline is at position.y
        text_y = y
        text_y = max(baseline, min(text_y, img_height - 1))
        text_x = max(0, min(x, img_width - 1))

        cv2.putText(
            image,
            text,
            (text_x, text_y),
            self.config["font"],
            self.config["text_scale"],
            (255, 255, 255),
            self.config["text_thickness"],
            cv2.LINE_AA,
        )

    def _log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def _log_error(self, message: str) -> None:
        """Log error message and raise exception (strict mode).

        Image-related errors are always downgraded to warnings
        regardless of ``strict_mode``.
        """
        from dataflow.util.logging_util import (
            detect_image_error,
            logging_error_or_raise,
        )

        logging_error_or_raise(
            message,
            self.logger,
            self.strict_mode,
            is_image_error=detect_image_error(message),
        )

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

    def _log_progress(self, current: int, message: str = ""):
        """Log progress information (counter-based for streaming).

        Uses a counter format since the total image count is unknown
        until the iterator exhausts.
        """
        if self.progress_logger:
            failed = self.summary_data.get("failed_images", 0)
            tail = f" - {message}" if message else ""
            self.progress_logger.info(
                f"Processed {current} images, {failed} failed{tail}"
            )

    def _create_progress_bar(self, current: int, total: int, width: int = 40) -> str:
        """Create text progress bar (retained for compatibility).

        No longer used by the streaming pipeline — kept for subclasses
        or external consumers that may reference it.
        """
        if total == 0:
            return "[>······································]"

        filled = int(width * current / total)
        if filled >= width:
            bar = "[" + "=" * width + "]"
        else:
            bar = "[" + "=" * filled + ">" + "." * (width - filled - 1) + "]"
        return bar
