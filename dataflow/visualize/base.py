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
    """Color manager that ensures consistent and unique colors for the same class."""

    def __init__(self, debug: bool = False) -> None:
        self.predefined_colors = []
        self.debug = debug
        self._generate_unique_colors(1000)
        self.color_cache: Dict[int, Tuple[int, int, int]] = {}

    def _generate_unique_colors(self, num_colors: int) -> None:
        """Generate N unique colors using HSV space."""
        colors_set = set()

        hue_step = max(1, 180 // max(1, num_colors // 3))
        sat_step = max(1, 100 // max(1, num_colors // 3))
        val_step = max(1, 100 // max(1, num_colors // 3))

        sat_range = 101
        val_divisor = 180 * max(1, sat_range // sat_step)

        for i in range(num_colors):
            hue = (i * hue_step) % 180
            saturation = 100 + ((i // 180) * sat_step) % sat_range
            value = 155 + ((i // val_divisor) * val_step) % sat_range

            hsv_color = np.uint8([[[hue, saturation, value]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
            color = (
                int(bgr_color[0, 0, 0]),
                int(bgr_color[0, 0, 1]),
                int(bgr_color[0, 0, 2]),
            )

            attempt = 0
            while color in colors_set and attempt < 10:
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

        if class_id < len(self.predefined_colors):
            color = self.predefined_colors[class_id]
            if self.debug:
                print(
                    f"[ColorManager] class_id={class_id}, using predefined unique color {color}",
                    file=sys.stderr,
                )
        else:
            hue = (class_id * 127) % 180
            saturation = 100 + (class_id * 67) % 100
            value = 155 + (class_id * 37) % 100

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
    def load_annotations(self) -> Dict[str, RenderData]:
        """Load annotation data and convert to RenderData (abstract method).

        Returns:
            Dict mapping image_path → RenderData for that image
        """
        pass

    def visualize(self) -> VisualizationResult:
        """Execute visualization pipeline."""
        start_time = datetime.datetime.now()
        self.summary_data["start_time"] = start_time

        if self.verbose:
            self.logger.debug(f"Starting visualization pipeline: {self.label_dir}")

        result = VisualizationResult(success=False, log_file_path=self.log_file_path)

        try:
            # 1. Load and convert all annotations to render data
            render_data_map = self.load_annotations()
            self.summary_data["total_images"] = len(render_data_map)
            self.summary_data["total_objects"] = sum(
                len(rd.annotations) for rd in render_data_map.values()
            )

            if self.verbose:
                self.logger.info(
                    f"Loaded annotations for {len(render_data_map)} images"
                )

            # 2. Validate output directory
            if self.is_save:
                if not self.output_dir:
                    error_msg = "Save mode requires output_dir parameter"
                    result.add_error(error_msg)
                    result.message = error_msg
                    return result
                self.file_ops.ensure_dir(self.output_dir)

            # 3. Process all images
            processed_count = 0
            user_interrupted = False
            image_paths = list(render_data_map.keys())

            for i, image_path_str in enumerate(image_paths):
                render_data = render_data_map[image_path_str]

                if self.progress_logger and i % 10 == 0:
                    self._log_progress(
                        i, len(image_paths), f"Processing {image_path_str}"
                    )

                if self.verbose:
                    self.logger.debug(f"Processing image: {image_path_str}")

                success = self._visualize_single_image(image_path_str, render_data)
                if success is None:
                    user_interrupted = True
                    break
                elif success:
                    processed_count += 1
                    self.summary_data["processed_images"] = processed_count
                else:
                    self.summary_data["failed_images"] += 1

            if user_interrupted:
                result.success = True
                result.message = (
                    f"Visualization interrupted by user after {processed_count} images. "
                    f"{self.summary_data['failed_images']} failed out of {len(render_data_map)} images"
                )
                result.data = {"processed_count": processed_count, "interrupted": True}
            else:
                result.success = True
                failed_count = self.summary_data["failed_images"]
                result.message = (
                    f"Visualization completed: {processed_count} successful, "
                    f"{failed_count} failed out of {len(render_data_map)} images"
                )
                result.data = {"processed_count": processed_count}

            self.summary_data["end_time"] = datetime.datetime.now()
            if self.verbose:
                self._log_visualization_summary(result)

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

        if render_ann.bbox is not None:
            self._draw_bbox(image, render_ann.bbox, color, render_ann.class_name)

        if render_ann.polygon is not None:
            self._draw_polygon(image, render_ann.polygon, color, render_ann.class_name)

        if render_ann.rle is not None:
            self._draw_rle_mask(image, render_ann.rle, color)

    def _draw_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        class_name: str,
    ) -> None:
        """Draw bounding box from absolute pixel coordinates [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox

        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.config["bbox_thickness"])

        # Draw class label
        self._draw_text(image, class_name, (x1, y1 - self.config["text_padding"]), color)

    def _draw_polygon(
        self,
        image: np.ndarray,
        polygon: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        class_name: str,
    ) -> None:
        """Draw polygon from absolute pixel points."""
        if len(polygon) < 2:
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

        # Draw class label
        if polygon:
            self._draw_text(image, class_name, polygon[0], color)

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
        """Draw text label."""
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

        x1 = x
        y1 = y - text_height - baseline
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

        text_y = y - baseline
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
        """Log error message and raise exception (strict mode)."""
        self.logger.error(message)
        if self.strict_mode:
            _msg = message.lower()
            is_image_error = (
                "image" in _msg
                and any(
                    kw in _msg
                    for kw in (
                        "not found",
                        "failed to load",
                        "failed to read",
                        "invalid",
                        "error getting",
                        "no corresponding",
                        "does not exist",
                    )
                )
            )
            if not is_image_error:
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
        if filled >= width:
            bar = "[" + "=" * width + "]"
        else:
            bar = "[" + "=" * filled + ">" + "." * (width - filled - 1) + "]"
        return bar
