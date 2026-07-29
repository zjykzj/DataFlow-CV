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
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

from dataflow.label.base import BaseAnnotationHandler
from dataflow.label.models import ImageAnnotation

# Arrow key codes for cv2.waitKeyEx() — cross-platform tuples.
# Linux (waitKey):             81, 82, 83, 84
# Linux (waitKeyEx, X11):      65361, 65362, 65363, 65364
# Windows (waitKeyEx):         2424832, 2490368, 2555904, 2621440
_ARROW_LEFT = (81, 65361, 2424832)
_ARROW_UP = (82, 65362, 2490368)
_ARROW_RIGHT = (83, 65363, 2555904)
_ARROW_DOWN = (84, 65364, 2621440)


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


@dataclass
class VisualizationResult:
    """Visualization processing result."""

    success: bool
    data: Optional[Any] = None
    message: str = ""
    errors: List[str] = field(default_factory=list)
    log_path: Optional[str] = None  # Log file path when verbose=True

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
        log_config: Optional[Any] = None,
    ):
        self.label_dir = Path(label_dir)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.is_show = is_show
        self.is_save = is_save

        # Configure logger via unified LogManager
        from dataflow.util.logging import LogConfig, LogManager

        if log_config is None:
            log_config = LogConfig(
                name=f"visualize.{self.__class__.__name__.lower()}"
            )
        self._log_manager = LogManager(log_config)
        self.logger = self._log_manager.logger

        self.config = {
            "bbox_thickness": 2,
            "seg_thickness": 1,
            "seg_alpha": 0.3,
            "text_thickness": 1,
            "text_scale": 0.5,
            "text_padding": 5,
            "font": cv2.FONT_HERSHEY_SIMPLEX,
        }

        self.color_manager = ColorManager(debug=log_config.verbose if log_config else False)

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

        result = VisualizationResult(
            success=False, log_path=self._log_manager.log_path
        )

        total_objects = 0
        # Navigation state (declared here so ValueError handler can read them
        # even when the exception fires before the main loop starts).
        buffer: List[Tuple[str, RenderData]] = []
        displayed_indices: Set[int] = set()

        try:
            # 1. Validate output directory
            if self.is_save:
                if not self.output_dir:
                    error_msg = "Save mode requires output_dir parameter"
                    result.add_error(error_msg)
                    result.message = error_msg
                    return result
                self.output_dir.mkdir(parents=True, exist_ok=True)

            # 2. Create handler and obtain streaming iterator
            handler = self._create_handler()
            image_iter = handler.iter_images()

            # 2a. Log header with format and path info
            format_name = self.__class__.__name__.replace("Visualizer", "")
            from dataflow.visualize.log_templates import format_viz_header

            self.logger.info(
                format_viz_header(
                    format_name=format_name,
                    label_dir=str(self.label_dir),
                    image_dir=str(self.image_dir),
                    is_show=self.is_show,
                    is_save=self.is_save,
                    output_dir=str(self.output_dir) if self.output_dir else None,
                )
            )

            # 3. Visualization pipeline — dispatches between interactive
            #    (buffered, bidirectional navigation) and batch (streaming)
            #    based on is_show.
            if self.is_show:
                # ── Interactive mode: buffered bidirectional navigation ──
                #
                # Images are fetched on demand and buffered as (image_path,
                # RenderData) tuples.  When navigating backward the RenderData
                # is reused (pixels re-loaded from disk).  Advancing past the
                # buffer end fetches the next item from the iterator.
                current_idx = 0
                iterator_exhausted = False
                user_interrupted = False
                hints_shown = False

                image_iter = handler.iter_images()

                # Prime the buffer with the first image
                try:
                    image_ann = next(image_iter)
                except StopIteration:
                    iterator_exhausted = True

                if not iterator_exhausted:
                    render_data = self._convert_to_render_data(image_ann)
                    buffer.append((image_ann.image_path, render_data))
                    total_objects += len(render_data.annotations)
                    if self._log_manager.log_path is not None:
                        self.logger.debug(
                            f"Buffered image: {image_ann.image_path}"
                            f" ({len(render_data.annotations)} objects)"
                        )

                # Navigation loop
                while buffer and 0 <= current_idx < len(buffer):
                    image_path_str, render_data = buffer[current_idx]
                    displayed_idx = current_idx

                    action = self._visualize_single_image(
                        image_path_str, render_data
                    )

                    if action is None:
                        # Image load failure — advance forward
                        self.summary_data["failed_images"] += 1
                        if current_idx < len(buffer) - 1:
                            current_idx += 1
                        elif not iterator_exhausted:
                            try:
                                image_ann = next(image_iter)
                            except StopIteration:
                                iterator_exhausted = True
                            if not iterator_exhausted:
                                render_data = self._convert_to_render_data(
                                    image_ann
                                )
                                buffer.append((image_ann.image_path, render_data))
                                total_objects += len(render_data.annotations)
                                current_idx = len(buffer) - 1
                                if self._log_manager.log_path is not None:
                                    self.logger.debug(
                                        f"Buffered image: {image_ann.image_path}"
                                        f" ({len(render_data.annotations)} objects)"
                                    )
                            else:
                                break
                        else:
                            break
                        continue

                    displayed_indices.add(displayed_idx)

                    if action == "quit":
                        user_interrupted = True
                        break
                    elif action == "prev":
                        if current_idx > 0:
                            current_idx -= 1
                    elif action == "next":
                        current_idx += 1
                        if current_idx >= len(buffer):
                            if not iterator_exhausted:
                                try:
                                    image_ann = next(image_iter)
                                except StopIteration:
                                    iterator_exhausted = True
                                if not iterator_exhausted:
                                    render_data = self._convert_to_render_data(
                                        image_ann
                                    )
                                    buffer.append(
                                        (image_ann.image_path, render_data)
                                    )
                                    total_objects += len(render_data.annotations)
                                    if self._log_manager.log_path is not None:
                                        self.logger.debug(
                                            f"Buffered image: {image_ann.image_path}"
                                            f" ({len(render_data.annotations)} objects)"
                                        )
                                else:
                                    current_idx = len(buffer) - 1
                            else:
                                current_idx = len(buffer) - 1

                    if not hints_shown:
                        self.logger.info(
                            "Controls: Enter/Space/→/↓ = next"
                            " | ←/↑ = previous | q/ESC = quit"
                        )
                        hints_shown = True

                    self._update_window_title(
                        current_idx, len(buffer), image_path_str
                    )
                    self.summary_data["processed_images"] = len(displayed_indices)

                self.summary_data["total_images"] = len(buffer)
                self.summary_data["total_objects"] = total_objects

                if user_interrupted:
                    result.success = True
                    result.message = (
                        f"Visualization interrupted by user after "
                        f"{len(displayed_indices)} unique images displayed. "
                        f"{self.summary_data['failed_images']} failed out of "
                        f"{len(buffer)} images"
                    )
                    result.data = {
                        "processed_count": len(displayed_indices),
                        "interrupted": True,
                    }
                else:
                    result.success = True
                    failed_count = self.summary_data["failed_images"]
                    result.message = (
                        f"Visualization completed: {len(displayed_indices)} rendered, "
                        f"{failed_count} failed out of {len(buffer)} images"
                    )
                    result.data = {"processed_count": len(displayed_indices)}
            else:
                # ── Batch mode: streaming forward-only (no display) ──
                processed_count = 0
                image_index = 0

                for image_ann in handler.iter_images():
                    image_index += 1

                    render_data = self._convert_to_render_data(image_ann)
                    total_objects += len(render_data.annotations)

                    if image_index % 10 == 0 or self._log_manager.log_path is not None:
                        self._log_progress(
                            image_index,
                            message=str(image_ann.image_path),
                            n_objects=len(render_data.annotations),
                        )

                    if self._log_manager.log_path is not None:
                        self.logger.debug(
                            f"Processing image: {image_ann.image_path}"
                            f" ({len(render_data.annotations)} objects)"
                        )

                    action = self._visualize_single_image(
                        image_ann.image_path, render_data
                    )
                    if action == "quit":
                        break
                    elif action is not None:
                        processed_count += 1
                        self.summary_data["processed_images"] = processed_count
                    else:
                        self.summary_data["failed_images"] += 1

                self.summary_data["total_images"] = image_index
                self.summary_data["total_objects"] = total_objects

                result.success = True
                failed_count = self.summary_data["failed_images"]
                result.message = (
                    f"Visualization completed: {processed_count} successful, "
                    f"{failed_count} failed out of {image_index} images"
                )
                result.data = {"processed_count": processed_count}

            self.summary_data["end_time"] = datetime.datetime.now()
            self._log_visualization_summary(result)

        except ValueError as e:
            # Handler structural errors or strict-mode parsing errors
            # during iteration. Partial results before the error are valid.
            error_msg = str(e)
            result.add_error(error_msg)
            self.summary_data["total_images"] = len(buffer)
            self.summary_data["total_objects"] = total_objects
            if len(displayed_indices) > 0:
                result.success = True
                result.message = (
                    f"Visualization completed with partial results: "
                    f"{len(displayed_indices)} displayed. Error: {error_msg}"
                )
                result.data = {
                    "processed_count": len(displayed_indices),
                    "partial": True,
                }
            else:
                result.message = error_msg
            self.summary_data["end_time"] = datetime.datetime.now()

        except Exception as e:
            error_msg = str(e)
            result.add_error(error_msg)
            result.message = error_msg
            if self._log_manager.log_path is not None:
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
    ) -> Optional[str]:
        """Visualize a single image using pre-computed RenderData.

        Returns:
            ``"next"`` — advance to next image (Enter/Space/→/↓/any key).
            ``"prev"`` — go back to previous image (←/↑).
            ``"quit"`` — user pressed q/ESC, stop visualization.
            ``None`` — image file not found or failed to load.
        """
        try:
            # 1. Resolve and load image
            image_path = Path(image_path_str)
            if not image_path.is_absolute():
                image_path = self.image_dir / image_path_str

            if not image_path.exists():
                self._log_warning(f"Image file not found: {image_path}")
                return None

            image = cv2.imread(str(image_path))
            if image is None:
                self._log_warning(f"Failed to load image: {image_path}")
                return None

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
                    key = cv2.waitKeyEx(0)

                    if key == ord("q") or key == 27:  # q or ESC — quit
                        return "quit"
                    if key in _ARROW_LEFT or key in _ARROW_UP:  # ← or ↑ — prev
                        return "prev"
                    # →, ↓, Enter, Space, or any other key — advance forward
                    return "next"
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

            return "next"

        except Exception as e:
            self._log_error(f"Error visualizing image {image_path_str}: {e}")
            return None

    def _draw_render_annotation(
        self, image: np.ndarray, render_ann: RenderAnnotation
    ) -> None:
        """Draw a single RenderAnnotation onto the image."""
        color = self.color_manager.get_color(render_ann.class_id)

        self.logger.debug(
            f"Drawing object: class_id={render_ann.class_id}, "
            f"class_name={render_ann.class_name}, color={color}"
        )

        # Determine label position (above bbox top-left;
        # fall back to polygon first point only if no bbox could be computed)
        label_pos = None
        label_bbox = None
        if render_ann.bbox is not None:
            x1, y1, x2, y2 = render_ann.bbox
            label_bbox = (x1, y1, x2, y2)
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
            self._draw_text(
                image, render_ann.class_name, label_pos, color, bbox=label_bbox
            )

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
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """Draw text label.

        Args:
            position: (x, y) baseline position for the text (above bbox top-left).
            bbox: Optional bbox (x1, y1, x2, y2) for edge-flip logic.
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

        # Edge flip: if text extends above image top, flip inside bbox
        text_top = y - text_height + baseline
        if bbox is not None and text_top < 0:
            _, y1, _, _ = bbox
            y = y1 + text_height + self.config["text_padding"]

        # Background rectangle: covers from text top (y - text_height + baseline)
        # to text bottom (y + baseline)
        x1 = x
        y1 = y - text_height + baseline
        x2 = x + text_width
        y2_rect = y + baseline

        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2_rect = max(0, min(y2_rect, img_height - 1))

        if x1 < x2 and y1 < y2_rect:
            try:
                cv2.rectangle(image, (x1, y1), (x2, y2_rect), color, -1)
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

    @staticmethod
    def _compute_bbox_from_polygon(
        polygon: List[Tuple[int, int]],
    ) -> Tuple[int, int, int, int]:
        """Compute bbox [x1, y1, x2, y2] from polygon points."""
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    def _log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def _log_error(self, message: str) -> None:
        """Log error message.

        Image-related errors are always downgraded to warnings — visualization
        is a read-only operation and a single bad file should never prevent
        inspecting the rest of the dataset.
        """
        from dataflow.util.logging import detect_image_error

        if detect_image_error(message):
            self.logger.warning(message)
        else:
            self.logger.error(message)

    def _log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def _log_visualization_summary(self, result: VisualizationResult):
        """Log visualization summary."""
        from dataflow.visualize.log_templates import format_viz_result

        duration = self.summary_data["end_time"] - self.summary_data["start_time"]

        stats = {
            "total": self.summary_data["total_images"],
            "success": self.summary_data["processed_images"],
            "failed": self.summary_data["failed_images"],
            "objects": self.summary_data["total_objects"],
            "duration": f"{duration.total_seconds():.2f}s",
            "log_path": result.log_path,
        }
        self.logger.info(format_viz_result(stats))

    def _log_progress(self, current: int, message: str = "", n_objects: int = 0):
        """Log progress information (counter-based for streaming).

        Uses a counter format since the total image count is unknown
        until the iterator exhausts. When ``message`` contains the
        current image name, outputs a per-image progress line.

        Args:
            current: Current (1-based) image index.
            message: Image filename or path for display.
            n_objects: Number of annotation objects in this image.
        """
        failed = self.summary_data.get("failed_images", 0)

        if message:
            from dataflow.visualize.log_templates import format_viz_progress

            self.logger.info(
                format_viz_progress(
                    index=current,
                    image_name=message,
                    n_objects=n_objects,
                    status="✓",
                )
            )
        else:
            self.logger.info(
                f"[====] Processed {current} images, {failed} failed"
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

    @staticmethod
    def _update_window_title(
        cursor: int, buffer_len: int, image_path_str: str
    ) -> None:
        """Update the OpenCV window title with navigation position.

        Shows ``[N/T] filename`` in the title bar for user orientation.
        Cosmetic only — failures are silently ignored.
        """
        try:
            fname = Path(image_path_str).name
            title = (
                f"DataFlow-CV Visualization"
                f" [{cursor + 1}/{buffer_len}] {fname}"
            )
            cv2.setWindowTitle("DataFlow-CV Visualization", title)
        except Exception:
            pass  # Window title update is cosmetic; ignore failures
