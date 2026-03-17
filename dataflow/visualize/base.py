# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:40
@File    : base.py
@Author  : zj
@Description: Base visualizer class for annotation visualization
"""

import os
import abc
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import cv2

from ..config import Config
from .config import VisualizeConfig


class BaseVisualizer:
    """Base class for annotation visualizers."""

    # Visualization defaults
    DEFAULT_COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 0),    # Dark Red
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Dark Blue
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
    ]

    def __init__(self, verbose: bool = None):
        """
        Initialize the visualizer.

        Args:
            verbose (bool, optional): Whether to print progress information.
                If None, uses Config.VERBOSE.
        """
        self.verbose = Config.VERBOSE if verbose is None else verbose
        self.logger = self._setup_logger()

        # Visualization settings
        self.line_thickness = 2
        self.font_scale = 0.5
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_thickness = 1
        self.text_color = (255, 255, 255)  # White text
        self.text_bg_color = (0, 0, 0)     # Black background for text
        self.text_padding = 5
        self.border_radius = 3

        # Display settings
        self.window_name = "DataFlow-CV Visualization"
        self.max_display_size = (1280, 720)  # Max window size

        # Polygon fill and transparency settings
        self.fill_polygons = VisualizeConfig.DEFAULT_FILL_POLYGONS
        self.fill_alpha = VisualizeConfig.DEFAULT_FILL_ALPHA
        self.outline_alpha = VisualizeConfig.DEFAULT_OUTLINE_ALPHA
        self.highlight_rle = VisualizeConfig.DEFAULT_HIGHLIGHT_RLE
        self.rle_fill_color = VisualizeConfig.DEFAULT_RLE_FILL_COLOR

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the visualizer."""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger

    def validate_input_path(self, path: str, is_dir: bool = False, create: bool = False) -> bool:
        """Validate input path using Config."""
        return Config.validate_path(path, is_dir=is_dir, create=create)

    def validate_output_path(self, path: str, is_dir: bool = False, create: bool = False) -> bool:
        """Validate output path, creating directories if needed."""
        return Config.validate_path(path, is_dir=is_dir, create=create)

    def ensure_directory(self, dir_path: str) -> bool:
        """Ensure directory exists, create if needed."""
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to create directory {dir_path}: {e}")
                return False
        return True

    def get_image_files(self, image_dir: str) -> List[str]:
        """
        Get list of image files in directory.

        Args:
            image_dir (str): Directory containing images

        Returns:
            List of image file paths
        """
        if not self.validate_input_path(image_dir, is_dir=True):
            raise ValueError(f"Invalid image directory: {image_dir}")

        image_files = []
        for ext in Config.get_image_extensions():
            import glob
            pattern = os.path.join(image_dir, f"*{ext}")
            image_files.extend(glob.glob(pattern))
            # Also check uppercase extensions
            pattern = os.path.join(image_dir, f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern))

        return sorted(image_files)

    def get_color_for_class(self, class_id: int, num_classes: int = None) -> Tuple[int, int, int]:
        """
        Get a consistent color for a given class ID.

        Args:
            class_id (int): Class ID (0-indexed)
            num_classes (int, optional): Total number of classes for color scaling

        Returns:
            Tuple of (B, G, R) color values (OpenCV format)
        """
        if num_classes is None or num_classes <= len(self.DEFAULT_COLORS):
            # Use predefined colors, cycling if needed
            color_idx = class_id % len(self.DEFAULT_COLORS)
            return self.DEFAULT_COLORS[color_idx]
        else:
            # Generate distinct colors using HSV color space with golden ratio distribution
            # This provides better color separation than linear spacing
            # Golden angle in degrees: 137.508 (gives optimal spacing on color wheel)
            golden_angle = 137.508
            # Use modulo 360 to wrap around the hue circle
            hue_angle = (class_id * golden_angle) % 360.0
            # Convert to OpenCV hue range (0-179, corresponding to 0-360 degrees)
            hue = int(hue_angle * 179.0 / 360.0)

            # Vary saturation and value slightly to increase color distinction
            # while keeping colors bright and vibrant
            # Use class_id to create patterns in saturation and value
            # This adds extra dimension of variation beyond just hue
            saturation = 220 + (class_id % 4) * 12  # 220-256 range
            value = 220 + ((class_id // 4) % 4) * 12  # 220-256 range

            hsv_color = np.uint8([[[hue, saturation, value]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
            return tuple(map(int, bgr_color[0][0]))

    def draw_bounding_box(self,
                         image: np.ndarray,
                         bbox: Tuple[float, float, float, float],
                         color: Tuple[int, int, int],
                         label: str = "",
                         confidence: float = None) -> np.ndarray:
        """
        Draw a bounding box on the image.

        Args:
            image (np.ndarray): Input image (H, W, C)
            bbox (Tuple[float, float, float, float]): Bounding box in format
                (x_min, y_min, x_max, y_max) in pixel coordinates
            color (Tuple[int, int, int]): BGR color for the box
            label (str): Class label text
            confidence (float): Confidence score (0-1)

        Returns:
            np.ndarray: Image with bounding box drawn
        """
        img_height, img_width = image.shape[:2]

        # Convert bbox coordinates to integers
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(img_width - 1, int(bbox[2]))
        y2 = min(img_height - 1, int(bbox[3]))

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.line_thickness)

        # Prepare label text
        label_text = label
        if confidence is not None:
            label_text += f" {confidence:.2f}"

        if label_text:
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, self.font, self.font_scale, self.text_thickness
            )

            # Draw text background
            text_bg_top_left = (x1, y1 - text_height - 2 * self.text_padding - baseline)
            text_bg_bottom_right = (x1 + text_width + 2 * self.text_padding, y1)

            # Ensure text background stays within image bounds
            if text_bg_top_left[1] < 0:
                text_bg_top_left = (x1, y2)
                text_bg_bottom_right = (x1 + text_width + 2 * self.text_padding,
                                        y2 + text_height + 2 * self.text_padding + baseline)

            cv2.rectangle(image, text_bg_top_left, text_bg_bottom_right,
                         self.text_bg_color, -1)

            # Draw text
            text_position = (x1 + self.text_padding,
                           text_bg_bottom_right[1] - self.text_padding - baseline)
            cv2.putText(image, label_text, text_position, self.font,
                       self.font_scale, self.text_color, self.text_thickness)

        return image

    def draw_polygon(self,
                    image: np.ndarray,
                    points: List[Tuple[float, float]],
                    color: Tuple[int, int, int],
                    label: str = "",
                    fill: Optional[bool] = None,
                    fill_alpha: Optional[float] = None,
                    outline_alpha: Optional[float] = None,
                    is_rle: bool = False) -> np.ndarray:
        """
        Draw a polygon on the image (for segmentation masks).
        Supports optional filled polygons with transparency and RLE mask highlighting.

        Args:
            image (np.ndarray): Input image (H, W, C)
            points (List[Tuple[float, float]]): List of (x, y) points
            color (Tuple[int, int, int]): BGR color for the polygon
            label (str): Class label text
            fill (bool, optional): Whether to fill the polygon. If None, uses
                self.fill_polygons setting.
            fill_alpha (float, optional): Alpha transparency for fill (0.0-1.0).
                If None, uses self.fill_alpha.
            outline_alpha (float, optional): Alpha transparency for outline (0.0-1.0).
                If None, uses self.outline_alpha.
            is_rle (bool): Whether this polygon originated from RLE mask.
                Affects fill color if self.rle_fill_color is set.

        Returns:
            np.ndarray: Image with polygon drawn
        """
        if len(points) < 2:
            return image

        # Use instance settings if parameters not provided
        if fill is None:
            fill = self.fill_polygons
        if fill_alpha is None:
            fill_alpha = self.fill_alpha
        if outline_alpha is None:
            outline_alpha = self.outline_alpha

        # Determine fill color for RLE masks
        fill_color = color
        if is_rle and self.rle_fill_color is not None:
            fill_color = self.rle_fill_color

        # Convert points to integer coordinates
        int_points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        # Simple case: no fill and full opacity outline (original behavior)
        if not fill and outline_alpha >= 1.0:
            cv2.polylines(image, [int_points], isClosed=True, color=color,
                         thickness=self.line_thickness)
        else:
            # Handle fill and/or transparency
            # We'll blend fill and outline separately to support different alphas
            # Fill overlay
            if fill and fill_alpha > 0:
                fill_overlay = np.zeros_like(image, dtype=np.uint8)
                cv2.fillPoly(fill_overlay, [int_points], fill_color)
                image = cv2.addWeighted(fill_overlay, fill_alpha, image, 1 - fill_alpha, 0)

            # Outline overlay
            if outline_alpha > 0:
                outline_overlay = np.zeros_like(image, dtype=np.uint8)
                cv2.polylines(outline_overlay, [int_points], isClosed=True, color=color,
                             thickness=self.line_thickness)
                image = cv2.addWeighted(outline_overlay, outline_alpha, image, 1 - outline_alpha, 0)

        # Draw label at first point if provided
        if label:
            first_point = (int(points[0][0]), int(points[0][1]))
            (text_width, text_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.text_thickness
            )

            text_bg_top_left = (first_point[0], first_point[1] - text_height - 2 * self.text_padding - baseline)
            text_bg_bottom_right = (first_point[0] + text_width + 2 * self.text_padding, first_point[1])

            if text_bg_top_left[1] < 0:
                text_bg_top_left = (first_point[0], first_point[1] + text_height + 2 * self.text_padding + baseline)
                text_bg_bottom_right = (first_point[0] + text_width + 2 * self.text_padding,
                                       first_point[1] + 2 * (text_height + 2 * self.text_padding + baseline))

            cv2.rectangle(image, text_bg_top_left, text_bg_bottom_right,
                         self.text_bg_color, -1)

            text_position = (first_point[0] + self.text_padding,
                           text_bg_bottom_right[1] - self.text_padding - baseline)
            cv2.putText(image, label, text_position, self.font,
                       self.font_scale, self.text_color, self.text_thickness)

        return image

    def read_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Read image using OpenCV.

        Args:
            image_path (str): Path to image file

        Returns:
            np.ndarray: Image as BGR numpy array, or None if failed
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                self.logger.warning(f"Could not read image: {image_path}")
                return None
            return image
        except Exception as e:
            self.logger.error(f"Error reading image {image_path}: {e}")
            return None

    def save_image(self, image: np.ndarray, output_path: str, quality: int = 95) -> bool:
        """
        Save image to file.

        Args:
            image (np.ndarray): Image to save
            output_path (str): Output file path
            quality (int): JPEG quality (1-100) for JPEG images

        Returns:
            bool: True if successful
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not self.ensure_directory(output_dir):
                return False

            # Determine compression parameters based on file extension
            ext = os.path.splitext(output_path)[1].lower()
            params = []
            if ext in ['.jpg', '.jpeg']:
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif ext == '.png':
                params = [cv2.IMWRITE_PNG_COMPRESSION, 9]

            success = cv2.imwrite(output_path, image, params)
            if success:
                self.logger.info(f"Saved image to: {output_path}")
            else:
                self.logger.error(f"Failed to save image: {output_path}")
            return success
        except Exception as e:
            self.logger.error(f"Error saving image {output_path}: {e}")
            return False

    def display_image(self, image: np.ndarray, wait_key: int = 0) -> int:
        """
        Display image in a window.

        Args:
            image (np.ndarray): Image to display
            wait_key (int): Time in milliseconds to wait for key press.
                0 means wait indefinitely.

        Returns:
            int: Key code pressed, or -1 if window closed
        """
        # Resize image if too large for display
        display_image = self._resize_for_display(image)

        cv2.imshow(self.window_name, display_image)
        key = cv2.waitKey(wait_key)
        return key

    def _resize_for_display(self, image: np.ndarray) -> np.ndarray:
        """Resize image to fit within max display size while maintaining aspect ratio."""
        height, width = image.shape[:2]
        max_width, max_height = self.max_display_size

        if width <= max_width and height <= max_height:
            return image

        # Calculate scaling factor
        width_ratio = max_width / width
        height_ratio = max_height / height
        scale = min(width_ratio, height_ratio)

        new_width = int(width * scale)
        new_height = int(height * scale)

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def close_windows(self):
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()

    def _print_progress(self, current: int, total: int, prefix: str = ""):
        """Print progress information."""
        if self.verbose:
            percent = (current / total) * 100
            self.logger.info(f"{prefix}Progress: {current}/{total} ({percent:.1f}%)")

    @abc.abstractmethod
    def visualize(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Perform visualization. Must be implemented by subclasses.

        Returns:
            Dict with visualization results
        """
        pass