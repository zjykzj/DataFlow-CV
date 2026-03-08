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
            # Generate distinct colors using HSV color space
            hue = int(179 * class_id / max(num_classes, 1))
            hsv_color = np.uint8([[[hue, 255, 255]]])
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
                    label: str = "") -> np.ndarray:
        """
        Draw a polygon on the image (for segmentation masks).

        Args:
            image (np.ndarray): Input image (H, W, C)
            points (List[Tuple[float, float]]): List of (x, y) points
            color (Tuple[int, int, int]): BGR color for the polygon
            label (str): Class label text

        Returns:
            np.ndarray: Image with polygon drawn
        """
        if len(points) < 2:
            return image

        # Convert points to integer coordinates
        int_points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        # Draw polygon
        cv2.polylines(image, [int_points], isClosed=True, color=color,
                     thickness=self.line_thickness)

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