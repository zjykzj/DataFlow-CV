"""
Utility functions for visualization module.
"""

from typing import Tuple, List
import cv2
import numpy as np


def scale_image_to_max_size(
    image: np.ndarray, max_size: int = 1920
) -> np.ndarray:
    """
    Scale image to fit within maximum dimension while maintaining aspect ratio.

    Args:
        image: Input image (numpy array)
        max_size: Maximum dimension (width or height)

    Returns:
        Scaled image
    """
    height, width = image.shape[:2]
    if max(height, width) <= max_size:
        return image

    scale = max_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )


def blend_with_background(foreground: np.ndarray, background: np.ndarray,
                          alpha: float = 0.7) -> np.ndarray:
    """
    Blend foreground image with background using alpha blending.

    Args:
        foreground: Foreground image
        background: Background image (must be same size as foreground)
        alpha: Foreground opacity (0.0 = fully transparent, 1.0 = fully opaque)

    Returns:
        Blended image
    """
    if foreground.shape != background.shape:
        raise ValueError(
            f"Shape mismatch: foreground {foreground.shape}, "
            f"background {background.shape}"
        )

    return cv2.addWeighted(foreground, alpha, background, 1 - alpha, 0)


def create_color_map(num_classes: int) -> List[Tuple[int, int, int]]:
    """
    Create a color map for given number of classes.

    Args:
        num_classes: Number of distinct colors needed

    Returns:
        List of BGR colors
    """
    # Use OpenCV's HSV color space for evenly distributed colors
    colors = []
    for i in range(num_classes):
        hue = int(180 * i / max(num_classes, 1))
        # Convert HSV to BGR
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        colors.append(tuple(map(int, bgr_color[0, 0])))

    return colors


def calculate_text_position(
    bbox_top_left: Tuple[int, int],
    text_size: Tuple[int, int],
    image_height: int,
    image_width: int,
) -> Tuple[int, int]:
    """
    Calculate text position ensuring it stays within image boundaries.

    Args:
        bbox_top_left: Top-left corner of bounding box
        text_size: (width, height) of text
        image_height: Image height
        image_width: Image width

    Returns:
        Adjusted (x, y) position for text
    """
    x, y = bbox_top_left
    text_width, text_height = text_size

    # Adjust y position to avoid going above image
    if y - text_height < 0:
        y = text_height

    # Adjust x position to avoid going beyond right edge
    if x + text_width > image_width:
        x = image_width - text_width - 5

    return (x, y)
