# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:41
@File    : __init__.py
@Author  : zj
@Description: DataFlow-CV: A data processing library for computer vision datasets
"""

__version__ = "0.5.0"
__author__ = "DataFlow Team"
__description__ = "A data processing library for computer vision datasets"

# Configuration
from .config import Config

# Converters
from .convert import (
    CocoToYoloConverter,
    YoloToCocoConverter,
    CocoToLabelMeConverter,
    LabelMeToCocoConverter,
    YoloToLabelMeConverter,
    LabelMeToYoloConverter
)

# Visualizers
from .visualize.yolo import YoloVisualizer
from .visualize.coco import CocoVisualizer
from .visualize.labelme import LabelMeVisualizer

# Base classes
from .convert.base import BaseConverter

# CLI
from .cli import cli, main

# Convenience functions for common conversions
def coco_to_yolo(coco_json_path: str, output_dir: str, **kwargs):
    """
    Convert COCO JSON to YOLO format.

    Args:
        coco_json_path: Path to COCO JSON file
        output_dir: Output directory where YOLO label files will be created
            (class.names will be auto-generated in output_dir)
        **kwargs: Additional options passed to CocoToYoloConverter.convert()

    Returns:
        Dictionary with conversion statistics
    """
    converter = CocoToYoloConverter()
    return converter.convert(coco_json_path, output_dir, classes_path=None, **kwargs)


def yolo_to_coco(
    image_dir: str,
    yolo_labels_dir: str,
    yolo_class_path: str,
    coco_json_path: str,
    **kwargs
):
    """
    Convert YOLO format to COCO JSON.

    Args:
        image_dir: Directory containing image files
        yolo_labels_dir: Directory containing YOLO label files
        yolo_class_path: Path to YOLO class names file
        coco_json_path: Path to save COCO JSON file
        **kwargs: Additional options passed to YoloToCocoConverter.convert()

    Returns:
        Dictionary with conversion statistics
    """
    converter = YoloToCocoConverter()
    return converter.convert(
        image_dir, yolo_labels_dir, yolo_class_path, coco_json_path, **kwargs
    )


# Convenience functions for LabelMe conversions
def coco_to_labelme(coco_json_path: str, output_dir: str, **kwargs):
    """
    Convert COCO JSON to LabelMe format.

    Args:
        coco_json_path: Path to COCO JSON file
        output_dir: Output directory where LabelMe JSON files will be created
        **kwargs: Additional options passed to CocoToLabelMeConverter.convert()

    Returns:
        Dictionary with conversion statistics
    """
    converter = CocoToLabelMeConverter()
    return converter.convert(coco_json_path, output_dir, **kwargs)


def labelme_to_coco(label_dir: str, classes_path: str, output_json_path: str, **kwargs):
    """
    Convert LabelMe format to COCO JSON.

    Args:
        label_dir: Directory containing LabelMe JSON files
        classes_path: Path to class names file (e.g., class.names)
        output_json_path: Path to save COCO JSON file
        **kwargs: Additional options passed to LabelMeToCocoConverter.convert()

    Returns:
        Dictionary with conversion statistics
    """
    converter = LabelMeToCocoConverter()
    return converter.convert(label_dir, classes_path, output_json_path, **kwargs)


def yolo_to_labelme(image_dir: str, label_dir: str, classes_path: str, output_dir: str, **kwargs):
    """
    Convert YOLO format to LabelMe format.

    Args:
        image_dir: Directory containing image files
        label_dir: Directory containing YOLO label files (.txt)
        classes_path: Path to YOLO class names file (e.g., class.names)
        output_dir: Output directory where LabelMe JSON files will be created
        **kwargs: Additional options passed to YoloToLabelMeConverter.convert()

    Returns:
        Dictionary with conversion statistics
    """
    converter = YoloToLabelMeConverter()
    return converter.convert(image_dir, label_dir, classes_path, output_dir, **kwargs)


def labelme_to_yolo(label_dir: str, classes_path: str, output_dir: str, **kwargs):
    """
    Convert LabelMe format to YOLO format.

    Args:
        label_dir: Directory containing LabelMe JSON files
        classes_path: Path to class names file (e.g., class.names)
        output_dir: Output directory where YOLO label files will be created
        **kwargs: Additional options passed to LabelMeToYoloConverter.convert()

    Returns:
        Dictionary with conversion statistics
    """
    converter = LabelMeToYoloConverter()
    return converter.convert(label_dir, classes_path, output_dir, **kwargs)


# Convenience functions for visualization
def visualize_yolo(
    image_dir: str,
    label_dir: str,
    class_path: str,
    save_dir: str = None,
    segmentation: bool = False,
    fill: bool = None,
    fill_alpha: float = None,
    outline_alpha: float = None,
    highlight_rle: bool = None,
    rle_color: str = None,
    verbose: bool = None,
    **kwargs
):
    """
    Visualize YOLO format annotations.

    Args:
        image_dir: Directory containing image files
        label_dir: Directory containing YOLO label files
        class_path: Path to class names file (e.g., class.names)
        save_dir: Directory to save visualized images (optional)
        segmentation: Whether to force segmentation mode (strict validation)
        fill: Whether to fill polygons (default: False)
        fill_alpha: Fill transparency (0.0-1.0, default: 0.3)
        outline_alpha: Outline transparency (0.0-1.0, default: 1.0)
        highlight_rle: Whether to highlight RLE masks (default: True)
        rle_color: RLE fill color as "R,G,B" (e.g., "255,0,0")
        verbose: Whether to print detailed progress information
        **kwargs: Additional options passed to YoloVisualizer.visualize()

    Returns:
        Dictionary with visualization statistics
    """
    visualizer = YoloVisualizer(verbose=verbose, segmentation=segmentation)

    # Set polygon fill and transparency options
    if fill is not None:
        visualizer.fill_polygons = fill
    if fill_alpha is not None:
        if not 0.0 <= fill_alpha <= 1.0:
            raise ValueError("fill_alpha must be between 0.0 and 1.0")
        visualizer.fill_alpha = fill_alpha
    if outline_alpha is not None:
        if not 0.0 <= outline_alpha <= 1.0:
            raise ValueError("outline_alpha must be between 0.0 and 1.0")
        visualizer.outline_alpha = outline_alpha
    if highlight_rle is not None:
        visualizer.highlight_rle = highlight_rle
    if rle_color is not None:
        # Parse R,G,B string
        try:
            parts = rle_color.split(',')
            if len(parts) != 3:
                raise ValueError
            r, g, b = map(int, parts)
            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                raise ValueError
            visualizer.rle_fill_color = (b, g, r)  # OpenCV uses BGR
        except ValueError:
            raise ValueError('rle_color must be "R,G,B" with integers 0-255')

    return visualizer.visualize(image_dir, label_dir, class_path, save_dir, **kwargs)


def visualize_coco(
    image_dir: str,
    annotation_json: str,
    save_dir: str = None,
    segmentation: bool = False,
    fill: bool = None,
    fill_alpha: float = None,
    outline_alpha: float = None,
    highlight_rle: bool = None,
    rle_color: str = None,
    verbose: bool = None,
    **kwargs
):
    """
    Visualize COCO format annotations.

    Args:
        image_dir: Directory containing image files
        annotation_json: Path to COCO JSON annotation file
        save_dir: Directory to save visualized images (optional)
        segmentation: Whether to force segmentation mode (strict validation)
        fill: Whether to fill polygons (default: False)
        fill_alpha: Fill transparency (0.0-1.0, default: 0.3)
        outline_alpha: Outline transparency (0.0-1.0, default: 1.0)
        highlight_rle: Whether to highlight RLE masks (default: True)
        rle_color: RLE fill color as "R,G,B" (e.g., "255,0,0")
        verbose: Whether to print detailed progress information
        **kwargs: Additional options passed to CocoVisualizer.visualize()

    Returns:
        Dictionary with visualization statistics
    """
    visualizer = CocoVisualizer(verbose=verbose, segmentation=segmentation)

    # Set polygon fill and transparency options
    if fill is not None:
        visualizer.fill_polygons = fill
    if fill_alpha is not None:
        if not 0.0 <= fill_alpha <= 1.0:
            raise ValueError("fill_alpha must be between 0.0 and 1.0")
        visualizer.fill_alpha = fill_alpha
    if outline_alpha is not None:
        if not 0.0 <= outline_alpha <= 1.0:
            raise ValueError("outline_alpha must be between 0.0 and 1.0")
        visualizer.outline_alpha = outline_alpha
    if highlight_rle is not None:
        visualizer.highlight_rle = highlight_rle
    if rle_color is not None:
        # Parse R,G,B string
        try:
            parts = rle_color.split(',')
            if len(parts) != 3:
                raise ValueError
            r, g, b = map(int, parts)
            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                raise ValueError
            visualizer.rle_fill_color = (b, g, r)  # OpenCV uses BGR
        except ValueError:
            raise ValueError('rle_color must be "R,G,B" with integers 0-255')

    return visualizer.visualize(image_dir, annotation_json, save_dir, **kwargs)


def visualize_labelme(
    image_dir: str,
    label_dir: str,
    save_dir: str = None,
    segmentation: bool = False,
    fill: bool = None,
    fill_alpha: float = None,
    outline_alpha: float = None,
    highlight_rle: bool = None,
    rle_color: str = None,
    verbose: bool = None,
    **kwargs
):
    """
    Visualize LabelMe format annotations.

    Args:
        image_dir: Directory containing image files
        label_dir: Directory containing LabelMe JSON files
        save_dir: Directory to save visualized images (optional)
        segmentation: Whether to force segmentation mode (strict validation)
        fill: Whether to fill polygons (default: False)
        fill_alpha: Fill transparency (0.0-1.0, default: 0.3)
        outline_alpha: Outline transparency (0.0-1.0, default: 1.0)
        highlight_rle: Whether to highlight RLE masks (default: True)
        rle_color: RLE fill color as "R,G,B" (e.g., "255,0,0")
        verbose: Whether to print detailed progress information
        **kwargs: Additional options passed to LabelMeVisualizer.visualize()

    Returns:
        Dictionary with visualization statistics
    """
    visualizer = LabelMeVisualizer(verbose=verbose, segmentation=segmentation)

    # Set polygon fill and transparency options
    if fill is not None:
        visualizer.fill_polygons = fill
    if fill_alpha is not None:
        if not 0.0 <= fill_alpha <= 1.0:
            raise ValueError("fill_alpha must be between 0.0 and 1.0")
        visualizer.fill_alpha = fill_alpha
    if outline_alpha is not None:
        if not 0.0 <= outline_alpha <= 1.0:
            raise ValueError("outline_alpha must be between 0.0 and 1.0")
        visualizer.outline_alpha = outline_alpha
    if highlight_rle is not None:
        visualizer.highlight_rle = highlight_rle
    if rle_color is not None:
        # Parse R,G,B string
        try:
            parts = rle_color.split(',')
            if len(parts) != 3:
                raise ValueError
            r, g, b = map(int, parts)
            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                raise ValueError
            visualizer.rle_fill_color = (b, g, r)  # OpenCV uses BGR
        except ValueError:
            raise ValueError('rle_color must be "R,G,B" with integers 0-255')

    return visualizer.visualize(image_dir, label_dir, save_dir, **kwargs)


__all__ = [
    # Configuration
    "Config",

    # Converters
    "CocoToYoloConverter",
    "YoloToCocoConverter",
    "CocoToLabelMeConverter",
    "LabelMeToCocoConverter",
    "YoloToLabelMeConverter",
    "LabelMeToYoloConverter",

    # Visualizers
    "YoloVisualizer",
    "CocoVisualizer",
    "LabelMeVisualizer",
    "GenericVisualizer",

    # Base classes
    "BaseConverter",

    # CLI
    "cli",
    "main",

    # Convenience functions
    "coco_to_yolo",
    "yolo_to_coco",
    "coco_to_labelme",
    "labelme_to_coco",
    "yolo_to_labelme",
    "labelme_to_yolo",
    "visualize_yolo",
    "visualize_coco",
    "visualize_labelme",

    # Metadata
    "__version__",
    "__author__",
    "__description__",
]