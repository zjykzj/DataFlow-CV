# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:41
@File    : __init__.py
@Author  : zj
@Description: DataFlow-CV: A data processing library for computer vision datasets
"""

__version__ = "0.2.0"
__author__ = "DataFlow Team"
__description__ = "A data processing library for computer vision datasets"

# Configuration
from .config import Config

# Converters
from .convert.coco_to_yolo import CocoToYoloConverter
from .convert.yolo_to_coco import YoloToCocoConverter

# Visualizers
from .visualize.yolo import YoloVisualizer
from .visualize.coco import CocoVisualizer

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
        output_dir: Output directory where labels/ and class.names will be created
        **kwargs: Additional options passed to CocoToYoloConverter.convert()

    Returns:
        Dictionary with conversion statistics
    """
    converter = CocoToYoloConverter()
    return converter.convert(coco_json_path, output_dir, **kwargs)


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


# Convenience functions for visualization
def visualize_yolo(
    image_dir: str,
    label_dir: str,
    class_path: str,
    save_dir: str = None,
    **kwargs
):
    """
    Visualize YOLO format annotations.

    Args:
        image_dir: Directory containing image files
        label_dir: Directory containing YOLO label files
        class_path: Path to class names file (e.g., class.names)
        save_dir: Directory to save visualized images (optional)
        **kwargs: Additional options passed to YoloVisualizer.visualize()

    Returns:
        Dictionary with visualization statistics
    """
    visualizer = YoloVisualizer(**kwargs)
    return visualizer.visualize(image_dir, label_dir, class_path, save_dir)


def visualize_coco(
    image_dir: str,
    annotation_json: str,
    save_dir: str = None,
    **kwargs
):
    """
    Visualize COCO format annotations.

    Args:
        image_dir: Directory containing image files
        annotation_json: Path to COCO JSON annotation file
        save_dir: Directory to save visualized images (optional)
        **kwargs: Additional options passed to CocoVisualizer.visualize()

    Returns:
        Dictionary with visualization statistics
    """
    visualizer = CocoVisualizer(**kwargs)
    return visualizer.visualize(image_dir, annotation_json, save_dir)


__all__ = [
    # Configuration
    "Config",

    # Converters
    "CocoToYoloConverter",
    "YoloToCocoConverter",

    # Visualizers
    "YoloVisualizer",
    "CocoVisualizer",

    # Base classes
    "BaseConverter",

    # CLI
    "cli",
    "main",

    # Convenience functions
    "coco_to_yolo",
    "yolo_to_coco",
    "visualize_yolo",
    "visualize_coco",

    # Metadata
    "__version__",
    "__author__",
    "__description__",
]