"""
Visualization module for DataFlow-CV.

This module provides annotation visualization for computer vision datasets,
supporting three major formats: LabelMe, YOLO, and COCO. The module features
unified visualization
for both object detection and instance segmentation annotations.

Key features:
- Multi-format support (LabelMe, YOLO, COCO)
- Dual-task support (object detection and instance segmentation)
- RLE mask support for COCO format
- Automatic color management (consistent colors per category)
- Interactive mode (show images with keyboard controls)
- Save mode (save visualizations as JPEG images)
- Batch processing support
- Strict error handling and format validation

Example usage:
    >>> from dataflow.visualize import LabelMeVisualizer
    >>> visualizer = LabelMeVisualizer(
    >>>     label_dir="path/to/labelme",
    >>>     image_dir="path/to/images",
    >>>     is_show=True,
    >>>     is_save=False
    >>> )
    >>> result = visualizer.visualize()
"""

from . import utils
from .base import BaseVisualizer, ColorManager, VisualizationResult
from .coco_visualizer import COCOVisualizer
from .labelme_visualizer import LabelMeVisualizer
from .yolo_visualizer import YOLOVisualizer

__all__ = [
    "BaseVisualizer",
    "VisualizationResult",
    "ColorManager",
    "LabelMeVisualizer",
    "YOLOVisualizer",
    "COCOVisualizer",
    "utils",
]
