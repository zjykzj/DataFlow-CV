"""
Visualization module for DataFlow-CV.

This module provides annotation visualization for computer vision datasets,
supporting three major formats: YOLO, LabelMe, and COCO. The module features
unified visualization
for both object detection and instance segmentation annotations.

Key features:
- Multi-format support (YOLO, LabelMe, COCO)
- Dual-task support (object detection and instance segmentation)
- RLE mask support for COCO format
- Automatic color management (consistent colors per category)
- Interactive mode (show images with keyboard controls)
- Save mode (save visualizations as JPEG images)
- Batch processing support
- Strict error handling and format validation

Example usage:
    >>> from dataflow.visualize import YOLOVisualizer
    >>> visualizer = YOLOVisualizer(
    >>>     label_dir="path/to/yolo/labels",
    >>>     image_dir="path/to/images",
    >>>     class_file="path/to/classes.txt",
    >>>     is_show=True,
    >>>     is_save=False
    >>> )
    >>> result = visualizer.visualize()
"""

from . import utils
from .base import (BaseVisualizer, ColorManager, RenderAnnotation, RenderData,
                   VisualizationResult)
from .yolo_visualizer import YOLOVisualizer
from .labelme_visualizer import LabelMeVisualizer
from .coco_visualizer import COCOVisualizer

__all__ = [
    "BaseVisualizer",
    "VisualizationResult",
    "ColorManager",
    "YOLOVisualizer",
    "LabelMeVisualizer",
    "COCOVisualizer",
    "utils",
]
