"""
Format conversion module for DataFlow-CV.

This module provides annotation format conversion between YOLO, LabelMe, and COCO
formats. It supports all six conversion directions with explicit coordinate
transforms.

Key features:
- Bidirectional conversion between all three formats (6 conversion directions)
- Support for object detection and instance segmentation annotations
- RLE mask format support for COCO (with optional pycocotools dependency)
- Explicit coordinate transforms (normalized ↔ absolute pixel)
- Batch processing of entire directories
- Strict and lenient error handling modes

Example usage:
    >>> from dataflow.convert import YoloAndCocoConverter
    >>> converter = YoloAndCocoConverter(source_to_target=True)
    >>> result = converter.convert(
    ...     source_path="path/to/yolo",
    ...     target_path="path/to/coco.json",
    ...     class_file="path/to/classes.txt",
    ...     image_dir="path/to/images"
    ... )
    >>> if result.success:
    >>>     print(f"Converted {result.num_images_converted} images")
"""

from . import utils
from .base import BaseConverter, ConversionResult
from .yolo_and_coco import YoloAndCocoConverter
from .labelme_and_yolo import LabelMeAndYoloConverter
from .coco_and_labelme import CocoAndLabelMeConverter

__all__ = [
    "BaseConverter",
    "ConversionResult",
    "YoloAndCocoConverter",
    "LabelMeAndYoloConverter",
    "CocoAndLabelMeConverter",
    "utils",
]
