"""
Format conversion module for DataFlow-CV.

This module provides annotation format conversion between LabelMe, YOLO, and COCO
formats. It supports all six conversion directions with lossless conversion
guarantees (except for RLE precision loss).

Key features:
- Bidirectional conversion between all three formats (6 conversion directions)
- Support for object detection and instance segmentation annotations
- RLE mask format support for COCO (with optional pycocotools dependency)
- Lossless conversion through OriginalData preservation
- Batch processing of entire directories
- Strict and lenient error handling modes

Example usage:
    >>> from dataflow.convert import LabelMeAndYoloConverter
    >>> converter = LabelMeAndYoloConverter(source_to_target=True)
    >>> result = converter.convert(
    ...     source_path="path/to/labelme",
    ...     target_path="path/to/yolo",
    ...     class_file="path/to/classes.txt"
    ... )
    >>> if result.success:
    >>>     print(f"Converted {result.num_images_converted} images")
"""

from . import utils
from .base import BaseConverter, ConversionResult
from .coco_and_labelme import CocoAndLabelMeConverter
from .labelme_and_yolo import LabelMeAndYoloConverter
from .yolo_and_coco import YoloAndCocoConverter

__all__ = [
    "BaseConverter",
    "ConversionResult",
    "LabelMeAndYoloConverter",
    "YoloAndCocoConverter",
    "CocoAndLabelMeConverter",
    "utils",
]
