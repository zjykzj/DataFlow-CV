"""
Label processing module for DataFlow-CV.

This module provides annotation handling for computer vision datasets, supporting
three major formats: LabelMe, YOLO, and COCO. Each handler stores coordinates in
its format's native representation — see DatasetAnnotations.format to determine
the coordinate semantics.

Key features:
- Format-native coordinate storage (no unified normalization)
- Read/write/validate for YOLO, COCO, and LabelMe formats
- Format-aware validation
- Category management utilities

Example usage:
    >>> from dataflow.label import LabelMeAnnotationHandler
    >>> handler = LabelMeAnnotationHandler(label_dir="path/to/labelme")
    >>> result = handler.read()
    >>> if result.success:
    >>>     handler.write(result.data, "path/to/output")
"""

from . import utils
from .base import AnnotationResult, BaseAnnotationHandler
from .coco_handler import CocoAnnotationHandler
from .labelme_handler import LabelMeAnnotationHandler
from .models import (AnnotationFormat, BoundingBox, DatasetAnnotations,
                     ImageAnnotation, ObjectAnnotation, Segmentation)
from .yolo_handler import YoloAnnotationHandler

__all__ = [
    "BaseAnnotationHandler",
    "AnnotationResult",
    "ImageError",
    "DatasetAnnotations",
    "ImageAnnotation",
    "ObjectAnnotation",
    "BoundingBox",
    "Segmentation",
    "AnnotationFormat",
    "LabelMeAnnotationHandler",
    "YoloAnnotationHandler",
    "CocoAnnotationHandler",
    "utils",
]
