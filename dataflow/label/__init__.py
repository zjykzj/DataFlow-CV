"""
Label processing module for DataFlow-CV.

This module provides annotation handling for computer vision datasets, supporting
three major formats: YOLO, LabelMe, and COCO. Each handler stores coordinates in
its format's native representation — see DatasetAnnotations.format to determine
the coordinate semantics.

Key features:
- Format-native coordinate storage (no unified normalization)
- Read/write/validate for YOLO, LabelMe, and COCO formats
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
from .yolo_handler import YoloAnnotationHandler
from .labelme_handler import LabelMeAnnotationHandler
from .coco_handler import CocoAnnotationHandler
from .models import (AnnotationFormat, BoundingBox, DatasetAnnotations,
                     ImageAnnotation, ObjectAnnotation, Segmentation)

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
    "YoloAnnotationHandler",
    "LabelMeAnnotationHandler",
    "CocoAnnotationHandler",
    "utils",
]
