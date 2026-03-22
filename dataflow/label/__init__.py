"""
Label processing module for DataFlow-CV.

This module provides annotation handling for computer vision datasets, supporting
three major formats: LabelMe, YOLO, and COCO. The module features completely
lossless read/write functionality, preserving original annotation data to ensure
bitwise identical output when reading and re-writing annotation files.

Key features:
- Lossless round-trip annotation processing
- Original data preservation for all supported formats
- Priority-based writing (original data > preserved RLE > converted data)
- Support for mixed-origin datasets
- Comprehensive format conversion with data integrity

Example usage:
    >>> from dataflow.label import LabelMeAnnotationHandler
    >>> handler = LabelMeAnnotationHandler(label_dir="path/to/labelme")
    >>> result = handler.read()
    >>> if result.success:
    >>>     handler.write(result.data, "path/to/output")
"""

from .base import BaseAnnotationHandler, AnnotationResult
from .models import (
    DatasetAnnotations,
    ImageAnnotation,
    ObjectAnnotation,
    BoundingBox,
    Segmentation,
    AnnotationFormat,
    OriginalData,
    OriginalDataManager
)
from .labelme_handler import LabelMeAnnotationHandler
from .yolo_handler import YoloAnnotationHandler
from .coco_handler import CocoAnnotationHandler
from . import utils
from .utils import verify_lossless_roundtrip

__all__ = [
    'BaseAnnotationHandler',
    'AnnotationResult',
    'DatasetAnnotations',
    'ImageAnnotation',
    'ObjectAnnotation',
    'BoundingBox',
    'Segmentation',
    'AnnotationFormat',
    'OriginalData',
    'OriginalDataManager',
    'LabelMeAnnotationHandler',
    'YoloAnnotationHandler',
    'CocoAnnotationHandler',
    'utils',
    'verify_lossless_roundtrip',
]