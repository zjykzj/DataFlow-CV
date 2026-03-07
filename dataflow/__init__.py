"""
DataFlow - A data processing library for computer vision datasets.

Features:
- Dataset format conversion (LabelMe, COCO, YOLO)
- Single-image visualization
- Dataset downloading (future)

Supported formats:
- LabelMe: JSON format for polygon/rectangle annotations
- COCO: JSON format for object detection and instance segmentation
- YOLO: TXT format with normalized coordinates

Version: 0.1.1
"""

__version__ = "0.1.1"
__author__ = "DataFlow Team"

from . import convert
from . import visualize
from . import cli
from . import config