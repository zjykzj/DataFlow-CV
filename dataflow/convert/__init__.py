# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:40
@File    : __init__.py
@Author  : zj
@Description: Conversion module for DataFlow-CV
"""

from .base import BaseConverter, LabelBasedConverter
# Old converters (to be removed after migration)
# from .coco_to_yolo import CocoToYoloConverter
# from .yolo_to_coco import YoloToCocoConverter

# New converters
from .coco_and_yolo import CocoToYoloConverter, YoloToCocoConverter
from .coco_and_labelme import CocoToLabelMeConverter, LabelMeToCocoConverter
from .yolo_and_labelme import YoloToLabelMeConverter, LabelMeToYoloConverter

__all__ = [
    "BaseConverter",
    "LabelBasedConverter",
    # New converters
    "CocoToYoloConverter",
    "YoloToCocoConverter",
    "CocoToLabelMeConverter",
    "LabelMeToCocoConverter",
    "YoloToLabelMeConverter",
    "LabelMeToYoloConverter",
]
