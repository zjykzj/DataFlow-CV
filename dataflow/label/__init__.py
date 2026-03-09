# -*- coding: utf-8 -*-
"""
@Time    : 2026/3/9 20:43
@File    : __init__.py
@Author  : zj
@Description: Label模块 - 计算机视觉标注格式处理器

提供LabelMe、COCO和YOLO格式标签文件的读写功能。
支持目标检测（bbox）和实例分割（polygon）标注。
"""

from .labelme import LabelMeHandler
from .coco import CocoHandler
from .yolo import YoloHandler

__all__ = [
    'LabelMeHandler',
    'CocoHandler',
    'YoloHandler'
]
