# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/14
@File    : config.py
@Author  : DataFlow Team
@Description: Convert module configuration for DataFlow-CV
"""

from ..config import Config


class ConvertConfig:
    """Convert module专用配置"""

    # 继承全局配置
    YOLO_CLASSES_FILENAME = Config.YOLO_CLASSES_FILENAME
    YOLO_LABELS_DIRNAME = Config.YOLO_LABELS_DIRNAME
    IMAGE_EXTENSIONS = Config.IMAGE_EXTENSIONS
    YOLO_LABEL_EXTENSION = Config.YOLO_LABEL_EXTENSION
    COCO_JSON_EXTENSION = Config.COCO_JSON_EXTENSION
    OVERWRITE_EXISTING = Config.OVERWRITE_EXISTING
    CREATE_DIRS = Config.CREATE_DIRS
    COCO_DEFAULT_INFO = Config.COCO_DEFAULT_INFO
    VERBOSE = Config.VERBOSE
    YOLO_NORMALIZE = Config.YOLO_NORMALIZE
    YOLO_SEGMENTATION = Config.YOLO_SEGMENTATION

    # 转换模块特有配置
    DEFAULT_SEGMENTATION = False
    VALIDATE_ANNOTATIONS = True
    BATCH_SIZE = 100

    @classmethod
    def update_from_cli(cls, verbose=False, overwrite=False):
        """根据CLI选项更新全局配置"""
        Config.VERBOSE = verbose  # Explicitly set to True or False
        if overwrite:
            Config.OVERWRITE_EXISTING = True