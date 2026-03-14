# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/14
@File    : config.py
@Author  : DataFlow Team
@Description: Visualize module configuration for DataFlow-CV
"""

from ..config import Config


class VisualizeConfig:
    """Visualize模块专用配置"""

    # 继承全局配置
    IMAGE_EXTENSIONS = Config.IMAGE_EXTENSIONS
    VERBOSE = Config.VERBOSE
    OVERWRITE_EXISTING = Config.OVERWRITE_EXISTING
    YOLO_CLASSES_FILENAME = Config.YOLO_CLASSES_FILENAME
    YOLO_LABELS_DIRNAME = Config.YOLO_LABELS_DIRNAME
    YOLO_LABEL_EXTENSION = Config.YOLO_LABEL_EXTENSION
    COCO_JSON_EXTENSION = Config.COCO_JSON_EXTENSION
    CREATE_DIRS = Config.CREATE_DIRS

    # 可视化模块特有配置
    DEFAULT_WINDOW_SIZE = (800, 600)
    DEFAULT_COLOR_SCHEME = "tab20c"
    SHOW_CONFIDENCE = False
    FONT_SCALE = 0.5
    LINE_THICKNESS = 2

    @classmethod
    def update_from_cli(cls, verbose=False, overwrite=False):
        """根据CLI选项更新全局配置"""
        Config.VERBOSE = verbose  # Explicitly set to True or False
        if overwrite:
            Config.OVERWRITE_EXISTING = True