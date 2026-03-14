# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:41
@File    : config.py
@Author  : zj
@Description: Configuration settings for DataFlow-CV
"""

import os


class Config:
    """Configuration class for DataFlow-CV conversion utilities."""

    # File and directory names
    YOLO_CLASSES_FILENAME = "class.names"  # Default filename for YOLO classes
    YOLO_LABELS_DIRNAME = "labels"         # Default directory name for YOLO labels

    # File extensions
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    YOLO_LABEL_EXTENSION = ".txt"
    COCO_JSON_EXTENSION = ".json"

    # Default values
    DEFAULT_IMAGE_WIDTH = 640
    DEFAULT_IMAGE_HEIGHT = 640
    DEFAULT_IMAGE_CHANNELS = 3

    # Conversion options
    OVERWRITE_EXISTING = False  # Whether to overwrite existing files
    VERBOSE = False             # Print progress information
    CREATE_DIRS = True          # Create missing directories automatically

    # COCO format defaults
    COCO_DEFAULT_INFO = {
        "year": 2026,
        "version": "1.0",
        "description": "Converted by DataFlow-CV",
        "contributor": "DataFlow Team",
        "url": "https://github.com/zjykzj/DataFlow-CV",
        "date_created": "2026-03-08"
    }

    # YOLO format defaults
    YOLO_NORMALIZE = True  # Normalize coordinates to [0, 1]
    YOLO_SEGMENTATION = False  # Whether to handle segmentation masks (polygon format)

    @classmethod
    def get_image_extensions(cls):
        """Get tuple of valid image extensions."""
        return cls.IMAGE_EXTENSIONS

    @classmethod
    def get_yolo_label_extension(cls):
        """Get YOLO label file extension."""
        return cls.YOLO_LABEL_EXTENSION

    @classmethod
    def get_coco_json_extension(cls):
        """Get COCO JSON file extension."""
        return cls.COCO_JSON_EXTENSION

    @classmethod
    def validate_path(cls, path, is_dir=False, create=False):
        """
        Validate a path and optionally create it.

        Args:
            path (str): Path to validate
            is_dir (bool): Whether the path should be a directory
            create (bool): Whether to create the directory if it doesn't exist

        Returns:
            bool: True if path is valid, False otherwise
        """
        if not path:
            return False

        if is_dir:
            if create and cls.CREATE_DIRS:
                try:
                    os.makedirs(path, exist_ok=True)
                except (OSError, PermissionError):
                    return False
            return os.path.isdir(path) or (create and cls.CREATE_DIRS)
        else:
            return os.path.exists(os.path.dirname(path)) if os.path.dirname(path) else True
