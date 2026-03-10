# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:40
@File    : base.py
@Author  : zj
@Description: Base converter class for format conversions
"""

import os
import abc
import logging
from typing import List, Dict, Any, Optional, Tuple
from ..config import Config


class BaseConverter(abc.ABC):
    """Abstract base class for format converters."""

    def __init__(self, verbose: bool = None):
        """
        Initialize the converter.

        Args:
            verbose (bool, optional): Whether to print progress information.
                If None, uses Config.VERBOSE.
        """
        self.verbose = Config.VERBOSE if verbose is None else verbose
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the converter."""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger

    @abc.abstractmethod
    def convert(self, *args, **kwargs) -> Any:
        """
        Perform conversion. Must be implemented by subclasses.

        Returns:
            Conversion result (depends on specific converter)
        """
        pass

    def batch_convert(self, inputs: List[Any], outputs: List[Any], **kwargs) -> List[Any]:
        """
        Convert multiple inputs to outputs.

        Args:
            inputs: List of input items
            outputs: List of output items (must match length of inputs)
            **kwargs: Additional arguments passed to convert()

        Returns:
            List of conversion results
        """
        if len(inputs) != len(outputs):
            raise ValueError(f"Inputs ({len(inputs)}) and outputs ({len(outputs)}) must have same length")

        results = []
        for i, (input_item, output_item) in enumerate(zip(inputs, outputs)):
            if self.verbose:
                self.logger.info(f"Processing item {i+1}/{len(inputs)}: {input_item}")

            try:
                result = self.convert(input_item, output_item, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to convert {input_item}: {e}")
                if Config.OVERWRITE_EXISTING:
                    raise
                else:
                    results.append(None)

        return results

    def validate_input_path(self, path: str, is_dir: bool = False, create: bool = False) -> bool:
        """Validate input path using Config."""
        return Config.validate_path(path, is_dir=is_dir, create=create)

    def validate_output_path(self, path: str, is_dir: bool = False, create: bool = False) -> bool:
        """Validate output path, creating directories if needed."""
        return Config.validate_path(path, is_dir=is_dir, create=create)

    def ensure_directory(self, dir_path: str) -> bool:
        """Ensure directory exists, create if needed."""
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to create directory {dir_path}: {e}")
                return False
        return True

    def get_image_files(self, image_dir: str) -> List[str]:
        """
        Get list of image files in directory.

        Args:
            image_dir (str): Directory containing images

        Returns:
            List of image file paths
        """
        if not self.validate_input_path(image_dir, is_dir=True):
            raise ValueError(f"Invalid image directory: {image_dir}")

        image_files = []
        for ext in Config.get_image_extensions():
            pattern = os.path.join(image_dir, f"*{ext}")
            import glob
            image_files.extend(glob.glob(pattern))
            # Also check uppercase extensions
            pattern = os.path.join(image_dir, f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern))

        return sorted(image_files)

    def get_label_files(self, label_dir: str, extension: str = None) -> List[str]:
        """
        Get list of label files in directory.

        Args:
            label_dir (str): Directory containing label files
            extension (str, optional): File extension to filter by.
                If None, uses Config.YOLO_LABEL_EXTENSION

        Returns:
            List of label file paths
        """
        if not self.validate_input_path(label_dir, is_dir=True):
            raise ValueError(f"Invalid label directory: {label_dir}")

        ext = extension or Config.get_yolo_label_extension()
        import glob
        pattern = os.path.join(label_dir, f"*{ext}")
        label_files = glob.glob(pattern)
        return sorted(label_files)

    def read_classes_file(self, classes_path: str) -> List[str]:
        """
        Read class names from file.

        Args:
            classes_path (str): Path to classes file

        Returns:
            List of class names
        """
        if not self.validate_input_path(classes_path, is_dir=False):
            raise ValueError(f"Invalid classes file: {classes_path}")

        with open(classes_path, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f if line.strip()]

        return classes

    def write_classes_file(self, classes: List[str], output_path: str) -> bool:
        """
        Write class names to file.

        Args:
            classes (List[str]): List of class names
            output_path (str): Output file path

        Returns:
            bool: True if successful
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not self.ensure_directory(output_dir):
            return False

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for cls in classes:
                    f.write(f"{cls}\n")
            self.logger.info(f"Saved classes to: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to write classes file {output_path}: {e}")
            return False

    def get_image_info(self, image_path: str) -> Tuple[int, int, int]:
        """
        Get image dimensions from file.

        Args:
            image_path (str): Path to image file

        Returns:
            Tuple of (width, height, channels)
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
                channels = len(img.getbands())
                return width, height, channels
        except Exception as e:
            self.logger.warning(f"Could not read image {image_path}: {e}. Using defaults.")
            return Config.DEFAULT_IMAGE_WIDTH, Config.DEFAULT_IMAGE_HEIGHT, Config.DEFAULT_IMAGE_CHANNELS

    def _print_progress(self, current: int, total: int, prefix: str = ""):
        """Print progress information."""
        if self.verbose:
            percent = (current / total) * 100
            self.logger.info(f"{prefix}Progress: {current}/{total} ({percent:.1f}%)")


class LabelBasedConverter(BaseConverter):
    """Base class for converters that use label module handlers."""

    def __init__(self, verbose: bool = None, segmentation: bool = False):
        """
        Initialize the label-based converter.

        Args:
            verbose (bool, optional): Whether to print progress information.
                If None, uses Config.VERBOSE.
            segmentation (bool): Whether to enforce segmentation annotations.
                If True, only annotations with segmentation data will be processed.
        """
        super().__init__(verbose)
        self.segmentation = segmentation

    def _validate_segmentation_annotations(self, annotations: List[Dict]) -> bool:
        """
        Validate that annotations have segmentation data when segmentation mode is enabled.

        Args:
            annotations: List of annotation dictionaries

        Returns:
            True if all annotations have segmentation data (or segmentation mode is off)
        """
        if not self.segmentation:
            return True

        for ann in annotations:
            # Check if annotation has segmentation data
            # segmentation field should be a list with at least one polygon
            if not ann.get("segmentation") or not ann["segmentation"]:
                return False
            # Check if the first polygon has coordinates
            if not ann["segmentation"][0]:
                return False

        return True

    def _extract_unique_categories(self, unified_data: List[Dict]) -> List[str]:
        """
        Extract unique category names from unified data format.

        Args:
            unified_data: List of unified format image data dictionaries

        Returns:
            Sorted list of unique category names (sorted by category_id if available, otherwise by name)
        """
        # Collect (category_id, category_name) pairs
        category_items = []
        for img_data in unified_data:
            for ann in img_data.get("annotations", []):
                cat_name = ann.get("category_name")
                cat_id = ann.get("category_id")
                if cat_name:
                    # Use category_id if available, otherwise use a large number to sort at end
                    sort_key = (cat_id if cat_id is not None else 999999, cat_name)
                    category_items.append((sort_key, cat_name))

        # Remove duplicates while preserving first occurrence order for same sort_key
        unique_categories = {}
        for sort_key, cat_name in category_items:
            if sort_key not in unique_categories:
                unique_categories[sort_key] = cat_name

        # Sort by sort_key (category_id first, then name)
        sorted_items = sorted(unique_categories.items(), key=lambda x: x[0])

        # Return only category names in sorted order
        return [cat_name for _, cat_name in sorted_items]

    def _get_or_create_classes_path(self, output_dir: str, categories: List[str] = None) -> str:
        """
        Get or create a classes file path in the output directory.

        Args:
            output_dir: Output directory path
            categories: Optional list of categories to write to the file

        Returns:
            Path to the classes file
        """
        import os
        from ..config import Config

        classes_path = os.path.join(output_dir, Config.YOLO_CLASSES_FILENAME)

        if categories and not os.path.exists(classes_path):
            self.write_classes_file(categories, classes_path)
            self.logger.info(f"Created classes file: {classes_path}")

        return classes_path

    def _create_labels_directory(self, output_dir: str) -> str:
        """
        Create labels directory in the output directory.

        Args:
            output_dir: Output directory path

        Returns:
            Path to the labels directory
        """
        import os
        from ..config import Config

        labels_dir = os.path.join(output_dir, Config.YOLO_LABELS_DIRNAME)
        self.ensure_directory(labels_dir)
        return labels_dir
