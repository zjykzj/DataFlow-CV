# -*- coding: utf-8 -*-
"""
@Time    : 2026/3/10 20:00
@File    : yolo_and_labelme.py
@Author  : zj
@Description: YOLO and LabelMe format converters

This module provides bidirectional conversion between YOLO and LabelMe formats,
reusing the label module handlers for consistent parsing and serialization.
"""

import os
from typing import Dict, List, Any

from .base import LabelBasedConverter
from ..config import Config
from ..label.yolo import YoloHandler
from ..label.labelme import LabelMeHandler


class YoloToLabelMeConverter(LabelBasedConverter):
    """Convert YOLO format to LabelMe format."""

    def convert(self, image_dir: str, label_dir: str, classes_path: str,
                output_dir: str, segmentation: bool = False) -> Dict[str, Any]:
        """
        Convert YOLO format to LabelMe format.

        Args:
            image_dir: Directory containing image files
            label_dir: Directory containing YOLO label files (.txt)
            classes_path: Path to YOLO class names file (e.g., class.names)
            output_dir: Output directory where LabelMe JSON files will be created
            segmentation: Whether to enforce segmentation annotations.
                If True, detection annotations (4 coordinates) will be converted to polygons
                from bounding boxes, and segmentation annotations (6+ coordinates) will be
                processed normally. If False, automatic format detection is used.

        Returns:
            Dictionary with conversion statistics

        Raises:
            ValueError: If input paths are invalid or conversion fails
        """
        self.segmentation = segmentation

        # 1. Validate input and output paths
        if not self.validate_input_path(image_dir, is_dir=True):
            raise ValueError(f"Invalid image directory: {image_dir}")

        if not self.validate_input_path(label_dir, is_dir=True):
            raise ValueError(f"Invalid label directory: {label_dir}")

        if not self.validate_input_path(classes_path, is_dir=False):
            raise ValueError(f"Invalid classes file: {classes_path}")

        if not self.validate_output_path(output_dir, is_dir=True, create=True):
            raise ValueError(f"Invalid output directory: {output_dir}")

        self.logger.info(f"Converting YOLO to LabelMe: {label_dir} -> {output_dir}")

        # 2. Use YoloHandler to read YOLO data in batch
        yolo_handler = YoloHandler(verbose=self.verbose)
        try:
            unified_data = yolo_handler.read_batch(
                label_dir, image_dir, classes_path, require_segmentation=segmentation
            )
        except Exception as e:
            raise ValueError(f"Failed to read YOLO data: {e}")

        if not unified_data:
            self.logger.warning("No valid annotations found in YOLO data")

        # 3. Validate segmentation format if required
        if segmentation:
            for img_data in unified_data:
                if not self._validate_segmentation_annotations(img_data["annotations"]):
                    raise ValueError(f"Image {img_data['image_id']} missing segmentation annotations")

        # 4. Use LabelMeHandler to write LabelMe format
        labelme_handler = LabelMeHandler(verbose=self.verbose)
        success = False
        if unified_data:
            success = labelme_handler.write_batch(unified_data, output_dir)
        else:
            # Create empty directory structure
            success = True
            self.logger.info("No annotations to write, created empty directory structure")

        if not success:
            raise ValueError(f"Failed to write LabelMe JSON files to {output_dir}")

        # 5. Return statistics
        total_annotations = sum(len(img["annotations"]) for img in unified_data)
        stats = {
            "image_dir": image_dir,
            "label_dir": label_dir,
            "classes_file": classes_path,
            "output_dir": output_dir,
            "images_processed": len(unified_data),
            "annotations_processed": total_annotations,
            "segmentation_mode": segmentation,
        }

        self.logger.info(f"Conversion completed successfully: {stats}")
        return stats


class LabelMeToYoloConverter(LabelBasedConverter):
    """Convert LabelMe format to YOLO format."""

    def convert(self, label_dir: str, classes_path: str, output_dir: str, segmentation: bool = False) -> Dict[str, Any]:
        """
        Convert LabelMe format to YOLO format.

        Args:
            label_dir: Directory containing LabelMe JSON files
            classes_path: Path to class names file (e.g., class.names)
            output_dir: Output directory where YOLO label files will be created
            segmentation: Whether to enforce segmentation annotations.
                If True, only polygon shapes (shape_type="polygon") will be processed,
                rectangle shapes will be skipped. If False, both rectangle and polygon
                shapes are processed.

        Returns:
            Dictionary with conversion statistics

        Raises:
            ValueError: If input paths are invalid or conversion fails
        """
        self.segmentation = segmentation

        # 1. Validate input and output paths
        if not self.validate_input_path(label_dir, is_dir=True):
            raise ValueError(f"Invalid label directory: {label_dir}")

        if not self.validate_input_path(classes_path, is_dir=False):
            raise ValueError(f"Invalid classes file: {classes_path}")

        if not self.validate_output_path(output_dir, is_dir=True, create=True):
            raise ValueError(f"Invalid output directory: {output_dir}")

        self.logger.info(f"Converting LabelMe to YOLO: {label_dir} -> {output_dir}")

        # 2. Use LabelMeHandler to read LabelMe data in batch
        labelme_handler = LabelMeHandler(verbose=self.verbose)
        try:
            unified_data = labelme_handler.read_batch(
                label_dir, require_segmentation=segmentation
            )
        except Exception as e:
            raise ValueError(f"Failed to read LabelMe data: {e}")

        if not unified_data:
            self.logger.warning("No valid annotations found in LabelMe data")

        # 3. Validate segmentation format if required
        if segmentation:
            for img_data in unified_data:
                if not self._validate_segmentation_annotations(img_data["annotations"]):
                    raise ValueError(f"Image {img_data['image_id']} missing segmentation annotations")

        # 4. Read provided classes file and validate
        categories = self.read_classes_file(classes_path)
        if not categories:
            raise ValueError(f"No categories found in classes file: {classes_path}")

        # 5. Extract unique categories from LabelMe data and validate against provided classes
        data_categories = self._extract_unique_categories(unified_data)
        if not data_categories:
            self.logger.warning("No categories found in LabelMe data")
        else:
            # Validate all categories in data are present in provided classes file
            for category in data_categories:
                if category not in categories:
                    raise ValueError(f"Category '{category}' found in LabelMe data but not in classes file")

        # 6. Use YoloHandler to write YOLO format directly to output_dir
        yolo_handler = YoloHandler(verbose=self.verbose)
        success = False
        if unified_data and categories:
            success = yolo_handler.write_batch(unified_data, output_dir, classes_path)
        else:
            # Create empty directory structure
            success = True
            self.logger.info("No annotations or categories to write, created empty directory structure")

        if not success:
            raise ValueError(f"Failed to write YOLO label files to {output_dir}")

        # 8. Return statistics
        total_annotations = sum(len(img["annotations"]) for img in unified_data)
        stats = {
            "label_dir": label_dir,
            "classes_file": classes_path,
            "output_dir": output_dir,
            "images_processed": len(unified_data),
            "annotations_processed": total_annotations,
            "categories_found": len(categories),
            "categories_in_data": len(data_categories) if unified_data else 0,
            "segmentation_mode": segmentation,
        }

        self.logger.info(f"Conversion completed successfully: {stats}")
        return stats