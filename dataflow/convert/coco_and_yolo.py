# -*- coding: utf-8 -*-
"""
@Time    : 2026/3/10 20:00
@File    : coco_and_yolo.py
@Author  : zj
@Description: COCO and YOLO format converters

This module provides bidirectional conversion between COCO and YOLO formats,
reusing the label module handlers for consistent parsing and serialization.
"""

import os
from typing import Dict, List, Any

from .base import LabelBasedConverter
from ..config import Config
from ..label.coco import CocoHandler
from ..label.yolo import YoloHandler


class CocoToYoloConverter(LabelBasedConverter):
    """Convert COCO JSON format to YOLO label format."""

    def convert(self, coco_json_path: str, classes_path: str, output_dir: str, segmentation: bool = False) -> Dict[str, Any]:
        """
        Convert COCO JSON file to YOLO format.

        Args:
            coco_json_path: Path to COCO JSON file
            classes_path: Path to class names file (e.g., class.names)
            output_dir: Output directory where YOLO label files will be created
            segmentation: Whether to enforce segmentation annotations.
                If True, only annotations with segmentation data will be processed.

        Returns:
            Dictionary with conversion statistics

        Raises:
            ValueError: If input paths are invalid or conversion fails
        """
        self.segmentation = segmentation

        # 1. Validate input and output paths
        if not self.validate_input_path(coco_json_path, is_dir=False):
            raise ValueError(f"Invalid COCO JSON file: {coco_json_path}")

        if not self.validate_input_path(classes_path, is_dir=False):
            raise ValueError(f"Invalid classes file: {classes_path}")

        if not self.validate_output_path(output_dir, is_dir=True, create=True):
            raise ValueError(f"Invalid output directory: {output_dir}")

        self.logger.info(f"Converting COCO to YOLO: {coco_json_path} -> {output_dir}")

        # 2. Use CocoHandler to read COCO data and convert to unified format
        coco_handler = CocoHandler(verbose=self.verbose)
        try:
            coco_data = coco_handler.read(coco_json_path)
        except Exception as e:
            raise ValueError(f"Failed to read COCO JSON file: {e}")

        # Convert to unified format with segmentation filtering
        unified_data = coco_handler.convert_to_unified_format(
            coco_data, image_dir="", require_segmentation=segmentation
        )

        if not unified_data:
            self.logger.warning("No valid annotations found in COCO data")
            # Still create output structure but with empty files

        # 3. Validate segmentation format if required
        if segmentation:
            for img_data in unified_data:
                if not self._validate_segmentation_annotations(img_data["annotations"]):
                    raise ValueError(f"Image {img_data['image_id']} missing segmentation annotations")
        else:
            # When segmentation=False, remove segmentation fields to force YOLO format to use bbox
            for img_data in unified_data:
                for ann in img_data.get("annotations", []):
                    ann.pop("segmentation", None)

        # 4. Read provided classes file and validate
        categories = self.read_classes_file(classes_path)
        if not categories:
            raise ValueError(f"No categories found in classes file: {classes_path}")

        # 5. Extract unique categories from COCO data and validate against provided classes
        data_categories = self._extract_unique_categories(unified_data)
        if not data_categories:
            self.logger.warning("No categories found in COCO data")
        else:
            # Validate all categories in data are present in provided classes file
            for category in data_categories:
                if category not in categories:
                    raise ValueError(f"Category '{category}' found in COCO data but not in classes file")

        # 6. Use YoloHandler to write YOLO format directly to output_dir
        yolo_handler = YoloHandler(verbose=self.verbose)
        success = False
        if unified_data:
            success = yolo_handler.write_batch(unified_data, output_dir, classes_path)
        else:
            # Create empty directory structure
            success = True
            self.logger.info("No annotations to write, created empty directory structure")

        if not success:
            raise ValueError(f"Failed to write YOLO label files to {output_dir}")

        # 7. Return statistics
        total_annotations = sum(len(img["annotations"]) for img in unified_data)
        stats = {
            "images_processed": len(unified_data),
            "annotations_processed": total_annotations,
            "categories_found": len(categories),
            "categories_in_data": len(data_categories) if unified_data else 0,
            "output_dir": output_dir,
            "classes_file": classes_path,
            "segmentation_mode": segmentation,
        }

        self.logger.info(f"Conversion completed successfully: {stats}")
        return stats


class YoloToCocoConverter(LabelBasedConverter):
    """Convert YOLO label format to COCO JSON format."""

    def convert(self, image_dir: str, label_dir: str, classes_path: str,
                output_json_path: str, segmentation: bool = False) -> Dict[str, Any]:
        """
        Convert YOLO format to COCO JSON.

        Args:
            image_dir: Directory containing image files
            label_dir: Directory containing YOLO label files (.txt)
            classes_path: Path to YOLO class names file (e.g., class.names)
            output_json_path: Path to save COCO JSON file
            segmentation: Whether to enforce segmentation annotations.
                If True, only annotations with segmentation data will be processed.

        Returns:
            Dictionary with conversion statistics

        Raises:
            ValueError: If input paths are invalid or conversion fails
        """
        self.segmentation = segmentation

        # 1. Validate input paths
        if not self.validate_input_path(image_dir, is_dir=True):
            raise ValueError(f"Invalid image directory: {image_dir}")

        if not self.validate_input_path(label_dir, is_dir=True):
            raise ValueError(f"Invalid label directory: {label_dir}")

        if not self.validate_input_path(classes_path, is_dir=False):
            raise ValueError(f"Invalid classes file: {classes_path}")

        # Output directory must exist (file will be created)
        output_dir = os.path.dirname(output_json_path) if os.path.dirname(output_json_path) else "."
        if not self.validate_output_path(output_dir, is_dir=True, create=True):
            raise ValueError(f"Invalid output directory: {output_dir}")

        self.logger.info(f"Converting YOLO to COCO: {label_dir} -> {output_json_path}")

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

        # 4. Read classes for category mapping
        try:
            classes = yolo_handler.read_classes(classes_path)
        except Exception as e:
            raise ValueError(f"Failed to read classes file: {e}")

        # 5. Use CocoHandler to convert unified format to COCO format
        coco_handler = CocoHandler(verbose=self.verbose)
        coco_data = coco_handler.convert_from_unified_format(unified_data)

        # 6. Write COCO JSON file
        success = coco_handler.write(coco_data, output_json_path)
        if not success:
            raise ValueError(f"Failed to write COCO JSON file: {output_json_path}")

        # 7. Return statistics
        total_images = len(coco_data.get("images", []))
        total_annotations = len(coco_data.get("annotations", []))
        total_categories = len(coco_data.get("categories", []))

        stats = {
            "image_dir": image_dir,
            "label_dir": label_dir,
            "classes_file": classes_path,
            "coco_json_path": output_json_path,
            "images_processed": total_images,
            "annotations_processed": total_annotations,
            "categories_found": total_categories,
            "segmentation_mode": segmentation,
        }

        self.logger.info(f"Conversion completed successfully: {stats}")
        return stats