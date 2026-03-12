# -*- coding: utf-8 -*-
"""
@Time    : 2026/3/10 20:00
@File    : coco_and_labelme.py
@Author  : zj
@Description: COCO and LabelMe format converters

This module provides bidirectional conversion between COCO and LabelMe formats,
reusing the label module handlers for consistent parsing and serialization.
"""

import os
from typing import Dict, List, Any

from .base import LabelBasedConverter
from ..config import Config
from ..label.coco import CocoHandler
from ..label.labelme import LabelMeHandler


class CocoToLabelMeConverter(LabelBasedConverter):
    """Convert COCO JSON format to LabelMe format."""

    def convert(self, coco_json_path: str, output_dir: str, segmentation: bool = False) -> Dict[str, Any]:
        """
        Convert COCO JSON file to LabelMe format.

        Args:
            coco_json_path: Path to COCO JSON file
            output_dir: Output directory where LabelMe JSON files will be created
            segmentation: Whether to enforce segmentation annotations.
                If True, only annotations with polygon segmentation data will be processed,
                bounding box annotations will be skipped. If False, both bounding box and
                segmentation annotations are processed.

        Returns:
            Dictionary with conversion statistics

        Raises:
            ValueError: If input paths are invalid or conversion fails
        """
        self.segmentation = segmentation

        # 1. Validate input and output paths
        if not self.validate_input_path(coco_json_path, is_dir=False):
            raise ValueError(f"Invalid COCO JSON file: {coco_json_path}")

        if not self.validate_output_path(output_dir, is_dir=True, create=True):
            raise ValueError(f"Invalid output directory: {output_dir}")

        self.logger.info(f"Converting COCO to LabelMe: {coco_json_path} -> {output_dir}")

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

        # 3. Validate segmentation format if required
        if segmentation:
            for img_data in unified_data:
                if not self._validate_segmentation_annotations(img_data["annotations"]):
                    raise ValueError(f"Image {img_data['image_id']} missing segmentation annotations")

        # 4. Extract unique categories and write class.names file (optional but consistent)
        categories = self._extract_unique_categories(unified_data)
        classes_path = os.path.join(output_dir, Config.YOLO_CLASSES_FILENAME)
        if categories:
            if not self.write_classes_file(categories, classes_path):
                self.logger.warning(f"Failed to write classes file: {classes_path}")
            else:
                self.logger.info(f"Written {len(categories)} categories to {classes_path}")

        # 5. Use LabelMeHandler to write LabelMe format
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

        # 6. Return statistics
        total_annotations = sum(len(img["annotations"]) for img in unified_data)
        stats = {
            "images_processed": len(unified_data),
            "annotations_processed": total_annotations,
            "categories_found": len(categories),
            "output_dir": output_dir,
            "classes_file": classes_path if categories else None,
            "segmentation_mode": segmentation,
        }

        self.logger.info(f"Conversion completed successfully: {stats}")
        return stats


class LabelMeToCocoConverter(LabelBasedConverter):
    """Convert LabelMe format to COCO JSON format."""

    def convert(self, label_dir: str, classes_path: str, output_json_path: str,
                segmentation: bool = False) -> Dict[str, Any]:
        """
        Convert LabelMe format to COCO JSON.

        Args:
            label_dir: Directory containing LabelMe JSON files
            classes_path: Path to class names file (e.g., class.names)
            output_json_path: Path to save COCO JSON file
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

        # 1. Validate input paths
        if not self.validate_input_path(label_dir, is_dir=True):
            raise ValueError(f"Invalid label directory: {label_dir}")

        if not self.validate_input_path(classes_path, is_dir=False):
            raise ValueError(f"Invalid classes file: {classes_path}")

        # Output directory must exist (file will be created)
        output_dir = os.path.dirname(output_json_path) if os.path.dirname(output_json_path) else "."
        if not self.validate_output_path(output_dir, is_dir=True, create=True):
            raise ValueError(f"Invalid output directory: {output_dir}")

        self.logger.info(f"Converting LabelMe to COCO: {label_dir} -> {output_json_path}")

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

        # 4. Read classes for category mapping (even though LabelMe has categories,
        #    we need to ensure consistent ordering and IDs)
        try:
            with open(classes_path, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f if line.strip()]
        except Exception as e:
            raise ValueError(f"Failed to read classes file: {e}")

        if not classes:
            raise ValueError(f"Classes file is empty: {classes_path}")

        # 5. Use CocoHandler to convert unified format to COCO format
        coco_handler = CocoHandler(verbose=self.verbose)
        try:
            coco_data = coco_handler.convert_from_unified_format(unified_data)
        except ValueError as e:
            if "输入数据为空" in str(e):
                # Create empty COCO structure manually
                coco_data = {
                    "info": Config.COCO_DEFAULT_INFO,
                    "licenses": [],
                    "images": [],
                    "annotations": [],
                    "categories": []
                }
            else:
                raise

        # 6. Write COCO JSON file
        success = coco_handler.write(coco_data, output_json_path)
        if not success:
            raise ValueError(f"Failed to write COCO JSON file: {output_json_path}")

        # 7. Return statistics
        total_images = len(coco_data.get("images", []))
        total_annotations = len(coco_data.get("annotations", []))
        total_categories = len(coco_data.get("categories", []))

        stats = {
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