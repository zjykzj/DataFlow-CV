# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:40
@File    : yolo_to_coco.py
@Author  : zj
@Description: YOLO to COCO format converter
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from .base import BaseConverter
from ..config import Config


class YoloToCocoConverter(BaseConverter):
    """Convert YOLO label format to COCO JSON format."""

    def convert(
        self,
        image_dir: str,
        yolo_labels_dir: str,
        yolo_class_path: str,
        coco_json_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convert YOLO labels to COCO JSON format.

        Args:
            image_dir (str): Directory containing image files
            yolo_labels_dir (str): Directory containing YOLO label files
            yolo_class_path (str): Path to YOLO class names file
            coco_json_path (str): Path to save COCO JSON file
            **kwargs: Additional conversion options

        Returns:
            Dictionary with conversion statistics
        """
        # Validate inputs
        if not self.validate_input_path(image_dir, is_dir=True):
            raise ValueError(f"Invalid image directory: {image_dir}")

        if not self.validate_input_path(yolo_labels_dir, is_dir=True):
            raise ValueError(f"Invalid YOLO labels directory: {yolo_labels_dir}")

        if not self.validate_input_path(yolo_class_path, is_dir=False):
            raise ValueError(f"Invalid YOLO classes file: {yolo_class_path}")

        # Validate and create output directory if needed
        output_dir = os.path.dirname(coco_json_path)
        if output_dir and not self.validate_output_path(output_dir, is_dir=True, create=True):
            raise ValueError(f"Invalid output directory for COCO JSON: {output_dir}")

        # Read class names
        class_names = self.read_classes_file(yolo_class_path)
        if not class_names:
            raise ValueError(f"No class names found in: {yolo_class_path}")

        self.logger.info(f"Loaded {len(class_names)} class names from: {yolo_class_path}")

        # Get image files
        image_files = self.get_image_files(image_dir)
        if not image_files:
            raise ValueError(f"No image files found in: {image_dir}")

        self.logger.info(f"Found {len(image_files)} image files in: {image_dir}")

        # Build COCO data structure
        coco_data = self._build_coco_structure(class_names)

        # Process images and annotations
        stats = self._process_images_and_labels(
            image_files, yolo_labels_dir, class_names, coco_data
        )

        # Save COCO JSON
        success = self._save_coco_json(coco_data, coco_json_path)
        if not success:
            raise ValueError(f"Failed to save COCO JSON to: {coco_json_path}")

        # Add summary information
        stats.update({
            "image_dir": image_dir,
            "yolo_labels_dir": yolo_labels_dir,
            "yolo_class_path": yolo_class_path,
            "coco_json_path": coco_json_path,
            "total_classes": len(class_names),
            "total_images": len(image_files),
            "coco_saved": success,
        })

        self.logger.info(f"Conversion completed: {stats}")
        return stats

    def _build_coco_structure(self, class_names: List[str]) -> Dict[str, Any]:
        """
        Build basic COCO data structure.

        Args:
            class_names: List of class names

        Returns:
            COCO data dictionary
        """
        # Create categories
        categories = []
        for i, class_name in enumerate(class_names):
            category = {
                "id": i + 1,  # COCO category IDs usually start from 1
                "name": class_name,
                "supercategory": "none"
            }
            categories.append(category)

        return {
            "info": Config.COCO_DEFAULT_INFO.copy(),
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories
        }

    def _process_images_and_labels(
        self,
        image_files: List[str],
        labels_dir: str,
        class_names: List[str],
        coco_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process images and corresponding YOLO label files.

        Args:
            image_files: List of image file paths
            labels_dir: Directory containing YOLO label files
            class_names: List of class names
            coco_data: COCO data dictionary to populate

        Returns:
            Dictionary with processing statistics
        """
        stats = {
            "images_processed": 0,
            "images_with_annotations": 0,
            "annotations_processed": 0,
            "images_without_labels": 0,
            "failed_images": 0,
            "total_label_files": 0,
        }

        # Get label files for quick lookup
        label_files = self.get_label_files(labels_dir)
        stats["total_label_files"] = len(label_files)

        label_dict = {}
        for label_file in label_files:
            base_name = os.path.splitext(os.path.basename(label_file))[0]
            label_dict[base_name] = label_file

        # Initialize annotation ID counter
        annotation_id = 1

        # Process each image
        for i, image_path in enumerate(image_files):
            if self.verbose and i % 100 == 0:
                self._print_progress(i, len(image_files), prefix="Processing images")

            # Get image info
            image_id = i + 1  # COCO image IDs usually start from 1
            image_file_name = os.path.basename(image_path)
            image_base_name = os.path.splitext(image_file_name)[0]

            # Get image dimensions
            width, height, channels = self.get_image_info(image_path)

            # Add image to COCO data
            image_entry = {
                "id": image_id,
                "file_name": image_file_name,
                "width": width,
                "height": height,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            }
            coco_data["images"].append(image_entry)

            # Check for corresponding label file
            label_file = label_dict.get(image_base_name)
            if not label_file:
                stats["images_without_labels"] += 1
                stats["images_processed"] += 1
                continue

            # Read and parse label file
            annotations = self._read_yolo_label_file(
                label_file, image_id, width, height, class_names, annotation_id
            )

            if annotations:
                coco_data["annotations"].extend(annotations)
                annotation_id += len(annotations)
                stats["annotations_processed"] += len(annotations)
                stats["images_with_annotations"] += 1

            stats["images_processed"] += 1

        return stats

    def _read_yolo_label_file(
        self,
        label_file: str,
        image_id: int,
        image_width: int,
        image_height: int,
        class_names: List[str],
        start_annotation_id: int
    ) -> List[Dict]:
        """
        Read YOLO label file and convert to COCO annotations.

        Args:
            label_file: Path to YOLO label file
            image_id: COCO image ID
            image_width: Image width in pixels
            image_height: Image height in pixels
            class_names: List of class names
            start_annotation_id: Starting annotation ID

        Returns:
            List of COCO annotation dictionaries
        """
        annotations = []
        annotation_id = start_annotation_id

        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                # Parse YOLO line
                parts = line.split()
                if len(parts) < 1:
                    continue

                try:
                    class_index = int(parts[0])
                    if class_index < 0 or class_index >= len(class_names):
                        self.logger.warning(
                            f"Invalid class index {class_index} in {label_file}, line {line_num}"
                        )
                        continue

                    # Determine if it's bounding box or segmentation
                    if len(parts) == 5:
                        # Bounding box format: class_id x_center y_center width height
                        annotation = self._parse_bbox_annotation(
                            parts, class_index, image_id, annotation_id,
                            image_width, image_height
                        )
                    else:
                        # Segmentation format: class_id x1 y1 x2 y2 ...
                        annotation = self._parse_segmentation_annotation(
                            parts, class_index, image_id, annotation_id,
                            image_width, image_height
                        )

                    if annotation:
                        annotations.append(annotation)
                        annotation_id += 1

                except (ValueError, IndexError) as e:
                    self.logger.warning(
                        f"Error parsing line {line_num} in {label_file}: {e}"
                    )
                    continue

        except (OSError, UnicodeDecodeError) as e:
            self.logger.error(f"Failed to read label file {label_file}: {e}")

        return annotations

    def _parse_bbox_annotation(
        self,
        parts: List[str],
        class_index: int,
        image_id: int,
        annotation_id: int,
        image_width: int,
        image_height: int
    ) -> Optional[Dict]:
        """
        Parse YOLO bounding box annotation.

        Args:
            parts: Split line parts
            class_index: YOLO class index
            image_id: COCO image ID
            annotation_id: COCO annotation ID
            image_width: Image width
            image_height: Image height

        Returns:
            COCO annotation dictionary or None
        """
        try:
            x_center = float(parts[1])
            y_center = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])

            # Convert normalized coordinates to absolute pixels
            x_min = (x_center - width_norm / 2) * image_width
            y_min = (y_center - height_norm / 2) * image_height
            width = width_norm * image_width
            height = height_norm * image_height

            # Ensure coordinates are within image bounds
            x_min = max(0, min(x_min, image_width - 1))
            y_min = max(0, min(y_min, image_height - 1))
            width = max(1, min(width, image_width - x_min))
            height = max(1, min(height, image_height - y_min))

            # Create COCO annotation
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_index + 1,  # COCO category IDs start from 1
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "segmentation": [],
                "iscrowd": 0
            }

            return annotation

        except (ValueError, IndexError) as e:
            self.logger.warning(f"Error parsing bbox annotation (line format incorrect): {e}")
            return None

    def _parse_segmentation_annotation(
        self,
        parts: List[str],
        class_index: int,
        image_id: int,
        annotation_id: int,
        image_width: int,
        image_height: int
    ) -> Optional[Dict]:
        """
        Parse YOLO segmentation annotation.

        Args:
            parts: Split line parts
            class_index: YOLO class index
            image_id: COCO image ID
            annotation_id: COCO annotation ID
            image_width: Image width
            image_height: Image height

        Returns:
            COCO annotation dictionary or None
        """
        try:
            # Parse normalized polygon coordinates
            coords = [float(x) for x in parts[1:]]
            if len(coords) < 6:  # Need at least 3 points (6 coordinates)
                self.logger.warning(f"Insufficient coordinates for segmentation: {len(coords)}")
                return None

            # Convert normalized coordinates to absolute pixels
            polygon = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = coords[i] * image_width
                    y = coords[i + 1] * image_height
                    polygon.extend([x, y])

            # Calculate bounding box from polygon
            x_coords = polygon[0::2]
            y_coords = polygon[1::2]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            width = x_max - x_min
            height = y_max - y_min
            area = 0.5 * abs(sum(
                x_coords[i] * y_coords[(i + 1) % len(x_coords)] -
                x_coords[(i + 1) % len(x_coords)] * y_coords[i]
                for i in range(len(x_coords))
            ))

            # Create COCO annotation
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_index + 1,  # COCO category IDs start from 1
                "bbox": [x_min, y_min, width, height],
                "area": area,
                "segmentation": [polygon],  # COCO expects list of polygons
                "iscrowd": 0
            }

            return annotation

        except (ValueError, IndexError, ZeroDivisionError) as e:
            self.logger.warning(f"Error parsing segmentation annotation: {e}")
            return None

    def _save_coco_json(self, coco_data: Dict[str, Any], output_path: str) -> bool:
        """
        Save COCO data to JSON file.

        Args:
            coco_data: COCO data dictionary
            output_path: Output file path

        Returns:
            bool: True if successful
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not self.ensure_directory(output_dir):
                return False

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved COCO JSON to: {output_path}")
            return True

        except (OSError, TypeError, ValueError) as e:
            self.logger.error(f"Failed to save COCO JSON {output_path}: {e}")
            return False
