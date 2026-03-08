# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:40
@File    : coco_to_yolo.py
@Author  : zj
@Description: COCO to YOLO format converter
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from .base import BaseConverter
from ..config import Config


class CocoToYoloConverter(BaseConverter):
    """Convert COCO JSON format to YOLO label format."""

    def convert(self, coco_json_path: str, output_dir: str, **kwargs) -> Dict[str, Any]:
        """
        Convert COCO JSON file to YOLO format.

        Args:
            coco_json_path (str): Path to COCO JSON file
            output_dir (str): Output directory where labels/ and class.names will be created
            **kwargs: Additional conversion options

        Returns:
            Dictionary with conversion statistics
        """
        # Validate input
        if not self.validate_input_path(coco_json_path, is_dir=False):
            raise ValueError(f"Invalid COCO JSON file: {coco_json_path}")

        # Validate and create output directory
        if not self.validate_output_path(output_dir, is_dir=True, create=True):
            raise ValueError(f"Invalid output directory: {output_dir}")

        # Read COCO JSON
        coco_data = self._read_coco_json(coco_json_path)
        if not coco_data:
            raise ValueError(f"Failed to read or parse COCO JSON: {coco_json_path}")

        # Extract data
        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])
        categories = coco_data.get("categories", [])

        if not images:
            self.logger.warning(f"No images found in COCO JSON: {coco_json_path}")
        if not categories:
            self.logger.warning(f"No categories found in COCO JSON: {coco_json_path}")

        # Create category mapping (COCO id -> YOLO class index)
        category_mapping = self._create_category_mapping(categories)

        # Write class names file
        classes_output_path = os.path.join(output_dir, Config.YOLO_CLASSES_FILENAME)
        self._write_class_names(category_mapping, classes_output_path)

        # Create labels directory
        labels_dir = os.path.join(output_dir, Config.YOLO_LABELS_DIRNAME)
        if not self.ensure_directory(labels_dir):
            raise ValueError(f"Failed to create labels directory: {labels_dir}")

        # Process images and annotations
        stats = self._process_annotations(images, annotations, category_mapping, labels_dir)

        # Add summary information
        stats.update({
            "coco_json_path": coco_json_path,
            "output_dir": output_dir,
            "labels_dir": labels_dir,
            "classes_file": classes_output_path,
            "total_categories": len(categories),
            "total_images": len(images),
            "total_annotations": len(annotations),
        })

        self.logger.info(f"Conversion completed: {stats}")
        return stats

    def _read_coco_json(self, coco_json_path: str) -> Dict[str, Any]:
        """Read and parse COCO JSON file."""
        try:
            with open(coco_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Basic validation of COCO structure
            if not isinstance(data, dict):
                self.logger.warning(f"COCO JSON {coco_json_path} is not a dictionary")
                return data

            # Check for expected keys (warning only, as some COCO files may have variations)
            expected_keys = {"images", "annotations", "categories"}
            missing_keys = expected_keys - set(data.keys())
            if missing_keys:
                self.logger.warning(f"COCO JSON {coco_json_path} missing keys: {missing_keys}")

            return data

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.logger.error(f"Failed to parse COCO JSON {coco_json_path}: {e}")
            return {}
        except OSError as e:
            self.logger.error(f"Failed to read COCO JSON {coco_json_path}: {e}")
            return {}

    def _create_category_mapping(self, categories: List[Dict]) -> Dict[int, Tuple[int, str]]:
        """
        Create mapping from COCO category ID to YOLO class index.

        Args:
            categories: List of category dictionaries from COCO

        Returns:
            Dict mapping COCO category ID to (yolo_class_index, category_name)
        """
        # Sort categories by ID for consistent ordering
        sorted_categories = sorted(categories, key=lambda x: x.get("id", 0))

        mapping = {}
        for idx, cat in enumerate(sorted_categories):
            coco_id = cat.get("id", 0)
            name = cat.get("name", f"class_{coco_id}")
            mapping[coco_id] = (idx, name)

        self.logger.info(f"Created category mapping: {len(mapping)} categories")
        return mapping

    def _write_class_names(self, category_mapping: Dict[int, Tuple[int, str]], output_path: str):
        """Write class names to file in YOLO format."""
        # Sort by YOLO class index
        class_items = [(idx, name) for _, (idx, name) in category_mapping.items()]
        class_items.sort(key=lambda x: x[0])

        class_names = [name for _, name in class_items]
        success = self.write_classes_file(class_names, output_path)
        if success:
            self.logger.info(f"Saved {len(class_names)} class names to: {output_path}")
        else:
            raise ValueError(f"Failed to write class names to: {output_path}")

    def _process_annotations(
        self,
        images: List[Dict],
        annotations: List[Dict],
        category_mapping: Dict[int, Tuple[int, str]],
        labels_dir: str
    ) -> Dict[str, Any]:
        """
        Process annotations and write YOLO label files.

        Args:
            images: List of image dictionaries
            annotations: List of annotation dictionaries
            category_mapping: Category ID to (class_index, name) mapping
            labels_dir: Directory to save YOLO label files

        Returns:
            Dictionary with processing statistics
        """
        # Create image ID to image mapping for quick lookup
        image_dict = {img["id"]: img for img in images}

        # Group annotations by image ID
        annotations_by_image = {}
        for ann in annotations:
            image_id = ann.get("image_id")
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)

        stats = {
            "images_processed": 0,
            "images_with_annotations": 0,
            "annotations_processed": 0,
            "images_without_annotations": 0,
            "failed_images": 0,
        }

        # Process each image
        for i, image in enumerate(images):
            image_id = image.get("id")
            image_name = image.get("file_name", f"image_{image_id}")
            image_width = image.get("width", Config.DEFAULT_IMAGE_WIDTH)
            image_height = image.get("height", Config.DEFAULT_IMAGE_HEIGHT)

            if self.verbose and i % 100 == 0:
                self._print_progress(i, len(images), prefix="Processing images")

            # Get annotations for this image
            image_anns = annotations_by_image.get(image_id, [])

            if not image_anns:
                stats["images_without_annotations"] += 1
                # Create empty label file
                self._create_empty_label_file(image_name, labels_dir)
                stats["images_processed"] += 1
                continue

            # Convert annotations to YOLO format
            yolo_lines = []
            for ann in image_anns:
                yolo_line = self._convert_annotation_to_yolo(
                    ann, image_width, image_height, category_mapping
                )
                if yolo_line:
                    yolo_lines.append(yolo_line)
                    stats["annotations_processed"] += 1

            # Write label file
            if self._write_yolo_label_file(image_name, labels_dir, yolo_lines):
                stats["images_processed"] += 1
                stats["images_with_annotations"] += 1
            else:
                stats["failed_images"] += 1

        return stats

    def _convert_annotation_to_yolo(
        self,
        annotation: Dict,
        image_width: int,
        image_height: int,
        category_mapping: Dict[int, Tuple[int, str]]
    ) -> Optional[str]:
        """
        Convert COCO annotation to YOLO format string.

        Args:
            annotation: COCO annotation dictionary
            image_width: Image width in pixels
            image_height: Image height in pixels
            category_mapping: Category mapping

        Returns:
            YOLO format string or None if conversion fails
        """
        # Get category ID and map to YOLO class index
        category_id = annotation.get("category_id")
        if category_id not in category_mapping:
            self.logger.warning(f"Unknown category ID {category_id} in annotation")
            return None

        class_index, _ = category_mapping[category_id]

        # Handle bounding box
        bbox = annotation.get("bbox")
        if bbox and len(bbox) == 4:
            # COCO bbox: [x_min, y_min, width, height]
            x_min, y_min, width, height = bbox

            # Ensure bbox is within image bounds
            x_min = max(0, min(x_min, image_width - 1))
            y_min = max(0, min(y_min, image_height - 1))
            width = max(1, min(width, image_width - x_min))
            height = max(1, min(height, image_height - y_min))

            # Convert to YOLO format: [x_center, y_center, width, height] normalized
            x_center = (x_min + width / 2) / image_width
            y_center = (y_min + height / 2) / image_height
            width_norm = width / image_width
            height_norm = height / image_height

            # Format: class_index x_center y_center width height
            return f"{class_index} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"

        # Handle segmentation (polygon) if needed
        segmentation = annotation.get("segmentation")
        if segmentation and Config.YOLO_SEGMENTATION:
            # Convert segmentation polygon to YOLO format
            return self._convert_segmentation_to_yolo(
                segmentation, class_index, image_width, image_height
            )

        self.logger.warning(f"Annotation missing bbox and segmentation: {annotation.get('id')}")
        return None

    def _convert_segmentation_to_yolo(
        self,
        segmentation: List,
        class_index: int,
        image_width: int,
        image_height: int
    ) -> Optional[str]:
        """
        Convert COCO segmentation polygon to YOLO format.

        Args:
            segmentation: COCO segmentation polygon
            class_index: YOLO class index
            image_width: Image width
            image_height: Image height

        Returns:
            YOLO format string for segmentation
        """
        # Note: YOLO segmentation format: class_id x1 y1 x2 y2 ...
        # where coordinates are normalized
        try:
            if isinstance(segmentation, list):
                # Handle polygon format
                if segmentation and isinstance(segmentation[0], list):
                    # COCO segmentation: list of polygons, take the first one
                    polygon = segmentation[0]
                else:
                    polygon = segmentation

                # Normalize coordinates
                normalized_coords = []
                for i in range(0, len(polygon), 2):
                    if i + 1 < len(polygon):
                        x = polygon[i] / image_width
                        y = polygon[i + 1] / image_height
                        normalized_coords.extend([x, y])

                if normalized_coords:
                    coords_str = " ".join(f"{coord:.6f}" for coord in normalized_coords)
                    return f"{class_index} {coords_str}"
        except (TypeError, ValueError, ZeroDivisionError) as e:
            self.logger.warning(f"Failed to convert segmentation: {e}")

        return None

    def _write_yolo_label_file(
        self,
        image_name: str,
        labels_dir: str,
        yolo_lines: List[str]
    ) -> bool:
        """
        Write YOLO label lines to file.

        Args:
            image_name: Original image filename (without extension)
            labels_dir: Directory to save label file
            yolo_lines: List of YOLO format strings

        Returns:
            bool: True if successful
        """
        try:
            # Remove extension from image name
            base_name = os.path.splitext(os.path.basename(image_name))[0]
            label_filename = f"{base_name}{Config.get_yolo_label_extension()}"
            label_path = os.path.join(labels_dir, label_filename)

            with open(label_path, 'w', encoding='utf-8') as f:
                for line in yolo_lines:
                    f.write(f"{line}\n")

            return True
        except Exception as e:
            self.logger.error(f"Failed to write label file for {image_name}: {e}")
            return False

    def _create_empty_label_file(self, image_name: str, labels_dir: str) -> bool:
        """Create empty label file for images without annotations."""
        try:
            base_name = os.path.splitext(os.path.basename(image_name))[0]
            label_filename = f"{base_name}{Config.get_yolo_label_extension()}"
            label_path = os.path.join(labels_dir, label_filename)

            # Check if file already exists
            if os.path.exists(label_path) and not Config.OVERWRITE_EXISTING:
                return True

            # Create empty file
            with open(label_path, 'w', encoding='utf-8') as f:
                pass  # Empty file

            return True
        except Exception as e:
            self.logger.error(f"Failed to create empty label file for {image_name}: {e}")
            return False
