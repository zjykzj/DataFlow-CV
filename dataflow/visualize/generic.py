# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/9 21:00
@File    : generic.py
@Author  : zj
@Description: Generic visualizer base class using label handlers
"""

import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

from .base import BaseVisualizer


class GenericVisualizer(BaseVisualizer):
    """Generic visualizer base class using label handlers.

    Provides common visualization logic that works with any label format
    through label handlers. Supports both bounding boxes and segmentation
    polygons with automatic detection and strict validation modes.
    """

    def __init__(self, verbose: bool = None, segmentation: bool = False):
        """Initialize generic visualizer.

        Args:
            verbose: Whether to print progress information.
                If None, uses Config.VERBOSE.
            segmentation: Whether to force segmentation mode (strict validation).
                If True, annotations must contain valid segmentation data.
        """
        super().__init__(verbose)
        self.segmentation = segmentation  # Force segmentation mode
        self.label_handler = None  # Should be set by subclass

    def _validate_segmentation_format(self, annotations: List[Dict]) -> bool:
        """Validate annotations contain segmentation data (strict mode).

        Args:
            annotations: List of annotation dictionaries

        Returns:
            True if all annotations contain valid segmentation data

        Raises:
            ValueError: If segmentation mode is enabled but annotation lacks
                segmentation data
        """
        if not self.segmentation:
            return True

        for ann in annotations:
            if not ann.get("segmentation") or not ann["segmentation"][0]:
                raise ValueError(
                    f"Annotation missing segmentation data but segmentation mode is required. "
                    f"Category: {ann.get('category_name', 'unknown')}, "
                    f"Has bbox: {'bbox' in ann}"
                )
        return True

    def _draw_annotations(self, image: np.ndarray, annotations: List[Dict],
                         classes: List[str]) -> np.ndarray:
        """Draw annotations on image with automatic detection or forced segmentation.

        Args:
            image: Input image (H, W, C) as BGR numpy array
            annotations: List of annotation dictionaries in unified format
            classes: List of class names for color assignment

        Returns:
            Image with annotations drawn
        """
        result_image = image.copy()
        self.logger.debug(f"_draw_annotations called with {len(annotations)} annotations, {len(classes)} classes: {classes}")

        # Validate segmentation format if required (strict mode)
        if self.segmentation:
            self._validate_segmentation_format(annotations)

        for ann in annotations:
            category_id = ann.get("category_id", 0)
            category_name = ann.get("category_name", f"class_{category_id}")

            # Get color based on class name index in classes list
            class_idx = None
            original_category_name = category_name

            # 1. Try exact match
            try:
                class_idx = classes.index(category_name)
                color = self.get_color_for_class(class_idx, len(classes))
                # Debug logging for successful color assignment
                self.logger.debug(f"Color assigned: category_name='{category_name}', class_idx={class_idx}, color={color}")
            except ValueError:
                # 2. Try normalized match (strip whitespace, case-insensitive)
                normalized_name = category_name.strip()
                # Try case-insensitive match
                try:
                    # Find case-insensitive match
                    for idx, cls in enumerate(classes):
                        if cls.strip().lower() == normalized_name.lower():
                            class_idx = idx
                            break
                except Exception:
                    pass

                if class_idx is not None:
                    color = self.get_color_for_class(class_idx, len(classes))
                    self.logger.warning(f"Class '{original_category_name}' matched case-insensitively to '{classes[class_idx]}', using index {class_idx}")
                    self.logger.debug(f"Case-insensitive match: category_name='{original_category_name}', matched='{classes[class_idx]}', class_idx={class_idx}, color={color}")
                else:
                    # 3. Try to parse "class_X" format
                    if category_name.startswith("class_") and category_name[6:].isdigit():
                        try:
                            parsed_id = int(category_name[6:])
                            # Use parsed_id as fallback, but ensure it's within reasonable bounds
                            if parsed_id < len(classes) * 2:  # Allow some flexibility
                                class_idx = parsed_id % len(classes) if len(classes) > 0 else 0
                                self.logger.warning(f"Class '{category_name}' parsed as class_{parsed_id}, using index {class_idx}")
                            else:
                                self.logger.warning(f"Parsed class ID {parsed_id} from '{category_name}' is too large, using category_id")
                        except ValueError:
                            pass

                    # 4. Final fallback to category_id
                    if class_idx is None:
                        self.logger.warning(f"Class '{category_name}' not found in classes list, using category_id {category_id} for color")
                        # Ensure category_id is within bounds
                        if category_id < len(classes):
                            class_idx = category_id
                        else:
                            # If category_id is out of bounds, use modulo
                            class_idx = category_id % len(classes) if len(classes) > 0 else 0

                    color = self.get_color_for_class(class_idx, len(classes))
                    # Debug logging for fallback case
                    self.logger.debug(f"Fallback color: category_name='{original_category_name}', category_id={category_id}, class_idx={class_idx}, color={color}")

            # Determine what to draw based on mode and available data
            if self.segmentation:
                # Force segmentation mode: must draw polygon
                self._draw_segmentation_polygon(result_image, ann, color, category_name)
            else:
                # Automatic detection: prefer segmentation, fallback to bbox
                if ann.get("segmentation") and ann["segmentation"][0]:
                    self._draw_segmentation_polygon(result_image, ann, color, category_name)
                elif ann.get("bbox"):
                    self._draw_bounding_box(result_image, ann, color, category_name)
                else:
                    self.logger.warning(
                        f"Annotation has neither segmentation nor bbox: {category_name}"
                    )

        return result_image

    def _draw_segmentation_polygon(self, image: np.ndarray, annotation: Dict,
                                  color: Tuple[int, int, int], label: str):
        """Draw segmentation polygon from annotation.

        Args:
            image: Image to draw on
            annotation: Annotation dictionary with segmentation field
            color: BGR color tuple
            label: Class label text
        """
        segmentation = annotation["segmentation"][0]
        if len(segmentation) < 6:  # Need at least 3 points (x,y,x,y,x,y)
            self.logger.warning(
                f"Invalid segmentation: need at least 3 points, got {len(segmentation)//2}"
            )
            return

        # Convert flat list to list of (x, y) points
        points = [(segmentation[i], segmentation[i+1])
                 for i in range(0, len(segmentation), 2)]

        # Check if this annotation originated from RLE mask
        is_rle = False
        if self.highlight_rle:
            is_rle = annotation.get("_is_rle", False)

        self.draw_polygon(image, points, color, label, is_rle=is_rle)

    def _draw_bounding_box(self, image: np.ndarray, annotation: Dict,
                          color: Tuple[int, int, int], label: str):
        """Draw bounding box from annotation.

        Args:
            image: Image to draw on
            annotation: Annotation dictionary with bbox field
            color: BGR color tuple
            label: Class label text
        """
        bbox = annotation["bbox"]
        if len(bbox) != 4:
            self.logger.warning(f"Invalid bbox format: {bbox}")
            return

        # Convert [x_min, y_min, width, height] to (x_min, y_min, x_max, y_max)
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        self.draw_bounding_box(image, (x_min, y_min, x_max, y_max), color, label)

    def visualize(self, *args, **kwargs) -> Dict[str, Any]:
        """Perform visualization. Must be implemented by subclasses.

        Returns:
            Dict with visualization results
        """
        raise NotImplementedError("Subclasses must implement visualize method")

    def _process_image_annotations(self, image_data: Dict, classes: List[str],
                                  save_dir: Optional[str] = None) -> Dict[str, Any]:
        """Process a single image with its annotations (common helper method).

        Args:
            image_data: Unified format image data dictionary
            classes: List of class names
            save_dir: Optional directory to save visualized image

        Returns:
            Dictionary with processing results for this image
        """
        image_path = image_data["image_path"]
        annotations = image_data.get("annotations", [])
        image_id = image_data.get("image_id", os.path.basename(image_path))

        # Read image
        image = self.read_image(image_path)
        if image is None:
            self.logger.warning(f"Could not read image: {image_path}")
            return {"processed": False, "reason": "image_read_failed"}

        # Validate segmentation format if required
        try:
            if self.segmentation:
                self._validate_segmentation_format(annotations)
        except ValueError as e:
            self.logger.error(f"Segmentation validation failed for {image_id}: {e}")
            return {"processed": False, "reason": "segmentation_validation_failed", "error": str(e)}

        # Draw annotations
        image_with_annotations = self._draw_annotations(image, annotations, classes)

        # Display or save
        if save_dir:
            output_filename = os.path.basename(image_path)
            output_path = os.path.join(save_dir, output_filename)
            if self.save_image(image_with_annotations, output_path):
                return {
                    "processed": True,
                    "saved": True,
                    "output_path": output_path,
                    "annotations_count": len(annotations)
                }
            else:
                return {"processed": False, "reason": "save_failed"}
        else:
            # Display image
            self.logger.info(f"Displaying: {os.path.basename(image_path)}")
            self.logger.info("Press any key to continue, 'ESC' to exit")
            key = self.display_image(image_with_annotations, wait_key=0)
            if key == 27:  # ESC key
                self.logger.info("Visualization stopped by user")
                return {"processed": True, "stopped": True, "annotations_count": len(annotations)}
            return {"processed": True, "annotations_count": len(annotations)}

    def _create_results_template(self, **kwargs) -> Dict[str, Any]:
        """Create a template results dictionary with common fields.

        Returns:
            Results template dictionary
        """
        results = {
            "images_processed": 0,
            "images_with_annotations": 0,
            "annotations_processed": 0,
            "saved_images": 0,
            "classes_found": set(),
            "errors": []
        }
        results.update(kwargs)
        return results

    def _update_results_from_image(self, results: Dict[str, Any],
                                   image_result: Dict[str, Any],
                                   annotations: List[Dict]):
        """Update aggregated results from single image processing result.

        Args:
            results: Aggregated results dictionary to update
            image_result: Single image processing result
            annotations: List of annotations for this image
        """
        if not image_result.get("processed"):
            if "error" in image_result:
                results["errors"].append(image_result["error"])
            return

        results["images_processed"] += 1
        if annotations:
            results["images_with_annotations"] += 1
            results["annotations_processed"] += len(annotations)

        # Track classes found
        for ann in annotations:
            # Prefer category_name, fallback to category_id as string
            class_name = ann.get("category_name")
            if class_name:
                results["classes_found"].add(class_name)
            else:
                category_id = ann.get("category_id")
                if category_id is not None:
                    results["classes_found"].add(str(category_id))

        # Track saved images
        if image_result.get("saved"):
            results["saved_images"] += 1

        # Check if visualization was stopped
        if image_result.get("stopped"):
            results["stopped_by_user"] = True