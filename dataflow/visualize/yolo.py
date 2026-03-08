# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:40
@File    : yolo.py
@Author  : zj
@Description: YOLO format visualizer
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .base import BaseVisualizer
from ..config import Config


class YoloVisualizer(BaseVisualizer):
    """Visualizer for YOLO format annotations."""

    def visualize(self,
                  image_dir: str,
                  label_dir: str,
                  class_path: str,
                  save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Visualize YOLO annotations on images.

        Args:
            image_dir (str): Directory containing image files
            label_dir (str): Directory containing YOLO label files (.txt)
            class_path (str): Path to class names file (e.g., class.names)
            save_dir (Optional[str]): Directory to save visualized images.
                If None, images are only displayed.

        Returns:
            Dict with visualization results:
                - images_processed: Number of images processed
                - images_with_annotations: Number of images with annotations
                - annotations_processed: Total number of annotations drawn
                - classes_found: List of class IDs found in annotations
                - saved_images: Number of images saved (if save_dir provided)
                - save_dir: Path where images were saved (if save_dir provided)
        """
        # Validate inputs
        if not self.validate_input_path(image_dir, is_dir=True):
            raise ValueError(f"Invalid image directory: {image_dir}")
        if not self.validate_input_path(label_dir, is_dir=True):
            raise ValueError(f"Invalid label directory: {label_dir}")
        if not self.validate_input_path(class_path, is_dir=False):
            raise ValueError(f"Invalid class file: {class_path}")
        if save_dir and not self.validate_output_path(save_dir, is_dir=True, create=True):
            raise ValueError(f"Invalid save directory: {save_dir}")

        # Read class names
        classes = self._read_classes_file(class_path)
        self.logger.info(f"Loaded {len(classes)} classes from {class_path}")

        # Get image and label files
        image_files = self.get_image_files(image_dir)
        label_files = self._get_label_files(label_dir)

        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")
        self.logger.info(f"Found {len(image_files)} image files")

        # Process images
        results = {
            "image_dir": image_dir,
            "label_dir": label_dir,
            "class_path": class_path,
            "total_images": len(image_files),
            "images_processed": 0,
            "images_with_annotations": 0,
            "annotations_processed": 0,
            "classes_found": set(),
            "saved_images": 0,
            "save_dir": save_dir,
        }

        self.logger.info("Starting visualization...")
        if save_dir:
            self.logger.info(f"Images will be saved to: {save_dir}")

        for i, image_file in enumerate(image_files):
            # Get corresponding label file
            label_file = self._get_matching_label_file(image_file, label_files)
            if not label_file:
                self.logger.debug(f"No label file found for {os.path.basename(image_file)}")
                continue

            # Read image
            image = self.read_image(image_file)
            if image is None:
                self.logger.warning(f"Could not read image: {image_file}")
                continue

            # Read and parse annotations
            annotations = self._parse_yolo_annotations(label_file, classes, image.shape)
            if not annotations:
                self.logger.debug(f"No valid annotations in {label_file}")
                continue

            # Draw annotations
            image_with_boxes = self._draw_yolo_annotations(image, annotations, classes)

            # Display or save
            if save_dir:
                # Save image
                output_filename = os.path.basename(image_file)
                output_path = os.path.join(save_dir, output_filename)
                if self.save_image(image_with_boxes, output_path):
                    results["saved_images"] += 1
            else:
                # Display image
                self.logger.info(f"Displaying: {os.path.basename(image_file)}")
                self.logger.info("Press any key to continue, 'ESC' to exit")
                key = self.display_image(image_with_boxes, wait_key=0)
                if key == 27:  # ESC key
                    self.logger.info("Visualization stopped by user")
                    break

            # Update results
            results["images_processed"] += 1
            results["images_with_annotations"] += 1
            results["annotations_processed"] += len(annotations)
            for ann in annotations:
                results["classes_found"].add(ann["class_id"])

            self._print_progress(i + 1, len(image_files), "Visualization ")

        # Close windows if they were opened
        if not save_dir:
            self.close_windows()

        # Convert set to list for JSON serialization
        results["classes_found"] = list(results["classes_found"])

        self.logger.info("Visualization completed")
        self.logger.info(f"Processed {results['images_processed']} images")
        self.logger.info(f"Found {results['annotations_processed']} annotations")
        self.logger.info(f"Classes found: {results['classes_found']}")
        if save_dir:
            self.logger.info(f"Saved {results['saved_images']} images to {save_dir}")

        return results

    def _read_classes_file(self, class_path: str) -> List[str]:
        """Read class names from file."""
        try:
            with open(class_path, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f if line.strip()]
            return classes
        except Exception as e:
            self.logger.error(f"Error reading class file {class_path}: {e}")
            raise ValueError(f"Could not read class file: {class_path}") from e

    def _get_label_files(self, label_dir: str) -> List[str]:
        """Get list of YOLO label files in directory."""
        import glob
        ext = ".txt"  # YOLO label extension
        pattern = os.path.join(label_dir, f"*{ext}")
        return sorted(glob.glob(pattern))

    def _get_matching_label_file(self,
                                 image_file: str,
                                 label_files: List[str]) -> Optional[str]:
        """Find label file that matches the image file name."""
        image_basename = os.path.splitext(os.path.basename(image_file))[0]
        for label_file in label_files:
            label_basename = os.path.splitext(os.path.basename(label_file))[0]
            if label_basename == image_basename:
                return label_file
        return None

    def _parse_yolo_annotations(self,
                                label_file: str,
                                classes: List[str],
                                image_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """
        Parse YOLO format annotations.

        YOLO format: class_id x_center y_center width height
        Coordinates are normalized (0-1).

        Args:
            label_file (str): Path to YOLO label file
            classes (List[str]): List of class names
            image_shape (Tuple[int, int, int]): Image dimensions (height, width, channels)

        Returns:
            List of annotation dictionaries with keys:
                - class_id (int)
                - class_name (str)
                - bbox (Tuple[float, float, float, float]): (x_min, y_min, x_max, y_max) in pixels
                - confidence (Optional[float]): If provided in YOLO format
        """
        annotations = []
        img_height, img_width = image_shape[:2]

        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        self.logger.warning(
                            f"Invalid line {line_num} in {label_file}: expected at least 5 values, got {len(parts)}"
                        )
                        continue

                    try:
                        class_id = int(parts[0])
                        if class_id < 0 or class_id >= len(classes):
                            self.logger.warning(
                                f"Invalid class ID {class_id} in {label_file}: "
                                f"must be between 0 and {len(classes)-1}"
                            )
                            continue

                        # Parse coordinates (normalized)
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Validate normalized coordinates
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                                0 <= width <= 1 and 0 <= height <= 1):
                            self.logger.warning(
                                f"Invalid normalized coordinates in {label_file} line {line_num}: "
                                f"x_center={x_center}, y_center={y_center}, width={width}, height={height}"
                            )
                            continue

                        # Convert to pixel coordinates
                        x_center_px = x_center * img_width
                        y_center_px = y_center * img_height
                        width_px = width * img_width
                        height_px = height * img_height

                        # Calculate bounding box corners
                        x_min = x_center_px - (width_px / 2)
                        y_min = y_center_px - (height_px / 2)
                        x_max = x_center_px + (width_px / 2)
                        y_max = y_center_px + (height_px / 2)

                        # Clamp to image boundaries
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(img_width - 1, x_max)
                        y_max = min(img_height - 1, y_max)

                        # Check for confidence score (optional, after bbox)
                        confidence = None
                        if len(parts) >= 6:
                            confidence = float(parts[5])

                        annotations.append({
                            "class_id": class_id,
                            "class_name": classes[class_id] if class_id < len(classes) else f"Class_{class_id}",
                            "bbox": (x_min, y_min, x_max, y_max),
                            "confidence": confidence,
                        })

                    except (ValueError, IndexError) as e:
                        self.logger.warning(
                            f"Error parsing line {line_num} in {label_file}: {e}"
                        )
                        continue

        except Exception as e:
            self.logger.error(f"Error reading label file {label_file}: {e}")
            return []

        return annotations

    def _draw_yolo_annotations(self,
                               image: np.ndarray,
                               annotations: List[Dict[str, Any]],
                               classes: List[str]) -> np.ndarray:
        """Draw YOLO annotations on image."""
        result_image = image.copy()

        for ann in annotations:
            class_id = ann["class_id"]
            class_name = ann["class_name"]
            bbox = ann["bbox"]
            confidence = ann.get("confidence")

            # Get color for this class
            color = self.get_color_for_class(class_id, len(classes))

            # Prepare label text
            label = class_name
            if confidence is not None:
                label += f" {confidence:.2f}"

            # Draw bounding box
            self.draw_bounding_box(result_image, bbox, color, label, confidence)

        return result_image

    def batch_visualize(self,
                        image_dirs: List[str],
                        label_dirs: List[str],
                        class_paths: List[str],
                        save_dirs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Visualize multiple YOLO datasets.

        Args:
            image_dirs (List[str]): List of image directories
            label_dirs (List[str]): List of label directories
            class_paths (List[str]): List of class files
            save_dirs (Optional[List[str]]): List of save directories

        Returns:
            List of visualization results
        """
        if len(image_dirs) != len(label_dirs) != len(class_paths):
            raise ValueError("All input lists must have the same length")

        if save_dirs is not None and len(save_dirs) != len(image_dirs):
            raise ValueError("save_dirs must have same length as image_dirs")

        results = []
        for i, (image_dir, label_dir, class_path) in enumerate(zip(image_dirs, label_dirs, class_paths)):
            save_dir = save_dirs[i] if save_dirs else None
            self.logger.info(f"Processing dataset {i+1}/{len(image_dirs)}")

            try:
                result = self.visualize(image_dir, label_dir, class_path, save_dir)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to visualize dataset {i+1}: {e}")
                if Config.OVERWRITE_EXISTING:
                    raise
                else:
                    results.append(None)

        return results