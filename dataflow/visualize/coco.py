# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:40
@File    : coco.py
@Author  : zj
@Description: COCO format visualizer
"""

import os
import json
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .base import BaseVisualizer
from ..config import Config


class CocoVisualizer(BaseVisualizer):
    """Visualizer for COCO format annotations."""

    def visualize(self,
                  image_dir: str,
                  annotation_json: str,
                  save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Visualize COCO annotations on images.

        Args:
            image_dir (str): Directory containing image files
            annotation_json (str): Path to COCO JSON annotation file
            save_dir (Optional[str]): Directory to save visualized images.
                If None, images are only displayed.

        Returns:
            Dict with visualization results:
                - images_processed: Number of images processed
                - images_with_annotations: Number of images with annotations
                - annotations_processed: Total number of annotations drawn
                - categories_found: List of category IDs found in annotations
                - saved_images: Number of images saved (if save_dir provided)
                - save_dir: Path where images were saved (if save_dir provided)
        """
        # Validate inputs
        if not self.validate_input_path(image_dir, is_dir=True):
            raise ValueError(f"Invalid image directory: {image_dir}")
        if not self.validate_input_path(annotation_json, is_dir=False):
            raise ValueError(f"Invalid annotation file: {annotation_json}")
        if save_dir and not self.validate_output_path(save_dir, is_dir=True, create=True):
            raise ValueError(f"Invalid save directory: {save_dir}")

        # Load COCO annotations
        coco_data = self._load_coco_annotations(annotation_json)
        if not coco_data:
            raise ValueError(f"Could not load COCO annotations from {annotation_json}")

        # Extract data
        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])
        categories = coco_data.get("categories", [])

        self.logger.info(f"Loaded {len(images)} images, {len(annotations)} annotations, "
                        f"{len(categories)} categories from {annotation_json}")

        # Create mappings for faster lookup
        image_dict = {img["id"]: img for img in images}
        category_dict = {cat["id"]: cat for cat in categories}

        # Group annotations by image_id
        annotations_by_image = self._group_annotations_by_image(annotations)

        # Process images
        results = {
            "annotation_json": annotation_json,
            "image_dir": image_dir,
            "total_images": len(images),
            "images_processed": 0,
            "images_with_annotations": 0,
            "annotations_processed": 0,
            "categories_found": set(),
            "saved_images": 0,
            "save_dir": save_dir,
        }

        self.logger.info("Starting visualization...")
        if save_dir:
            self.logger.info(f"Images will be saved to: {save_dir}")

        # Process each image in the COCO dataset
        for i, image_info in enumerate(images):
            image_id = image_info["id"]
            image_filename = image_info.get("file_name", "")

            if not image_filename:
                self.logger.warning(f"Image {image_id} has no filename, skipping")
                continue

            # Build full image path
            image_path = os.path.join(image_dir, image_filename)
            if not os.path.exists(image_path):
                # Try to find image by ID or other names
                self.logger.debug(f"Image not found at {image_path}, trying alternatives...")
                found_path = self._find_image_file(image_dir, image_filename, image_id)
                if not found_path:
                    self.logger.warning(f"Could not find image file for {image_filename}, skipping")
                    continue
                image_path = found_path

            # Get annotations for this image
            img_annotations = annotations_by_image.get(image_id, [])
            if not img_annotations:
                self.logger.debug(f"No annotations for image {image_filename}")
                continue

            # Read image
            image = self.read_image(image_path)
            if image is None:
                self.logger.warning(f"Could not read image: {image_path}")
                continue

            # Draw annotations
            image_with_boxes = self._draw_coco_annotations(
                image, img_annotations, category_dict, image_info
            )

            # Display or save
            if save_dir:
                # Save image with original filename
                output_filename = os.path.basename(image_path)
                output_path = os.path.join(save_dir, output_filename)
                if self.save_image(image_with_boxes, output_path):
                    results["saved_images"] += 1
            else:
                # Display image
                self.logger.info(f"Displaying: {os.path.basename(image_path)}")
                self.logger.info("Press any key to continue, 'ESC' to exit")
                key = self.display_image(image_with_boxes, wait_key=0)
                if key == 27:  # ESC key
                    self.logger.info("Visualization stopped by user")
                    break

            # Update results
            results["images_processed"] += 1
            results["images_with_annotations"] += 1
            results["annotations_processed"] += len(img_annotations)
            for ann in img_annotations:
                results["categories_found"].add(ann["category_id"])

            self._print_progress(i + 1, len(images), "Visualization ")

        # Close windows if they were opened
        if not save_dir:
            self.close_windows()

        # Convert set to list for JSON serialization
        results["categories_found"] = list(results["categories_found"])

        self.logger.info("Visualization completed")
        self.logger.info(f"Processed {results['images_processed']} images")
        self.logger.info(f"Found {results['annotations_processed']} annotations")
        self.logger.info(f"Categories found: {results['categories_found']}")
        if save_dir:
            self.logger.info(f"Saved {results['saved_images']} images to {save_dir}")

        return results

    def _load_coco_annotations(self, annotation_path: str) -> Optional[Dict[str, Any]]:
        """Load COCO JSON annotations."""
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {annotation_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading annotation file {annotation_path}: {e}")
            return None

    def _group_annotations_by_image(self, annotations: List[Dict]) -> Dict[int, List[Dict]]:
        """Group annotations by image_id."""
        grouped = {}
        for ann in annotations:
            image_id = ann.get("image_id")
            if image_id is not None:
                if image_id not in grouped:
                    grouped[image_id] = []
                grouped[image_id].append(ann)
        return grouped

    def _find_image_file(self,
                         image_dir: str,
                         original_filename: str,
                         image_id: int) -> Optional[str]:
        """
        Try to find image file when direct path doesn't exist.

        Args:
            image_dir: Directory containing images
            original_filename: Original filename from COCO
            image_id: COCO image ID

        Returns:
            Full path to image file, or None if not found
        """
        # First try original filename
        original_path = os.path.join(image_dir, original_filename)
        if os.path.exists(original_path):
            return original_path

        # Try without path components (in case COCO has subdirectories)
        basename = os.path.basename(original_filename)
        simple_path = os.path.join(image_dir, basename)
        if os.path.exists(simple_path):
            return simple_path

        # Try common extensions
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            # Try with image_id
            id_path = os.path.join(image_dir, f"{image_id}{ext}")
            if os.path.exists(id_path):
                return id_path

            # Try with basename + ext
            name_no_ext = os.path.splitext(basename)[0]
            name_path = os.path.join(image_dir, f"{name_no_ext}{ext}")
            if os.path.exists(name_path):
                return name_path

        # Last resort: search all files for image_id in filename
        import glob
        for ext in ['*jpg', '*jpeg', '*png', '*bmp', '*tif', '*tiff']:
            pattern = os.path.join(image_dir, f"*{image_id}*{ext}")
            matches = glob.glob(pattern)
            if matches:
                return matches[0]

        return None

    def _draw_coco_annotations(self,
                               image: np.ndarray,
                               annotations: List[Dict],
                               category_dict: Dict[int, Dict],
                               image_info: Dict) -> np.ndarray:
        """Draw COCO annotations on image."""
        result_image = image.copy()
        img_height, img_width = image.shape[:2]

        for ann in annotations:
            category_id = ann.get("category_id")
            if category_id not in category_dict:
                self.logger.warning(f"Unknown category ID {category_id} in annotation {ann.get('id')}")
                continue

            category = category_dict[category_id]
            category_name = category.get("name", f"Category_{category_id}")

            # Get color for this category
            color = self.get_color_for_class(category_id, len(category_dict))

            # Draw bounding box if available
            bbox = ann.get("bbox")
            if bbox and len(bbox) == 4:
                # COCO bbox format: [x, y, width, height]
                x, y, w, h = bbox
                x_min = x
                y_min = y
                x_max = x + w
                y_max = y + h

                # Clamp to image boundaries
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img_width - 1, x_max)
                y_max = min(img_height - 1, y_max)

                # Prepare label text
                label = category_name
                if ann.get("score"):
                    label += f" {ann['score']:.2f}"

                # Draw bounding box
                self.draw_bounding_box(result_image, (x_min, y_min, x_max, y_max),
                                      color, label)

            # Draw segmentation if available
            segmentation = ann.get("segmentation")
            if segmentation:
                self._draw_segmentation(result_image, segmentation, color, category_name)

            # Draw keypoints if available
            keypoints = ann.get("keypoints")
            if keypoints:
                self._draw_keypoints(result_image, keypoints, color)

        return result_image

    def _draw_segmentation(self,
                          image: np.ndarray,
                          segmentation,
                          color: Tuple[int, int, int],
                          label: str):
        """Draw segmentation polygon(s)."""
        if isinstance(segmentation, list):
            # Multiple polygons
            for poly in segmentation:
                if isinstance(poly, list) and len(poly) >= 6:  # Need at least 3 points (x,y,x,y,x,y)
                    points = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                    self.draw_polygon(image, points, color, label)
        elif isinstance(segmentation, dict):
            # RLE format - would need to decode
            self.logger.debug("RLE segmentation format not yet supported for visualization")
        else:
            self.logger.warning(f"Unsupported segmentation format: {type(segmentation)}")

    def _draw_keypoints(self,
                       image: np.ndarray,
                       keypoints: List[float],
                       color: Tuple[int, int, int]):
        """Draw keypoints (x, y, visibility)."""
        if len(keypoints) % 3 != 0:
            self.logger.warning(f"Invalid keypoints length: {len(keypoints)} (must be multiple of 3)")
            return

        for i in range(0, len(keypoints), 3):
            x = keypoints[i]
            y = keypoints[i+1]
            v = keypoints[i+2]

            if v > 0:  # Visible keypoint
                point = (int(x), int(y))
                cv2.circle(image, point, 3, color, -1)

    def batch_visualize(self,
                        image_dirs: List[str],
                        annotation_jsons: List[str],
                        save_dirs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Visualize multiple COCO datasets.

        Args:
            image_dirs (List[str]): List of image directories
            annotation_jsons (List[str]): List of annotation JSON files
            save_dirs (Optional[List[str]]): List of save directories

        Returns:
            List of visualization results
        """
        if len(image_dirs) != len(annotation_jsons):
            raise ValueError("image_dirs and annotation_jsons must have the same length")

        if save_dirs is not None and len(save_dirs) != len(image_dirs):
            raise ValueError("save_dirs must have same length as image_dirs")

        results = []
        for i, (image_dir, annotation_json) in enumerate(zip(image_dirs, annotation_jsons)):
            save_dir = save_dirs[i] if save_dirs else None
            self.logger.info(f"Processing dataset {i+1}/{len(image_dirs)}")

            try:
                result = self.visualize(image_dir, annotation_json, save_dir)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to visualize dataset {i+1}: {e}")
                if Config.OVERWRITE_EXISTING:
                    raise
                else:
                    results.append(None)

        return results