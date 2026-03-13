# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/9 21:15
@File    : yolo.py
@Author  : zj
@Description: YOLO format visualizer using label handler
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

from .generic import GenericVisualizer
from ..label.yolo import YoloHandler
from ..config import Config


class YoloVisualizer(GenericVisualizer):
    """Visualizer for YOLO format annotations using YoloHandler."""

    def __init__(self, verbose: bool = None, segmentation: bool = False):
        """Initialize YOLO visualizer.

        Args:
            verbose: Whether to print progress information.
                If None, uses Config.VERBOSE.
            segmentation: Whether to force segmentation mode (strict validation).
                If True, annotations must contain valid segmentation data.
        """
        super().__init__(verbose, segmentation)
        self.label_handler = YoloHandler(verbose=verbose)

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
        # Temporarily enable debug logging for troubleshooting
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("Debug logging enabled for YOLO visualization")

        if not self.validate_input_path(image_dir, is_dir=True):
            raise ValueError(f"Invalid image directory: {image_dir}")
        if not self.validate_input_path(label_dir, is_dir=True):
            raise ValueError(f"Invalid label directory: {label_dir}")
        if not self.validate_input_path(class_path, is_dir=False):
            raise ValueError(f"Invalid class file: {class_path}")
        if save_dir and not self.validate_output_path(save_dir, is_dir=True, create=True):
            raise ValueError(f"Invalid save directory: {save_dir}")


        # Read classes
        classes = self._read_classes_file(class_path)
        self.logger.info(f"Loaded {len(classes)} classes from {class_path}")
        self.logger.debug(f"Classes list: {classes}")

        # Read annotations using YoloHandler
        try:
            annotations_list = self.label_handler.read_batch(
                labels_dir=label_dir,
                images_dir=image_dir,
                classes_path=class_path,
                require_segmentation=self.segmentation
            )
            # Debug logging for annotation data
            if annotations_list:
                print(f"[DEBUG] Read {len(annotations_list)} image annotations", flush=True)
                for i, img_data in enumerate(annotations_list[:3]):  # Check first 3 images
                    anns = img_data.get("annotations", [])
                    print(f"[DEBUG]   Image {i}: {img_data.get('image_id', 'unknown')}, {len(anns)} annotations", flush=True)
                    for j, ann in enumerate(anns[:3]):  # Check first 3 annotations per image
                        print(f"[DEBUG]     Annotation {j}: category_id={ann.get('category_id')}, category_name={ann.get('category_name')}", flush=True)
        except Exception as e:
            raise ValueError(f"Failed to read YOLO annotations: {e}")

        if not annotations_list:
            self.logger.warning(f"No annotations found in {label_dir}")
            # If segmentation mode is enabled and there are label files, raise error
            if self.segmentation:
                import glob
                label_files = glob.glob(os.path.join(label_dir, "*.txt"))
                if label_files:
                    raise ValueError(
                        f"Segmentation mode required but no valid segmentation annotations found in {label_dir}. "
                        f"Found {len(label_files)} label file(s) with only detection format or no annotations."
                    )
            # Return empty results
            return self._create_results_template(
                image_dir=image_dir,
                label_dir=label_dir,
                class_path=class_path,
                save_dir=save_dir,
                total_images=len(self.get_image_files(image_dir))
            )

        # Check if any annotations exist when segmentation mode is enabled
        total_annotations = sum(len(img.get("annotations", [])) for img in annotations_list)
        if self.segmentation and total_annotations == 0:
            import glob
            label_files = glob.glob(os.path.join(label_dir, "*.txt"))
            if label_files:
                raise ValueError(
                    f"Segmentation mode required but no valid segmentation annotations found in {label_dir}. "
                    f"Found {len(label_files)} label file(s) with only detection format or no annotations."
                )

        # Merge classes from file with classes found in annotations
        classes = self._extract_classes(annotations_list, classes)
        self.logger.info(f"Using {len(classes)} classes for visualization")

        # Create results template
        results = self._create_results_template(
            image_dir=image_dir,
            label_dir=label_dir,
            class_path=class_path,
            save_dir=save_dir,
            total_images=len(annotations_list)
        )

        self.logger.info("Starting visualization...")
        if save_dir:
            self.logger.info(f"Images will be saved to: {save_dir}")

        # Process each image
        for i, image_data in enumerate(annotations_list):
            image_result = self._process_image_annotations(
                image_data=image_data,
                classes=classes,
                save_dir=save_dir
            )

            # Update aggregated results
            self._update_results_from_image(
                results=results,
                image_result=image_result,
                annotations=image_data.get("annotations", [])
            )

            # Check if visualization was stopped by user
            if image_result.get("stopped"):
                results["stopped_by_user"] = True
                break

            self._print_progress(i + 1, len(annotations_list), "Visualization ")

        # Close windows if they were opened and visualization wasn't saved
        if not save_dir and not results.get("stopped_by_user"):
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
        """Read class names from file.

        Args:
            class_path: Path to class names file

        Returns:
            List of class names
        """
        try:
            with open(class_path, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f if line.strip()]
            return classes
        except Exception as e:
            self.logger.error(f"Error reading class file {class_path}: {e}")
            raise ValueError(f"Could not read class file: {class_path}") from e

    def _extract_classes(self, annotations_list: List[Dict], file_classes: List[str]) -> List[str]:
        """Extract and merge class names from annotations and file.

        Args:
            annotations_list: List of image annotation data
            file_classes: List of class names from file

        Returns:
            Merged list of class names, ensuring all annotations have matching names
        """
        # Normalize file classes: strip whitespace, create mapping from normalized to original
        file_class_map = {}  # normalized -> original
        normalized_file_classes = []
        for class_name in file_classes:
            normalized = class_name.strip()
            file_class_map[normalized] = class_name
            normalized_file_classes.append(normalized)

        # Extract unique class names from annotations with normalization
        annotation_classes = set()  # Store normalized names
        original_annotation_names = {}  # normalized -> first original encountered
        for image_data in annotations_list:
            for annotation in image_data.get("annotations", []):
                class_name = annotation.get("category_name")
                if class_name:
                    normalized = class_name.strip()
                    annotation_classes.add(normalized)
                    if normalized not in original_annotation_names:
                        original_annotation_names[normalized] = class_name  # Keep original for reference
                else:
                    # Fallback to category_id
                    category_id = annotation.get("category_id", 0)
                    class_name = f"class_{category_id}"
                    normalized = class_name.strip()
                    annotation_classes.add(normalized)
                    if normalized not in original_annotation_names:
                        original_annotation_names[normalized] = class_name

        # Merge with file classes, preserving file order for existing classes
        merged_classes = []
        matched_normalized = set()

        # First add file classes that appear in annotations (case-insensitive and whitespace-insensitive)
        for normalized, original in file_class_map.items():
            if normalized in annotation_classes:
                merged_classes.append(original)  # Use original file class name
                matched_normalized.add(normalized)
                annotation_classes.remove(normalized)
            else:
                # Also check case-insensitive match
                matched = False
                for ann_normalized in list(annotation_classes):
                    if ann_normalized.lower() == normalized.lower():
                        merged_classes.append(original)  # Use original file class name
                        matched_normalized.add(ann_normalized)
                        annotation_classes.remove(ann_normalized)
                        matched = True
                        self.logger.warning(f"Class name '{ann_normalized}' matched case-insensitively to file class '{original}'")
                        break
                if not matched:
                    # File class not found in annotations, still include it
                    merged_classes.append(original)

        # Add remaining annotation classes (not matched to file classes)
        remaining = sorted(annotation_classes)
        for normalized in remaining:
            # Use original annotation name if available, otherwise normalized
            original = original_annotation_names.get(normalized, normalized)
            merged_classes.append(original)

        self.logger.info(f"Merged classes: {len(file_classes)} from file, {len(merged_classes)} total after merge")
        if len(merged_classes) > len(file_classes):
            self.logger.warning(f"Found {len(merged_classes) - len(file_classes)} classes in annotations not in file")

        # Debug logging for color assignment consistency
        self.logger.debug(f"Merged classes list: {merged_classes}")
        self.logger.debug(f"File classes: {file_classes}")
        self.logger.debug(f"Annotation classes (normalized): {original_annotation_names}")

        return merged_classes

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