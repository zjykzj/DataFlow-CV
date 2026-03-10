# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/9 21:25
@File    : labelme.py
@Author  : zj
@Description: LabelMe format visualizer using label handler
"""

import os
import sys
from typing import List, Dict, Any, Optional

from .generic import GenericVisualizer
from ..label.labelme import LabelMeHandler
from ..config import Config


class LabelMeVisualizer(GenericVisualizer):
    """Visualizer for LabelMe format annotations using LabelMeHandler."""

    def __init__(self, verbose: bool = None, segmentation: bool = False):
        """Initialize LabelMe visualizer.

        Args:
            verbose: Whether to print progress information.
                If None, uses Config.VERBOSE.
            segmentation: Whether to force segmentation mode (strict validation).
                If True, annotations must have shape_type="polygon".
        """
        super().__init__(verbose, segmentation)
        self.label_handler = LabelMeHandler(verbose=verbose)

    def visualize(self,
                  image_dir: str,
                  label_dir: str,
                  save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Visualize LabelMe annotations on images.

        Args:
            image_dir (str): Directory containing image files
            label_dir (str): Directory containing LabelMe JSON files
            save_dir (Optional[str]): Directory to save visualized images.
                If None, images are only displayed.

        Returns:
            Dict with visualization results:
                - images_processed: Number of images processed
                - images_with_annotations: Number of images with annotations
                - annotations_processed: Total number of annotations drawn
                - classes_found: List of class names found in annotations
                - saved_images: Number of images saved (if save_dir provided)
                - save_dir: Path where images were saved (if save_dir provided)
        """
        # Validate inputs
        if not self.validate_input_path(image_dir, is_dir=True):
            raise ValueError(f"Invalid image directory: {image_dir}")
        if not self.validate_input_path(label_dir, is_dir=True):
            raise ValueError(f"Invalid label directory: {label_dir}")
        if save_dir and not self.validate_output_path(save_dir, is_dir=True, create=True):
            raise ValueError(f"Invalid save directory: {save_dir}")

        # Read annotations using LabelMeHandler
        try:
            annotations_list = self.label_handler.read_batch(
                json_dir=label_dir,
                require_segmentation=self.segmentation
            )
        except Exception as e:
            raise ValueError(f"Failed to read LabelMe annotations: {e}")

        # Resolve image paths: always use the provided image_dir
        annotations_list = self._resolve_image_paths(annotations_list, image_dir)

        if not annotations_list:
            self.logger.warning(f"No annotations found in {label_dir}")
            # If segmentation mode is enabled and there are JSON files, raise error
            if self.segmentation:
                import glob
                json_files = glob.glob(os.path.join(label_dir, "*.json"))
                if json_files:
                    raise ValueError(
                        f"Segmentation mode required but no valid polygon annotations found in {label_dir}. "
                        f"Found {len(json_files)} JSON file(s) with only rectangle shapes or no shapes."
                    )
            # Return empty results
            return self._create_results_template(
                image_dir=image_dir,
                label_dir=label_dir,
                save_dir=save_dir,
                total_images=len(self.get_image_files(image_dir))
            )

        # Check if any annotations exist when segmentation mode is enabled
        total_annotations = sum(len(img.get("annotations", [])) for img in annotations_list)
        if self.segmentation and total_annotations == 0:
            import glob
            json_files = glob.glob(os.path.join(label_dir, "*.json"))
            if json_files:
                raise ValueError(
                    f"Segmentation mode required but no valid polygon annotations found in {label_dir}. "
                    f"Found {len(json_files)} JSON file(s) with only rectangle shapes or no shapes."
                )

        # Create results template
        results = self._create_results_template(
            image_dir=image_dir,
            label_dir=label_dir,
            save_dir=save_dir,
            total_images=len(annotations_list)
        )

        # Extract unique class names for color assignment
        classes = self._extract_classes(annotations_list)

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

    def _extract_classes(self, annotations_list: List[Dict]) -> List[str]:
        """Extract unique class names from annotations list.

        Args:
            annotations_list: List of image annotation data

        Returns:
            Sorted list of unique class names
        """
        class_names = set()
        for image_data in annotations_list:
            for annotation in image_data.get("annotations", []):
                class_name = annotation.get("category_name")
                if class_name:
                    class_names.add(class_name)
        return sorted(class_names)

    def _resolve_image_paths(self, annotations_list: List[Dict], image_dir: str) -> List[Dict]:
        """Resolve image paths in annotations list.

        Always use the provided image_dir to locate images, ignoring the
        image_path stored in the annotation (which may point to the label
        directory).

        Args:
            annotations_list: List of image annotation data
            image_dir: Directory containing image files

        Returns:
            Updated annotations list with resolved image paths
        """
        import os

        for image_data in annotations_list:
            original_path = image_data.get("image_path", "")
            if not original_path:
                continue

            # Always use image_dir to locate the image
            filename = os.path.basename(original_path)
            candidate_path = os.path.join(image_dir, filename)

            # Update image_path to the resolved path
            image_data["image_path"] = candidate_path
            if self.verbose:
                self.logger.info(f"Resolved image path: {original_path} -> {candidate_path}")
            else:
                self.logger.debug(f"Resolved image path: {original_path} -> {candidate_path}")

        return annotations_list

    def batch_visualize(self,
                        image_dirs: List[str],
                        label_dirs: List[str],
                        save_dirs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Visualize multiple LabelMe datasets.

        Args:
            image_dirs (List[str]): List of image directories
            label_dirs (List[str]): List of label directories
            save_dirs (Optional[List[str]]): List of save directories

        Returns:
            List of visualization results
        """
        if len(image_dirs) != len(label_dirs):
            raise ValueError("image_dirs and label_dirs must have the same length")

        if save_dirs is not None and len(save_dirs) != len(image_dirs):
            raise ValueError("save_dirs must have same length as image_dirs")

        results = []
        for i, (image_dir, label_dir) in enumerate(zip(image_dirs, label_dirs)):
            save_dir = save_dirs[i] if save_dirs else None
            self.logger.info(f"Processing dataset {i+1}/{len(image_dirs)}")

            try:
                result = self.visualize(image_dir, label_dir, save_dir)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to visualize dataset {i+1}: {e}")
                if Config.OVERWRITE_EXISTING:
                    raise
                else:
                    results.append(None)

        return results