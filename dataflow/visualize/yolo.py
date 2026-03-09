# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/9 21:15
@File    : yolo.py
@Author  : zj
@Description: YOLO format visualizer using label handler
"""

import os
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

        # Read annotations using YoloHandler
        try:
            annotations_list = self.label_handler.read_batch(
                labels_dir=label_dir,
                images_dir=image_dir,
                classes_path=class_path,
                require_segmentation=self.segmentation
            )
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