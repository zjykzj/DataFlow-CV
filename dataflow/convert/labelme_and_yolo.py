"""
LabelMe and YOLO format converter.

Handles bidirectional conversion between LabelMe and YOLO annotation formats.
Supports both object detection and instance segmentation annotations.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .base import BaseConverter, ConversionResult
from ..label.labelme_handler import LabelMeAnnotationHandler
from ..label.yolo_handler import YoloAnnotationHandler
from ..label.base import AnnotationResult
from ..label.models import DatasetAnnotations
from . import utils


class LabelMeAndYoloConverter(BaseConverter):
    """Converter for bidirectional conversion between LabelMe and YOLO formats."""

    def __init__(self, source_to_target: bool, **kwargs):
        """
        Initialize converter.

        Args:
            source_to_target: True for LabelMe→YOLO, False for YOLO→LabelMe
            **kwargs: Arguments passed to BaseConverter
        """
        if source_to_target:
            source_format = "labelme"
            target_format = "yolo"
        else:
            source_format = "yolo"
            target_format = "labelme"

        super().__init__(source_format, target_format, **kwargs)
        self.source_to_target = source_to_target

    def convert(self, source_path: str, target_path: str, **kwargs) -> ConversionResult:
        """
        Convert annotations between LabelMe and YOLO formats.

        Args:
            source_path: Path to source annotations
            target_path: Path for target annotations
            **kwargs: Additional conversion parameters:
                - class_file: Required, path to class file
                - image_dir: Optional for LabelMe→YOLO, required for YOLO→LabelMe

        Returns:
            ConversionResult with conversion status and details
        """
        # Store source_annotations for use in create_target_handler if needed
        self._source_annotations_for_target = None

        # 1. Validate inputs
        if not self.validate_inputs(source_path, target_path, kwargs):
            return self._create_conversion_result(
                success=False,
                source_path=source_path,
                target_path=target_path,
                errors=["Input validation failed"]
            )

        # 2. Read data using source handler
        source_handler = self.create_source_handler(source_path, kwargs)
        read_result = source_handler.read()
        if not read_result.success:
            return self._create_conversion_result(
                success=False,
                source_path=source_path,
                target_path=target_path,
                errors=read_result.errors
            )

        # 3. Convert data (format-specific conversions like category mapping)
        annotations = read_result.data
        converted_annotations = self.convert_annotations(annotations, kwargs)

        # Store for potential use in create_target_handler
        self._source_annotations_for_target = converted_annotations

        # 4. Write data using target handler
        target_handler = self.create_target_handler(target_path, kwargs)

        # For YOLO target, write to labels directory instead of root directory
        if self.source_to_target:  # LabelMe → YOLO
            # YoloAnnotationHandler writes to output_dir, which should be the labels directory
            # Get labels_dir from handler if available, or construct from target_path
            if hasattr(target_handler, 'label_dir'):
                write_output_path = target_handler.label_dir
            else:
                write_output_path = str(Path(target_path) / "labels")
        else:  # YOLO → LabelMe
            write_output_path = target_path

        write_result = target_handler.write(converted_annotations, write_output_path)

        # Clear stored annotations
        self._source_annotations_for_target = None

        # 5. Return result
        return self._create_conversion_result(
            success=write_result.success,
            source_path=source_path,
            target_path=target_path,
            annotations=converted_annotations,
            write_result=write_result
        )

    def validate_inputs(self, source_path: str, target_path: str, kwargs: Dict) -> bool:
        """
        Validate conversion input parameters.

        Args:
            source_path: Path to source annotations
            target_path: Path for target annotations
            kwargs: Additional conversion parameters

        Returns:
            True if inputs are valid, False otherwise
        """
        # First, call parent validation for basic checks
        if not super().validate_inputs(source_path, target_path, kwargs):
            return False

        # Check required parameters
        class_file = kwargs.get("class_file")
        if not class_file:
            self.logger.error("class_file parameter is required")
            return False

        class_file_path = Path(class_file)
        if not class_file_path.exists():
            self.logger.error(f"Class file does not exist: {class_file}")
            return False

        # Direction-specific validation
        if self.source_to_target:  # LabelMe → YOLO
            # image_dir is optional, default to source_path parent
            pass
        else:  # YOLO → LabelMe
            image_dir = kwargs.get("image_dir")
            if not image_dir:
                self.logger.error("image_dir parameter is required for YOLO→LabelMe conversion")
                return False

            image_dir_path = Path(image_dir)
            if not image_dir_path.exists():
                self.logger.error(f"Image directory does not exist: {image_dir}")
                return False

        return True

    def create_source_handler(self, source_path: str, kwargs: Dict) -> Any:
        """
        Create source annotation handler.

        Args:
            source_path: Path to source annotations
            kwargs: Additional conversion parameters

        Returns:
            BaseAnnotationHandler subclass instance
        """
        class_file = kwargs.get("class_file")

        if self.source_to_target:  # LabelMe → YOLO
            # LabelMe handler only needs label_dir and class_file
            handler = LabelMeAnnotationHandler(
                label_dir=source_path,
                class_file=class_file,
                logger=self.logger
            )
        else:  # YOLO → LabelMe
            image_dir = kwargs.get("image_dir")
            if not image_dir:
                raise ValueError("image_dir is required for YOLO→LabelMe conversion")

            handler = YoloAnnotationHandler(
                label_dir=source_path,
                class_file=class_file,
                image_dir=image_dir,
                logger=self.logger
            )

        return handler

    def create_target_handler(self, target_path: str, kwargs: Dict) -> Any:
        """
        Create target annotation handler.

        Args:
            target_path: Path for target annotations
            kwargs: Additional conversion parameters

        Returns:
            BaseAnnotationHandler subclass instance
        """
        class_file = kwargs.get("class_file")

        if self.source_to_target:  # LabelMe → YOLO
            # For YOLO target, we need to create appropriate directory structure
            # YOLO expects images/ and labels/ subdirectories
            target_path_obj = Path(target_path)

            # Create base directory
            target_path_obj.mkdir(parents=True, exist_ok=True)

            # Create labels directory for YOLO labels
            labels_dir = target_path_obj / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)

            # Create images directory for YOLO images
            images_dir = target_path_obj / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            # Copy class file to target directory if it doesn't exist there
            import shutil
            source_class_file = Path(class_file)
            target_class_file = target_path_obj / "classes.txt"
            if source_class_file.exists() and not target_class_file.exists():
                try:
                    shutil.copy2(source_class_file, target_class_file)
                    self.logger.info(f"Copied class file to: {target_class_file}")
                    # Update class_file to point to the copied file for handler
                    class_file = str(target_class_file)
                except Exception as e:
                    self.logger.warning(f"Failed to copy class file: {e}")

            # Copy image files if source annotations are available
            if hasattr(self, '_source_annotations_for_target') and self._source_annotations_for_target:
                for image_ann in self._source_annotations_for_target.images:
                    source_image_path = Path(image_ann.image_path)
                    if source_image_path.exists():
                        target_image_path = images_dir / source_image_path.name
                        if not target_image_path.exists():
                            try:
                                shutil.copy2(source_image_path, target_image_path)
                                self.logger.info(f"Copied image to: {target_image_path}")
                            except Exception as e:
                                self.logger.warning(f"Failed to copy image {source_image_path}: {e}")
                    else:
                        self.logger.warning(f"Source image does not exist: {source_image_path}")

            # Get image_dir from kwargs or use images_dir as default
            image_dir = kwargs.get("image_dir")
            if not image_dir:
                # If no image_dir provided, use the images directory we created
                image_dir = str(images_dir)

            handler = YoloAnnotationHandler(
                label_dir=str(labels_dir),
                class_file=class_file,
                image_dir=image_dir,
                logger=self.logger
            )
        else:  # YOLO → LabelMe
            # For LabelMe target, just create the output directory
            target_path_obj = Path(target_path)
            target_path_obj.mkdir(parents=True, exist_ok=True)

            handler = LabelMeAnnotationHandler(
                label_dir=target_path,
                class_file=class_file,
                logger=self.logger
            )

        return handler

    def convert_annotations(self,
                           source_annotations: DatasetAnnotations,
                           kwargs: Dict) -> DatasetAnnotations:
        """
        Convert annotation data between LabelMe and YOLO formats.

        Args:
            source_annotations: Annotations read from source format
            kwargs: Additional conversion parameters

        Returns:
            Converted DatasetAnnotations ready for writing to target format
        """
        # For LabelMe ↔ YOLO conversion, the main differences are:
        # 1. Category management (already handled by handlers)
        # 2. Path resolution (handled by resolve_image_paths)
        # 3. Coordinate system conversion (handled by handlers)

        # Default implementation: return as-is
        # Subclasses can override for format-specific conversions
        return source_annotations