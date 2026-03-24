"""
COCO and LabelMe format converter.

Handles bidirectional conversion between COCO and LabelMe annotation formats.
Supports both object detection and instance segmentation annotations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..label.base import AnnotationResult
from ..label.coco_handler import CocoAnnotationHandler
from ..label.labelme_handler import LabelMeAnnotationHandler
from ..label.models import DatasetAnnotations
from . import utils
from .base import BaseConverter, ConversionResult
from .rle_converter import RLEConverter


class CocoAndLabelMeConverter(BaseConverter):
    """Converter for bidirectional conversion between COCO and LabelMe formats."""

    def __init__(self, source_to_target: bool, verbose: bool = False, **kwargs):
        """
        Initialize converter.

        Args:
            source_to_target: True for COCO→LabelMe, False for LabelMe→COCO
            verbose: Whether to enable verbose logging (new)
            **kwargs: Arguments passed to BaseConverter
        """
        if source_to_target:
            source_format = "coco"
            target_format = "labelme"
        else:
            source_format = "labelme"
            target_format = "coco"

        super().__init__(source_format, target_format, verbose=verbose, **kwargs)
        self.source_to_target = source_to_target

        if verbose:
            direction = "COCO→LabelMe" if source_to_target else "LabelMe→COCO"
            self.logger.debug(f"Initialized converter, direction: {direction}")

    def convert(self, source_path: str, target_path: str, **kwargs) -> ConversionResult:
        """
        Convert annotations between COCO and LabelMe formats.

        Args:
            source_path: Path to source annotations
            target_path: Path for target annotations
            **kwargs: Additional conversion parameters:
                - class_file: Required for LabelMe→COCO, optional for COCO→LabelMe
                - image_dir: Optional for both directions
                - do_rle: Whether to output RLE format (default False, only for LabelMe→COCO)

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
                errors=["Input validation failed"],
            )

        # 2. Read data using source handler
        source_handler = self.create_source_handler(source_path, kwargs)
        read_result = source_handler.read()
        if not read_result.success:
            return self._create_conversion_result(
                success=False,
                source_path=source_path,
                target_path=target_path,
                errors=read_result.errors,
            )

        # 3. Convert data (format-specific conversions like category mapping)
        annotations = read_result.data
        converted_annotations = self.convert_annotations(annotations, kwargs)

        # Store for potential use in create_target_handler (COCO→LabelMe case)
        self._source_annotations_for_target = converted_annotations

        # 4. Write data using target handler
        target_handler = self.create_target_handler(target_path, kwargs)
        write_result = target_handler.write(converted_annotations, target_path)

        # Clear stored annotations
        self._source_annotations_for_target = None

        # 5. Create result
        result = self._create_conversion_result(
            success=write_result.success,
            source_path=source_path,
            target_path=target_path,
            annotations=converted_annotations,
            write_result=write_result,
        )

        # 6. Add RLE accuracy warning if do_rle is True (LabelMe → COCO only)
        if not self.source_to_target:  # LabelMe → COCO
            do_rle = kwargs.get("do_rle", False)
            if do_rle:
                rle_converter = RLEConverter(logger=self.logger)
                warning_msg = rle_converter.get_rle_accuracy_warning()
                result.add_warning(warning_msg)
                self.logger.warning(f"RLE conversion accuracy loss: {warning_msg}")

        return result

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

        # Direction-specific validation
        if not self.source_to_target:  # LabelMe → COCO
            # Check required parameters for LabelMe→COCO
            class_file = kwargs.get("class_file")
            if not class_file:
                self.logger.error(
                    "class_file parameter is required for LabelMe→COCO conversion"
                )
                return False

            class_file_path = Path(class_file)
            if not class_file_path.exists():
                self.logger.error(f"Class file does not exist: {class_file}")
                return False

            # Check if RLE conversion is requested but pycocotools is not available
            do_rle = kwargs.get("do_rle", False)
            if do_rle:
                # Import HAS_COCO_MASK from coco_handler
                from ..label.coco_handler import HAS_COCO_MASK

                if not HAS_COCO_MASK:
                    error_msg = (
                        "RLE conversion requested (do_rle=True) but pycocotools is not available. "
                        "Install with: pip install pycocotools"
                    )
                    self.logger.error(error_msg)
                    raise ImportError(error_msg)

        else:  # COCO → LabelMe
            # For COCO→LabelMe, class_file is optional (can be extracted from COCO)
            # image_dir is optional (can be extracted from COCO or derived)
            pass

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
        if self.source_to_target:  # COCO → LabelMe
            handler = CocoAnnotationHandler(
                annotation_file=source_path, logger=self.logger
            )
        else:  # LabelMe → COCO
            class_file = kwargs.get("class_file")
            if not class_file:
                raise ValueError("class_file is required for LabelMe→COCO conversion")

            handler = LabelMeAnnotationHandler(
                label_dir=source_path, class_file=class_file, logger=self.logger
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
        if self.source_to_target:  # COCO → LabelMe
            # For LabelMe target, we need to create the output directory
            target_path_obj = Path(target_path)
            target_path_obj.mkdir(parents=True, exist_ok=True)

            # Get class_file from kwargs or extract from COCO
            class_file = kwargs.get("class_file")
            if not class_file:
                # Default to classes.txt in target directory
                class_file = str(target_path_obj / "classes.txt")

            # If class_file doesn't exist and we have source annotations, generate it
            class_file_path = Path(class_file)
            if not class_file_path.exists() and hasattr(
                self, "_source_annotations_for_target"
            ):
                if (
                    self._source_annotations_for_target
                    and self._source_annotations_for_target.categories
                ):
                    # Generate classes.txt from source annotations
                    from . import utils

                    if utils.generate_classes_file(
                        self._source_annotations_for_target.categories, class_file_path
                    ):
                        self.logger.info(
                            f"Generated class file from COCO categories: {class_file_path}"
                        )
                    else:
                        self.logger.warning(
                            f"Failed to generate class file: {class_file_path}"
                        )
                else:
                    self.logger.warning(
                        f"No categories available to generate class file: {class_file_path}"
                    )

            handler = LabelMeAnnotationHandler(
                label_dir=target_path, class_file=class_file, logger=self.logger
            )
        else:  # LabelMe → COCO
            # For COCO target, we need to create the JSON file
            target_path_obj = Path(target_path)
            target_path_obj.parent.mkdir(parents=True, exist_ok=True)

            do_rle = kwargs.get("do_rle", False)
            handler = CocoAnnotationHandler(
                annotation_file=target_path, logger=self.logger, do_rle=do_rle
            )

        return handler

    def convert_annotations(
        self, source_annotations: DatasetAnnotations, kwargs: Dict
    ) -> DatasetAnnotations:
        """
        Convert annotation data between COCO and LabelMe formats.

        Args:
            source_annotations: Annotations read from source format
            kwargs: Additional conversion parameters

        Returns:
            Converted DatasetAnnotations ready for writing to target format
        """
        # For COCO ↔ LabelMe conversion, the main differences are:
        # 1. Category management (already handled by handlers)
        # 2. COCO-specific fields (is_crowd, area, etc.)
        # 3. Dataset-level information

        # Default implementation: return as-is
        # Subclasses can override for format-specific conversions
        return source_annotations
