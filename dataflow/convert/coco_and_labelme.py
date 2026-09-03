"""
COCO and LabelMe format converter.

Handles bidirectional conversion between COCO and LabelMe annotation formats.
Supports both object detection and instance segmentation annotations.
"""

from pathlib import Path
from typing import Any, Dict

from ..label.base import BaseAnnotationHandler
from ..label.coco_handler import CocoAnnotationHandler
from ..label.labelme_handler import LabelMeAnnotationHandler
from ..label.models import (
    AnnotationFormat,
    BoundingBox,
    DatasetAnnotations,
    ImageAnnotation,
    ObjectAnnotation,
    Segmentation,
)
from .base import BaseConverter


class CocoAndLabelMeConverter(BaseConverter):
    """Converter for bidirectional conversion between COCO and LabelMe formats."""

    def __init__(self, source_to_target: bool, log_config=None, **kwargs):
        """
        Initialize converter.

        Args:
            source_to_target: True for COCO→LabelMe, False for LabelMe→COCO
            log_config: Optional ``LogConfig`` instance for logging configuration.
            **kwargs: Arguments passed to BaseConverter
        """
        if source_to_target:
            source_format = "coco"
            target_format = "labelme"
        else:
            source_format = "labelme"
            target_format = "coco"

        super().__init__(source_format, target_format, log_config=log_config, **kwargs)
        self.source_to_target = source_to_target

        direction = "COCO→LabelMe" if source_to_target else "LabelMe→COCO"
        self.logger.debug(f"Initialized converter, direction: {direction}")

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
                self.logger.error("class_file parameter is required for LabelMe→COCO conversion")
                return False

            class_file_path = Path(class_file)
            if not class_file_path.exists():
                self.logger.error(f"Class file does not exist: {class_file}")
                return False

            # Check if RLE conversion is requested but pycocotools is not available
            do_rle = kwargs.get("do_rle", False)
            if do_rle:
                try:
                    from pycocotools import mask as coco_mask  # noqa: F401

                    _has_coco = True
                except ImportError:
                    _has_coco = False

                if not _has_coco:
                    error_msg = (
                        "RLE conversion requested (do_rle=True) but pycocotools is not available. "
                        "Install with: pip install pycocotools"
                    )
                    self.logger.error(error_msg)
                    return False

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
                annotation_file=source_path,
                strict_mode=self.strict_mode,
                logger=self.logger,
            )
        else:  # LabelMe → COCO
            class_file = kwargs.get("class_file")
            if not class_file:
                raise ValueError("class_file is required for LabelMe→COCO conversion")

            handler = LabelMeAnnotationHandler(
                label_dir=source_path,
                class_file=class_file,
                strict_mode=self.strict_mode,
                logger=self.logger,
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
            if not class_file_path.exists() and hasattr(self, "_source_annotations_for_target"):
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
                        self.logger.warning(f"Failed to generate class file: {class_file_path}")
                else:
                    self.logger.warning(
                        f"No categories available to generate class file: {class_file_path}"
                    )

            handler = LabelMeAnnotationHandler(
                label_dir=target_path,
                class_file=class_file,
                strict_mode=self.strict_mode,
                logger=self.logger,
            )
        else:  # LabelMe → COCO
            # For COCO target, we need to create the JSON file
            target_path_obj = Path(target_path)
            target_path_obj.parent.mkdir(parents=True, exist_ok=True)

            do_rle = kwargs.get("do_rle", False)
            handler = CocoAnnotationHandler(
                annotation_file=target_path,
                logger=self.logger,
                strict_mode=self.strict_mode,
                do_rle=do_rle,
            )

        return handler

    def _ensure_categories_for_streaming(
        self,
        source_handler: BaseAnnotationHandler,
        source_path: str,
        kwargs: Dict,
    ) -> None:
        """Ensure COCO categories are loaded before streaming.

        For COCO→LabelMe, delegates to
        ``convert.utils.ensure_coco_categories_for_streaming()``.
        """
        from .utils import ensure_coco_categories_for_streaming

        ensure_coco_categories_for_streaming(self, source_handler, source_path)

    def _convert_single_image(self, image_ann: ImageAnnotation, **kwargs) -> ImageAnnotation:
        """Convert a single ImageAnnotation between COCO and LabelMe.

        Both formats share the same absolute-pixel coordinate semantics.
        Only the structural representation differs — coordinate values pass
        through unchanged.
        """
        new_objects = []
        for obj in image_ann.objects:
            new_bbox = None
            new_seg = None

            if obj.bbox:
                new_bbox = BoundingBox(
                    x=obj.bbox.x,
                    y=obj.bbox.y,
                    width=obj.bbox.width,
                    height=obj.bbox.height,
                )

            if obj.segmentation:
                new_seg = Segmentation(
                    points=obj.segmentation.points.copy(),
                    rle=obj.segmentation.rle,
                )

            new_objects.append(
                ObjectAnnotation(
                    class_id=obj.class_id,
                    class_name=obj.class_name,
                    bbox=new_bbox,
                    segmentation=new_seg,
                    confidence=obj.confidence,
                    is_crowd=obj.is_crowd,
                )
            )

        return ImageAnnotation(
            image_id=image_ann.image_id,
            image_path=image_ann.image_path,
            width=image_ann.width,
            height=image_ann.height,
            objects=new_objects,
        )

    def convert_annotations(
        self, source_annotations: DatasetAnnotations, kwargs: Dict
    ) -> DatasetAnnotations:
        """Convert annotation data between COCO and LabelMe formats.

        Both formats use absolute pixel coordinates with identical semantics.
        Delegates to ``_convert_single_image()`` per image.
        """
        target_format = (
            AnnotationFormat.LABELME if self.target_format == "labelme" else AnnotationFormat.COCO
        )
        target = DatasetAnnotations(format=target_format)
        target.categories = source_annotations.categories.copy()

        for img in source_annotations.images:
            target.add_image(self._convert_single_image(img, **kwargs))

        return target
