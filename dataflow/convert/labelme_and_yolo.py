"""
LabelMe and YOLO format converter.

Handles bidirectional conversion between LabelMe and YOLO annotation formats.
Supports both object detection and instance segmentation annotations.
"""

import shutil
from pathlib import Path
from typing import Any, Dict

from ..label.labelme_handler import LabelMeAnnotationHandler
from ..label.models import AnnotationFormat, DatasetAnnotations, ImageAnnotation, ObjectAnnotation
from ..label.yolo_handler import YoloAnnotationHandler
from .base import BaseConverter


class LabelMeAndYoloConverter(BaseConverter):
    """Converter for bidirectional conversion between LabelMe and YOLO formats."""

    def __init__(self, source_to_target: bool, log_config=None, **kwargs):
        """
        Initialize converter.

        Args:
            source_to_target: True for LabelMe→YOLO, False for YOLO→LabelMe
            log_config: Optional ``LogConfig`` instance for logging configuration.
            **kwargs: Arguments passed to BaseConverter
        """
        if source_to_target:
            source_format = "labelme"
            target_format = "yolo"
        else:
            source_format = "yolo"
            target_format = "labelme"

        super().__init__(source_format, target_format, log_config=log_config, **kwargs)
        self.source_to_target = source_to_target

        direction = "LabelMe→YOLO" if source_to_target else "YOLO→LabelMe"
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
                strict_mode=self.strict_mode,
                logger=self.logger,
            )
        else:  # YOLO → LabelMe
            image_dir = kwargs.get("image_dir")
            if not image_dir:
                raise ValueError("image_dir is required for YOLO→LabelMe conversion")

            handler = YoloAnnotationHandler(
                label_dir=source_path,
                class_file=class_file,
                image_dir=image_dir,
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
            source_class_file = Path(class_file)
            target_class_file = target_path_obj / "classes.txt"
            if source_class_file.exists() and not target_class_file.exists():
                try:
                    shutil.copy2(source_class_file, target_class_file)
                    self.logger.info(f"Copied class file to: {target_class_file}")
                    class_file = str(target_class_file)
                except Exception as e:
                    self.logger.warning(f"Failed to copy class file: {e}")

            # Get image_dir from kwargs or use images_dir as default
            image_dir = kwargs.get("image_dir")
            if not image_dir:
                # If no image_dir provided, use the images directory we created
                image_dir = str(images_dir)

            handler = YoloAnnotationHandler(
                label_dir=str(labels_dir),
                class_file=class_file,
                image_dir=image_dir,
                strict_mode=self.strict_mode,
                logger=self.logger,
            )
        else:  # YOLO → LabelMe
            # For LabelMe target, just create the output directory
            target_path_obj = Path(target_path)
            target_path_obj.mkdir(parents=True, exist_ok=True)

            handler = LabelMeAnnotationHandler(
                label_dir=target_path,
                class_file=class_file,
                strict_mode=self.strict_mode,
                logger=self.logger,
            )

        return handler

    def _ensure_categories_for_streaming(
        self,
        source_handler: Any,
        source_path: str,
        kwargs: Dict,
    ) -> None:
        """Pre-load categories for streaming conversion.

        Uses the source handler's ``categories`` dict (already loaded
        during handler construction). For LabelMe source with a class
        file, categories come from the class file; without a class
        file, LabelMe auto-detects categories during construction.
        For YOLO source, categories always come from the class_file.
        """
        self._source_annotations_for_target = None
        cats = getattr(source_handler, "categories", None)
        if isinstance(cats, dict) and cats:
            self._source_annotations_for_target = DatasetAnnotations(
                format=AnnotationFormat.UNKNOWN,
                categories=cats.copy(),
            )

    def _post_stream_image(
        self,
        source_ann: ImageAnnotation,
        target_ann: ImageAnnotation,
        target_path: str,
        kwargs: Dict,
    ) -> None:
        """Copy the source image to the target directory (LabelMe→YOLO only).

        This is called per-image during the streaming loop, ensuring each
        image is copied immediately after its annotation is written.
        """
        if not self.source_to_target:
            return  # YOLO→LabelMe: image_dir already points to source images

        target_path_obj = Path(target_path)
        images_dir = target_path_obj / "images"

        source_image_path = Path(source_ann.image_path)
        if not source_image_path.is_absolute():
            # Resolve relative to the source path
            source_image_path = Path(self._source_path) / source_image_path

        target_image_path = images_dir / source_image_path.name

        try:
            # `exists()` is an optimization, not a TOCTOU guard —
            # `copy2()` overwriting an already-present file is harmless
            # (content is identical by stem), and the only downside is
            # redundant I/O.  A genuine race here would at worst cost an
            # extra copy.
            if not target_image_path.exists():
                shutil.copy2(source_image_path, target_image_path)
        except FileNotFoundError:
            self.logger.warning(f"Source image does not exist, skipping copy: {source_image_path}")
        except OSError as e:
            self.logger.warning(f"Failed to copy image {source_image_path}: {e}")

    def _convert_single_image(self, image_ann: ImageAnnotation, **kwargs) -> ImageAnnotation:
        """Convert a single ImageAnnotation from source to target format.

        Dispatches based on source format.
        """
        if self.source_format == "labelme":
            return self._absolute_to_normalized_one(image_ann)
        else:
            return self._normalized_to_absolute_one(image_ann)

    def _absolute_to_normalized_one(self, img: ImageAnnotation) -> ImageAnnotation:
        """Convert single image: LabelMe absolute px → YOLO normalized center."""
        from .utils import absolute_pixel_to_yolo

        new_objects = []
        for obj in img.objects:
            new_bbox, new_seg = absolute_pixel_to_yolo(
                obj.bbox, obj.segmentation, img.width, img.height
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
            image_id=img.image_id,
            image_path=img.image_path,
            width=img.width,
            height=img.height,
            objects=new_objects,
        )

    def _normalized_to_absolute_one(self, img: ImageAnnotation) -> ImageAnnotation:
        """Convert single image: YOLO normalized center → LabelMe absolute px."""
        from .utils import yolo_to_absolute_pixel

        new_objects = []
        for obj in img.objects:
            new_bbox, new_seg = yolo_to_absolute_pixel(
                obj.bbox, obj.segmentation, img.width, img.height
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
            image_id=img.image_id,
            image_path=img.image_path,
            width=img.width,
            height=img.height,
            objects=new_objects,
        )

    def convert_annotations(
        self, source_annotations: DatasetAnnotations, kwargs: Dict
    ) -> DatasetAnnotations:
        """Convert annotation data between LabelMe and YOLO formats.

        Delegates to ``_convert_single_image()`` per image.
        """
        target_format = (
            AnnotationFormat.YOLO if self.target_format == "yolo" else AnnotationFormat.LABELME
        )
        target = DatasetAnnotations(format=target_format)
        target.categories = source_annotations.categories.copy()

        for img in source_annotations.images:
            target.add_image(self._convert_single_image(img, **kwargs))

        return target
