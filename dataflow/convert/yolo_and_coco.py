"""
YOLO and COCO format converter.

Handles bidirectional conversion between YOLO and COCO annotation formats.
Supports both object detection and instance segmentation annotations.
"""

from pathlib import Path
from typing import Any, Dict

from ..label.base import BaseAnnotationHandler
from ..label.coco_handler import CocoAnnotationHandler
from ..label.models import AnnotationFormat, DatasetAnnotations, ImageAnnotation, ObjectAnnotation
from ..label.yolo_handler import YoloAnnotationHandler
from .base import BaseConverter


class YoloAndCocoConverter(BaseConverter):
    """Converter for bidirectional conversion between YOLO and COCO formats."""

    def __init__(
        self,
        source_to_target: bool,
        prediction: bool = False,
        log_config=None,
        **kwargs,
    ):
        """
        Initialize converter.

        Args:
            source_to_target: True for YOLO→COCO, False for COCO→YOLO
            prediction: If True, read YOLO files in prediction format
                (with confidence scores). Only meaningful for YOLO→COCO.
                Default False (label format).
            log_config: Optional ``LogConfig`` instance for logging configuration.
            **kwargs: Arguments passed to BaseConverter
        """
        if source_to_target:
            source_format = "yolo"
            target_format = "coco"
        else:
            source_format = "coco"
            target_format = "yolo"

        super().__init__(source_format, target_format, log_config=log_config, **kwargs)
        self.source_to_target = source_to_target
        self.prediction = prediction

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
        if self.source_to_target:  # YOLO → COCO
            # Check required parameters for YOLO→COCO
            class_file = kwargs.get("class_file")
            if not class_file:
                self.logger.error("class_file parameter is required for YOLO→COCO conversion")
                return False

            class_file_path = Path(class_file)
            if not class_file_path.exists():
                self.logger.error(f"Class file does not exist: {class_file}")
                return False

            image_dir = kwargs.get("image_dir")
            if not image_dir:
                self.logger.error("image_dir parameter is required for YOLO→COCO conversion")
                return False

            image_dir_path = Path(image_dir)
            if not image_dir_path.exists():
                self.logger.error(f"Image directory does not exist: {image_dir}")
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

        else:  # COCO → YOLO
            # For COCO→YOLO, class_file is optional (can be extracted from COCO)
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
        if self.source_to_target:  # YOLO → COCO
            class_file = kwargs.get("class_file")
            if not class_file:
                raise ValueError("class_file is required for YOLO→COCO conversion")

            image_dir = kwargs.get("image_dir")
            if not image_dir:
                raise ValueError("image_dir is required for YOLO→COCO conversion")

            handler = YoloAnnotationHandler(
                label_dir=source_path,
                class_file=class_file,
                image_dir=image_dir,
                prediction=self.prediction,
                pose=kwargs.get("pose", False),
                strict_mode=self.strict_mode,
                logger=self.logger,
            )
        else:  # COCO → YOLO
            handler = CocoAnnotationHandler(
                annotation_file=source_path,
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
        if self.source_to_target:  # YOLO → COCO
            # For COCO target, we need to create the JSON file
            target_path_obj = Path(target_path)
            target_path_obj.parent.mkdir(parents=True, exist_ok=True)

            do_rle = kwargs.get("do_rle", False)
            handler = CocoAnnotationHandler(
                annotation_file=target_path,
                logger=self.logger,
                strict_mode=self.strict_mode,
                do_rle=do_rle,
                prediction=self.prediction,
            )
        else:  # COCO → YOLO
            # For YOLO target, we need to create appropriate directory structure
            target_path_obj = Path(target_path)

            # Create base directory
            target_path_obj.mkdir(parents=True, exist_ok=True)

            # Create labels directory for YOLO labels
            labels_dir = target_path_obj / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)

            # Create images directory for YOLO images
            images_dir = target_path_obj / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

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

            # Get image_dir from kwargs or use images_dir as default
            image_dir = kwargs.get("image_dir")
            if not image_dir:
                # If no image_dir provided, use the images directory we created
                image_dir = str(images_dir)

            # Pose sidecar (--save-kpts-config): generate a keypoint-
            # definition YAML from the COCO categories before iteration
            # (contract: spec_conversion.md §3.7.2)
            save_kpts_config = kwargs.get("save_kpts_config")
            if save_kpts_config and self._source_path:
                from .utils import read_coco_categories_meta, write_kpts_config_sidecar

                source_cats = {}
                if self._source_annotations_for_target is not None:
                    source_cats = self._source_annotations_for_target.categories
                meta = read_coco_categories_meta(self._source_path)
                ok, dropped = write_kpts_config_sidecar(source_cats, meta, str(save_kpts_config))
                if ok and dropped:
                    self.logger.warning(
                        f"Generated keypoint definition: {save_kpts_config} "
                        f"({dropped} out-of-range skeleton edge(s) dropped)"
                    )
                elif ok:
                    self.logger.info(f"Generated keypoint definition: {save_kpts_config}")
                else:
                    self.logger.warning(
                        "No keypoint metadata in COCO categories — sidecar not written"
                    )

            handler = YoloAnnotationHandler(
                label_dir=str(labels_dir),
                class_file=class_file,
                image_dir=image_dir,
                pose=kwargs.get("pose", False),
                strict_mode=self.strict_mode,
                logger=self.logger,
            )

        return handler

    def _ensure_categories_for_streaming(
        self,
        source_handler: BaseAnnotationHandler,
        source_path: str,
        kwargs: Dict,
    ) -> None:
        """Ensure COCO categories are loaded before streaming.

        For COCO→YOLO, delegates to
        ``convert.utils.ensure_coco_categories_for_streaming()``.
        """
        from .utils import ensure_coco_categories_for_streaming

        ensure_coco_categories_for_streaming(self, source_handler, source_path)

    def _convert_single_image(self, image_ann: ImageAnnotation, **kwargs) -> ImageAnnotation:
        """Convert a single ImageAnnotation from source to target format.

        Dispatches to ``_yolo_to_coco_one`` or ``_coco_to_yolo_one`` based
        on ``self.source_format``.
        """
        if self.source_format == "yolo":
            return self._yolo_to_coco_one(image_ann, pose=kwargs.get("pose", False))
        else:
            return self._coco_to_yolo_one(
                image_ann,
                save_crowd=kwargs.get("save_crowd", False),
                pose=kwargs.get("pose", False),
            )

    def _yolo_to_coco_one(self, img: ImageAnnotation, pose: bool = False) -> ImageAnnotation:
        """Convert single image: YOLO normalized center → COCO absolute px.

        Pose mode (contract: spec_conversion.md §3.7.1): keypoints are
        transformed to absolute pixels; objects without keypoints are
        skipped. Non-pose mode drops keypoints with a WARNING.
        """
        from .utils import (
            enforce_pose_k,
            keypoints_normalized_to_absolute,
            yolo_to_absolute_pixel,
        )

        new_objects = []
        dropped_kpts = 0
        skipped_no_kpts = 0
        for obj in img.objects:
            if pose and obj.keypoints is None:
                skipped_no_kpts += 1
                continue
            new_bbox, new_seg = yolo_to_absolute_pixel(
                obj.bbox, obj.segmentation, img.width, img.height
            )
            new_kpts = None
            if obj.keypoints is not None:
                if pose:
                    if not enforce_pose_k(self, len(obj.keypoints.points)):
                        continue
                    new_kpts = keypoints_normalized_to_absolute(
                        obj.keypoints, img.width, img.height
                    )
                else:
                    dropped_kpts += 1
            new_objects.append(
                ObjectAnnotation(
                    class_id=obj.class_id,
                    class_name=obj.class_name,
                    bbox=new_bbox,
                    segmentation=new_seg,
                    keypoints=new_kpts,
                    confidence=obj.confidence,
                    is_crowd=obj.is_crowd,
                )
            )

        if dropped_kpts:
            self._log_warning(
                f"Dropped keypoints from {dropped_kpts} object(s) in image "
                f"'{img.image_id}' (pass pose=True to convert them)"
            )
        if skipped_no_kpts:
            self._log_warning(
                f"Skipped {skipped_no_kpts} object(s) without keypoints in image "
                f"'{img.image_id}' (pose mode)"
            )

        return ImageAnnotation(
            image_id=img.image_id,
            image_path=img.image_path,
            width=img.width,
            height=img.height,
            objects=new_objects,
        )

    def _coco_to_yolo_one(
        self, img: ImageAnnotation, save_crowd: bool = False, pose: bool = False
    ) -> ImageAnnotation:
        """Convert single image: COCO absolute px → YOLO normalized center.

        YOLO has no crowd representation, so ``is_crowd=True`` objects are
        skipped by default (contract: spec_conversion.md §7.1);
        ``save_crowd=True`` writes them as ordinary rows.

        Pose mode (contract: spec_conversion.md §3.7.1): keypoints are
        transformed to normalized coordinates and segmentation is dropped
        (pose rows cannot carry it); objects without keypoints are skipped.
        Non-pose mode drops keypoints with a WARNING.
        """
        from .utils import (
            absolute_pixel_to_yolo,
            coco_file_name_to_image_id,
            enforce_pose_k,
            keypoints_absolute_to_normalized,
        )

        new_objects = []
        skipped_crowd = 0
        skipped_no_kpts = 0
        dropped_kpts = 0
        dropped_seg = 0
        for obj in img.objects:
            if obj.is_crowd and not save_crowd:
                skipped_crowd += 1
                continue
            if pose and obj.keypoints is None:
                skipped_no_kpts += 1
                continue
            new_bbox, new_seg = absolute_pixel_to_yolo(
                obj.bbox, obj.segmentation, img.width, img.height
            )
            new_kpts = None
            if obj.keypoints is not None:
                if pose:
                    if not enforce_pose_k(self, len(obj.keypoints.points)):
                        continue
                    new_kpts = keypoints_absolute_to_normalized(
                        obj.keypoints, img.width, img.height
                    )
                    if new_seg is not None:
                        dropped_seg += 1
                        new_seg = None
                else:
                    dropped_kpts += 1
            new_objects.append(
                ObjectAnnotation(
                    class_id=obj.class_id,
                    class_name=obj.class_name,
                    bbox=new_bbox,
                    segmentation=new_seg,
                    keypoints=new_kpts,
                    confidence=obj.confidence,
                    is_crowd=obj.is_crowd,
                )
            )

        if skipped_crowd:
            self._log_warning(
                f"Skipped {skipped_crowd} crowd annotation(s) in image "
                f"'{img.image_id}' (pass save_crowd=True to preserve them "
                "as ordinary rows)"
            )
        if skipped_no_kpts:
            self._log_warning(
                f"Skipped {skipped_no_kpts} object(s) without keypoints in image "
                f"'{img.image_id}' (pose mode)"
            )
        if dropped_kpts:
            self._log_warning(
                f"Dropped keypoints from {dropped_kpts} object(s) in image "
                f"'{img.image_id}' (pass pose=True to convert them)"
            )
        if dropped_seg:
            self._log_warning(
                f"Dropped segmentation from {dropped_seg} object(s) in image "
                f"'{img.image_id}' (pose rows cannot carry it)"
            )

        # YOLO format requires label files to share the stem with the image
        # file — derive the output name from the COCO file_name stem (which
        # preserves leading zeros), not from the numeric COCO id.
        return ImageAnnotation(
            image_id=coco_file_name_to_image_id(img.image_path, img.image_id),
            image_path=img.image_path,
            width=img.width,
            height=img.height,
            objects=new_objects,
        )

    def convert_annotations(
        self, source_annotations: DatasetAnnotations, kwargs: Dict
    ) -> DatasetAnnotations:
        """Convert annotation data between YOLO and COCO formats.

        Delegates to ``_convert_single_image()`` per image.
        """
        target_format = (
            AnnotationFormat.COCO if self.source_format == "yolo" else AnnotationFormat.YOLO
        )
        target = DatasetAnnotations(format=target_format)
        target.categories = source_annotations.categories.copy()

        for img in source_annotations.images:
            target.add_image(self._convert_single_image(img, **kwargs))

        # Pose metadata for COCO categories: keypoint names + skeleton from
        # the kpts_config (or placeholders) (contract: spec_conversion.md
        # §3.7.2). The COCO handler merges dataset_info["categories_meta"]
        # back into the output categories on write.
        if self.target_format == "coco" and kwargs.get("pose", False):
            from ..label.utils import load_kpts_config

            from .utils import build_categories_meta

            kpts_config = None
            cfg_path = kwargs.get("kpts_config")
            if cfg_path:
                kpts_config = load_kpts_config(Path(cfg_path))
            target.dataset_info["categories_meta"] = build_categories_meta(
                kpts_config, target.categories, self._pose_k or 0
            )

        return target
