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
    Keypoints,
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
                kpts_config=kwargs.get("kpts_config"),
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

            # Pose: build an in-memory keypoint definition from the COCO
            # categories so the LabelMe writer can name the point shapes;
            # also support the --save-kpts-config sidecar
            # (contract: spec_conversion.md §3.7.2).
            kpts_config_for_target = kwargs.get("kpts_config")
            save_kpts_config = kwargs.get("save_kpts_config")
            if self._source_path and (
                (not kpts_config_for_target and kwargs.get("pose", False)) or save_kpts_config
            ):
                from ..label.utils import KptsConfig

                from .utils import read_coco_categories_meta, write_kpts_config_sidecar

                source_cats = {}
                if self._source_annotations_for_target is not None:
                    source_cats = self._source_annotations_for_target.categories
                meta = read_coco_categories_meta(self._source_path)

                if save_kpts_config:
                    ok, dropped = write_kpts_config_sidecar(
                        source_cats, meta, str(save_kpts_config), logger=self.logger
                    )
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

                if not kpts_config_for_target and kwargs.get("pose", False) and meta:
                    classes = {}
                    skeletons = set()
                    for cat_id, entry in meta.items():
                        name = source_cats.get(cat_id)
                        if name and entry.get("keypoints"):
                            classes[name] = list(entry["keypoints"])
                            if entry.get("skeleton"):
                                skeletons.add(tuple(tuple(e) for e in entry["skeleton"]))
                    skeleton = list(skeletons.pop()) if len(skeletons) == 1 else None
                    kpts_config_for_target = KptsConfig(
                        keypoints=None,
                        classes=classes or None,
                        skeleton=skeleton,
                        sigmas=None,
                    )

            handler = LabelMeAnnotationHandler(
                label_dir=target_path,
                class_file=class_file,
                kpts_config=kpts_config_for_target,
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

        LabelMe has no crowd representation, so on the COCO → LabelMe
        direction ``is_crowd=True`` objects are skipped by default
        (contract: spec_conversion.md §7.1); ``save_crowd=True`` writes
        them as ordinary shapes.
        """
        if self.source_format == "coco" and not kwargs.get("save_crowd", False):
            crowd_count = sum(1 for obj in image_ann.objects if obj.is_crowd)
            if crowd_count:
                self._log_warning(
                    f"Skipped {crowd_count} crowd annotation(s) in image "
                    f"'{image_ann.image_id}' (pass save_crowd=True to preserve "
                    "them as ordinary shapes)"
                )
                image_ann = ImageAnnotation(
                    image_id=image_ann.image_id,
                    image_path=image_ann.image_path,
                    width=image_ann.width,
                    height=image_ann.height,
                    objects=[obj for obj in image_ann.objects if not obj.is_crowd],
                )

        # COCO → LabelMe only: rewrite image_id to the COCO file_name stem so
        # the output .json shares the stem with the image file (LabelMe format
        # requirement; leading zeros preserved). LabelMe → COCO must NOT
        # rewrite — the COCO output id derives from the numeric image_id.
        new_image_id = image_ann.image_id
        if self.source_format == "coco":
            from .utils import coco_file_name_to_image_id

            new_image_id = coco_file_name_to_image_id(image_ann.image_path, image_ann.image_id)

        from .utils import enforce_pose_k

        pose = kwargs.get("pose", False)
        new_objects = []
        dropped_kpts = 0
        skipped_no_kpts = 0
        for obj in image_ann.objects:
            if pose and obj.keypoints is None:
                skipped_no_kpts += 1
                continue
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

            # Keypoints pass through unchanged — both formats use absolute
            # pixels (contract: spec_conversion.md §3.7).
            new_kpts = None
            if obj.keypoints is not None:
                if pose:
                    if not enforce_pose_k(self, len(obj.keypoints.points)):
                        continue
                    new_kpts = Keypoints(
                        points=list(obj.keypoints.points),
                        visibility=list(obj.keypoints.visibility),
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
                f"'{image_ann.image_id}' (pass pose=True to convert them)"
            )
        if skipped_no_kpts:
            self._log_warning(
                f"Skipped {skipped_no_kpts} object(s) without keypoints in image "
                f"'{image_ann.image_id}' (pose mode)"
            )

        return ImageAnnotation(
            image_id=new_image_id,
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

        # Pose metadata for COCO categories: keypoint names + skeleton from
        # the kpts_config (or placeholders) (contract: spec_conversion.md
        # §3.7.2).
        if self.target_format == "coco" and kwargs.get("pose", False):
            from ..label.utils import load_kpts_config

            from .utils import build_categories_meta

            kpts_config = None
            cfg_path = kwargs.get("kpts_config")
            if cfg_path:
                kpts_config = load_kpts_config(Path(cfg_path))
            target.dataset_info["categories_meta"] = build_categories_meta(
                kpts_config, target.categories, self._pose_k or 0, logger=self.logger
            )

        return target
