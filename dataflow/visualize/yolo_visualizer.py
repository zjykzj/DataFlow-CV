"""
YOLO annotation visualizer.

Visualizes YOLO format annotation files.
Supports both object detection and instance segmentation formats.
"""

from pathlib import Path
from typing import List, Optional, Union

from dataflow.label.models import ImageAnnotation
from dataflow.label.utils import KptsConfig, load_kpts_config
from dataflow.label.yolo_handler import YoloAnnotationHandler

from .base import BaseVisualizer, RenderAnnotation, RenderData


class YOLOVisualizer(BaseVisualizer):
    """YOLO format visualizer."""

    def __init__(
        self,
        label_dir: Union[str, Path],
        image_dir: Union[str, Path],
        class_file: Union[str, Path],
        pose: bool = False,
        kpts_config: Optional[Union[str, Path]] = None,
        log_config=None,
        **kwargs,
    ):
        """
        Initialize YOLO visualizer.

        Args:
            label_dir: YOLO label directory (contains TXT files)
            image_dir: Image directory
            class_file: Class file path (required)
            pose: If True, parse YOLO pose rows (explicit — see
                spec_yolo_format.md §2.3). Default False.
            kpts_config: Optional keypoint-definition YAML — the skeleton
                source for pose visualization.
            log_config: Optional ``LogConfig`` instance for logging configuration.
            **kwargs: Additional arguments for BaseVisualizer
        """
        super().__init__(label_dir, image_dir, log_config=log_config, **kwargs)
        self.class_file = Path(class_file)
        self.pose = pose
        self._kpts_config: Optional[KptsConfig] = None
        if kpts_config:
            self._kpts_config = load_kpts_config(Path(kpts_config))
            self.logger.debug(f"Loaded keypoint definition from {kpts_config}")
        self._kpts_skeleton_warned = False

        self.logger.debug(f"YOLO visualizer initialization complete, class file: {class_file}")

    def _create_handler(self) -> YoloAnnotationHandler:
        """Create a YOLO handler for streaming iteration."""
        return YoloAnnotationHandler(
            label_dir=str(self.label_dir),
            class_file=str(self.class_file),
            image_dir=str(self.image_dir),
            pose=self.pose,
            strict_mode=False,
            logger=self.logger,
        )

    def _convert_to_render_data(self, image_ann: ImageAnnotation) -> RenderData:
        """Convert a single YOLO ImageAnnotation to RenderData.

        YOLO coordinates are normalized [0,1] center-based.
        Converts to absolute pixel integers for rendering.
        """
        render_annotations: List[RenderAnnotation] = []
        w, h = image_ann.width, image_ann.height

        for obj in image_ann.objects:
            render_ann = RenderAnnotation(
                class_name=obj.class_name,
                class_id=obj.class_id,
            )

            if obj.bbox:
                # YOLO normalized center → absolute pixel [x1,y1,x2,y2]
                # Defer int() to final result to minimize precision loss
                cx = obj.bbox.x * w
                cy = obj.bbox.y * h
                half_w = obj.bbox.width * w / 2
                half_h = obj.bbox.height * h / 2
                x1 = int(cx - half_w)
                y1 = int(cy - half_h)
                x2 = int(cx + half_w)
                y2 = int(cy + half_h)
                render_ann.bbox = (x1, y1, x2, y2)

            if obj.segmentation:
                # YOLO normalized points → absolute pixel points
                render_ann.polygon = [(int(x * w), int(y * h)) for x, y in obj.segmentation.points]

            if obj.keypoints:
                # YOLO normalized points → absolute pixel points, v as-is
                render_ann.keypoints = [
                    (int(x * w), int(y * h), v)
                    for (x, y), v in zip(obj.keypoints.points, obj.keypoints.visibility)
                ]
                # Skeleton from the keypoint-definition YAML (spec_visualize.md §2.3)
                if self._kpts_config is not None and self._kpts_config.skeleton:
                    render_ann.skeleton = [(int(i), int(j)) for i, j in self._kpts_config.skeleton]
                elif not self._kpts_skeleton_warned:
                    self.logger.warning(
                        "No kpts_config provided — keypoints are drawn without skeleton lines"
                    )
                    self._kpts_skeleton_warned = True

            # Bbox from polygon fallback: compute axis-aligned bounds
            if render_ann.bbox is None and render_ann.polygon:
                render_ann.bbox = self._compute_bbox_from_polygon(render_ann.polygon)
            # Bbox from keypoints fallback: labeled (v>0) point extent
            elif render_ann.bbox is None and render_ann.keypoints:
                if any(v > 0 for _, _, v in render_ann.keypoints):
                    render_ann.bbox = self._compute_bbox_from_keypoints(render_ann.keypoints)

            render_annotations.append(render_ann)

        return RenderData(
            annotations=render_annotations,
        )
