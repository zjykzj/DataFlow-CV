"""
COCO annotation visualizer.

Visualizes COCO format annotation files.
Supports both polygon and RLE segmentation formats.
"""

from pathlib import Path
from typing import List, Union

from dataflow.label.coco_handler import CocoAnnotationHandler
from dataflow.label.models import ImageAnnotation

from .base import BaseVisualizer, RenderAnnotation, RenderData


class COCOVisualizer(BaseVisualizer):
    """COCO format visualizer."""

    def __init__(
        self,
        annotation_file: Union[str, Path],
        image_dir: Union[str, Path],
        log_config=None,
        **kwargs,
    ):
        """
        Initialize COCO visualizer.

        Args:
            annotation_file: COCO JSON annotation file path
            image_dir: Image directory
            log_config: Optional ``LogConfig`` instance for logging configuration.
            **kwargs: Additional arguments for BaseVisualizer
        """
        super().__init__(annotation_file, image_dir, log_config=log_config, **kwargs)
        self.annotation_file = Path(annotation_file)
        self._kpts_skeleton_warned = False

        self.logger.debug(
            f"COCO visualizer initialization complete, annotation file: {annotation_file}"
        )

    def _create_handler(self) -> CocoAnnotationHandler:
        """Create a COCO handler for streaming iteration."""
        return CocoAnnotationHandler(
            annotation_file=str(self.annotation_file),
            strict_mode=False,
            logger=self.logger,
        )

    def _convert_to_render_data(self, image_ann: ImageAnnotation) -> RenderData:
        """Convert a single COCO ImageAnnotation to RenderData.

        COCO coordinates are in absolute pixels (top-left bbox).
        Converts to [x1,y1,x2,y2] format for rendering.
        """
        render_annotations: List[RenderAnnotation] = []

        for obj in image_ann.objects:
            render_ann = RenderAnnotation(
                class_name=obj.class_name,
                class_id=obj.class_id,
            )

            if obj.bbox:
                # COCO bbox: (x_tl, y_tl, w, h) in absolute pixels
                # Render: [x1, y1, x2, y2]
                x1 = int(obj.bbox.x)
                y1 = int(obj.bbox.y)
                x2 = int(obj.bbox.x + obj.bbox.width)
                y2 = int(obj.bbox.y + obj.bbox.height)
                render_ann.bbox = (x1, y1, x2, y2)

            if obj.segmentation:
                if obj.segmentation.has_rle():
                    render_ann.rle = obj.segmentation.rle
                else:
                    # Polygon points are already in absolute pixels
                    render_ann.polygon = [(int(x), int(y)) for x, y in obj.segmentation.points]

            if obj.keypoints:
                # Absolute pixel points as-is, v as-is
                render_ann.keypoints = [
                    (int(x), int(y), v)
                    for (x, y), v in zip(obj.keypoints.points, obj.keypoints.visibility)
                ]
                # Skeleton from the COCO categories metadata (the handler
                # is stored on the visualizer during visualize()).
                meta = {}
                if self._handler is not None:
                    meta = getattr(self._handler, "categories_meta", {}).get(obj.class_id, {})
                if meta.get("skeleton"):
                    render_ann.skeleton = [(int(i), int(j)) for i, j in meta["skeleton"]]
                elif not self._kpts_skeleton_warned:
                    self.logger.warning(
                        "No skeleton metadata for keypoint objects — keypoints are "
                        "drawn without skeleton lines"
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
