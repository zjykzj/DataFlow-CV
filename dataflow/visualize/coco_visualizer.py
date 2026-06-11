"""
COCO annotation visualizer.

Visualizes COCO format annotation files.
Supports both polygon and RLE segmentation formats.
"""

from pathlib import Path
from typing import Dict, List, Union

from dataflow.label.coco_handler import CocoAnnotationHandler
from dataflow.label.models import ImageAnnotation

from .base import BaseVisualizer, RenderAnnotation, RenderData


class COCOVisualizer(BaseVisualizer):
    """COCO format visualizer."""

    def __init__(
        self,
        annotation_file: Union[str, Path],
        image_dir: Union[str, Path],
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize COCO visualizer.

        Args:
            annotation_file: COCO JSON annotation file path
            image_dir: Image directory
            verbose: Whether to enable verbose logging
            **kwargs: Additional arguments for BaseVisualizer
        """
        super().__init__(annotation_file, image_dir, verbose=verbose, **kwargs)
        self.annotation_file = Path(annotation_file)

        if verbose:
            self.logger.debug(
                f"COCO visualizer initialization complete, "
                f"annotation file: {annotation_file}"
            )

    def _create_handler(self) -> CocoAnnotationHandler:
        """Create a COCO handler for streaming iteration."""
        return CocoAnnotationHandler(
            annotation_file=str(self.annotation_file),
            strict_mode=self.strict_mode,
            logger=self.logger,
        )

    def _convert_to_render_data(
        self, image_ann: ImageAnnotation
    ) -> RenderData:
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
                    render_ann.polygon = [
                        (int(x), int(y))
                        for x, y in obj.segmentation.points
                    ]

            render_annotations.append(render_ann)

        return RenderData(
            annotations=render_annotations,
            image_width=image_ann.width,
            image_height=image_ann.height,
        )
