"""
YOLO annotation visualizer.

Visualizes YOLO format annotation files.
Supports both object detection and instance segmentation formats.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

from dataflow.label.models import ImageAnnotation
from dataflow.label.yolo_handler import YoloAnnotationHandler

from .base import BaseVisualizer, RenderAnnotation, RenderData


class YOLOVisualizer(BaseVisualizer):
    """YOLO format visualizer."""

    def __init__(
        self,
        label_dir: Union[str, Path],
        image_dir: Union[str, Path],
        class_file: Union[str, Path],
        log_config=None,
        **kwargs,
    ):
        """
        Initialize YOLO visualizer.

        Args:
            label_dir: YOLO label directory (contains TXT files)
            image_dir: Image directory
            class_file: Class file path (required)
            log_config: Optional ``LogConfig`` instance for logging configuration.
            **kwargs: Additional arguments for BaseVisualizer
        """
        super().__init__(label_dir, image_dir, log_config=log_config, **kwargs)
        self.class_file = Path(class_file)

        self.logger.debug(
            f"YOLO visualizer initialization complete, "
            f"class file: {class_file}"
        )

    def _create_handler(self) -> YoloAnnotationHandler:
        """Create a YOLO handler for streaming iteration."""
        return YoloAnnotationHandler(
            label_dir=str(self.label_dir),
            class_file=str(self.class_file),
            image_dir=str(self.image_dir),
            strict_mode=False,
            logger=self.logger,
        )

    def _convert_to_render_data(
        self, image_ann: ImageAnnotation
    ) -> RenderData:
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
                render_ann.polygon = [
                    (int(x * w), int(y * h))
                    for x, y in obj.segmentation.points
                ]

            render_annotations.append(render_ann)

        return RenderData(
            annotations=render_annotations,
            image_width=w,
            image_height=h,
        )
