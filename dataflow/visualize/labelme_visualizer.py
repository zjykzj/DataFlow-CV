"""
LabelMe annotation visualizer.

Visualizes LabelMe JSON annotation format.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dataflow.label.labelme_handler import LabelMeAnnotationHandler
from dataflow.label.models import ImageAnnotation

from .base import BaseVisualizer, RenderAnnotation, RenderData


class LabelMeVisualizer(BaseVisualizer):
    """LabelMe format visualizer."""

    def __init__(
        self,
        label_dir: Union[str, Path],
        image_dir: Union[str, Path],
        class_file: Optional[Union[str, Path]] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize LabelMe visualizer.

        Args:
            label_dir: LabelMe label directory (contains JSON files)
            image_dir: Image directory
            class_file: Optional class file path
            verbose: Whether to enable verbose logging
            **kwargs: Additional arguments for BaseVisualizer
        """
        super().__init__(label_dir, image_dir, verbose=verbose, **kwargs)
        self.class_file = Path(class_file) if class_file else None

    def _create_handler(self) -> LabelMeAnnotationHandler:
        """Create a LabelMe handler for streaming iteration."""
        kwargs: Dict[str, Any] = dict(
            strict_mode=False,
            logger=self.logger,
        )
        if self.class_file:
            kwargs["class_file"] = str(self.class_file)
        return LabelMeAnnotationHandler(
            label_dir=str(self.label_dir),
            **kwargs,
        )

    def _convert_to_render_data(
        self, image_ann: ImageAnnotation
    ) -> RenderData:
        """Convert a single LabelMe ImageAnnotation to RenderData.

        LabelMe coordinates are in absolute pixels.
        Converts to [x1,y1,x2,y2] format for rendering.
        """
        render_annotations: List[RenderAnnotation] = []

        for obj in image_ann.objects:
            render_ann = RenderAnnotation(
                class_name=obj.class_name,
                class_id=obj.class_id,
            )

            if obj.bbox:
                # LabelMe bbox: (x_tl, y_tl, w, h) in absolute pixels
                # Render: [x1, y1, x2, y2]
                x1 = int(obj.bbox.x)
                y1 = int(obj.bbox.y)
                x2 = int(obj.bbox.x + obj.bbox.width)
                y2 = int(obj.bbox.y + obj.bbox.height)
                render_ann.bbox = (x1, y1, x2, y2)

            if obj.segmentation:
                # Points are already in absolute pixels
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
