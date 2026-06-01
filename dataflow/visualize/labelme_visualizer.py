"""
LabelMe annotation visualizer.

Visualizes LabelMe JSON annotation format.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from dataflow.label.labelme_handler import LabelMeAnnotationHandler

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
        self.handler = LabelMeAnnotationHandler(
            label_dir=str(label_dir),
            class_file=str(class_file) if class_file else None,
            strict_mode=self.strict_mode,
            logger=self.logger,
        )

    def load_annotations(self) -> Dict[str, RenderData]:
        """Load LabelMe annotation data and convert to RenderData.

        LabelMe coordinates are in absolute pixels.
        Converts to [x1,y1,x2,y2] format for rendering.
        """
        result = self.handler.read()
        if not result.success:
            raise ValueError(
                f"Failed to load LabelMe annotations: {result.message}"
            )

        dataset = result.data
        render_data_map: Dict[str, RenderData] = {}

        for image_ann in dataset.images:
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
                        (int(x), int(y)) for x, y in obj.segmentation.points
                    ]

                render_annotations.append(render_ann)

            render_data_map[image_ann.image_path] = RenderData(
                annotations=render_annotations,
                image_width=image_ann.width,
                image_height=image_ann.height,
            )

        return render_data_map
