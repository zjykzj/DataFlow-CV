"""
YOLO annotation visualizer.

Visualizes YOLO format annotation files.
Supports both object detection and instance segmentation formats.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

from dataflow.label.yolo_handler import YoloAnnotationHandler

from .base import BaseVisualizer, RenderAnnotation, RenderData


class YOLOVisualizer(BaseVisualizer):
    """YOLO format visualizer."""

    def __init__(
        self,
        label_dir: Union[str, Path],
        image_dir: Union[str, Path],
        class_file: Union[str, Path],
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize YOLO visualizer.

        Args:
            label_dir: YOLO label directory (contains TXT files)
            image_dir: Image directory
            class_file: Class file path (required)
            verbose: Whether to enable verbose logging
            **kwargs: Additional arguments for BaseVisualizer
        """
        super().__init__(label_dir, image_dir, verbose=verbose, **kwargs)
        self.class_file = Path(class_file)
        self.handler = YoloAnnotationHandler(
            label_dir=str(label_dir),
            class_file=str(class_file),
            image_dir=str(image_dir),
            strict_mode=self.strict_mode,
            logger=self.logger,
        )

        if verbose:
            self.logger.debug(
                f"YOLO visualizer initialization complete, class file: {class_file}"
            )

    def load_annotations(self) -> Dict[str, RenderData]:
        """Load YOLO annotation data and convert to RenderData.

        YOLO coordinates are normalized [0,1] center-based.
        Converts to absolute pixel integers for rendering.
        """
        result = self.handler.read()
        if not result.success:
            raise ValueError(f"Failed to load YOLO annotations: {result.message}")

        dataset = result.data
        render_data_map: Dict[str, RenderData] = {}

        for image_ann in dataset.images:
            render_annotations: List[RenderAnnotation] = []
            w, h = image_ann.width, image_ann.height

            for obj in image_ann.objects:
                render_ann = RenderAnnotation(
                    class_name=obj.class_name,
                    class_id=obj.class_id,
                )

                if obj.bbox:
                    # YOLO normalized center → absolute pixel [x1,y1,x2,y2]
                    cx = int(obj.bbox.x * w)
                    cy = int(obj.bbox.y * h)
                    bw = int(obj.bbox.width * w / 2)
                    bh = int(obj.bbox.height * h / 2)
                    render_ann.bbox = (cx - bw, cy - bh, cx + bw, cy + bh)

                if obj.segmentation:
                    # YOLO normalized points → absolute pixel points
                    render_ann.polygon = [
                        (int(x * w), int(y * h)) for x, y in obj.segmentation.points
                    ]

                render_annotations.append(render_ann)

            render_data_map[image_ann.image_path] = RenderData(
                annotations=render_annotations,
                image_width=w,
                image_height=h,
            )

        return render_data_map
