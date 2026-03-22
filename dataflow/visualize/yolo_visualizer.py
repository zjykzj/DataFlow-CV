"""
YOLO annotation visualizer.

Visualizes YOLO format annotation files.
Supports both object detection and instance segmentation formats.
"""

from typing import Union
from pathlib import Path

from .base import BaseVisualizer
from dataflow.label.yolo_handler import YoloAnnotationHandler


class YOLOVisualizer(BaseVisualizer):
    """YOLO format visualizer."""

    def __init__(self,
                 label_dir: Union[str, Path],
                 image_dir: Union[str, Path],
                 class_file: Union[str, Path],
                 **kwargs):
        """
        Initialize YOLO visualizer.

        Args:
            label_dir: YOLO label directory (contains TXT files)
            image_dir: Image directory
            class_file: Class file path (required)
            **kwargs: Additional arguments for BaseVisualizer
        """
        super().__init__(label_dir, image_dir, **kwargs)
        self.class_file = Path(class_file)
        self.handler = YoloAnnotationHandler(
            label_dir=str(label_dir),
            class_file=str(class_file),
            image_dir=str(image_dir),
            strict_mode=self.strict_mode,
            logger=self.logger
        )

    def load_annotations(self):
        """Load YOLO annotation data."""
        result = self.handler.read()
        if not result.success:
            raise ValueError(
                f"Failed to load YOLO annotations: {result.message}"
            )
        return result.data
