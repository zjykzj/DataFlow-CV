"""
YOLO annotation visualizer.

Visualizes YOLO format annotation files.
Supports both object detection and instance segmentation formats.
"""

from pathlib import Path
from typing import Union

from dataflow.label.yolo_handler import YoloAnnotationHandler

from .base import BaseVisualizer


class YOLOVisualizer(BaseVisualizer):
    """YOLO format visualizer."""

    def __init__(
        self,
        label_dir: Union[str, Path],
        image_dir: Union[str, Path],
        class_file: Union[str, Path],
        verbose: bool = False,  # New: verbose parameter
        **kwargs,
    ):
        """
        Initialize YOLO visualizer.

        Args:
            label_dir: YOLO label directory (contains TXT files)
            image_dir: Image directory
            class_file: Class file path (required)
            verbose: Whether to enable verbose logging (new)
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

    def load_annotations(self):
        """Load YOLO annotation data."""
        result = self.handler.read()
        if not result.success:
            raise ValueError(f"Failed to load YOLO annotations: {result.message}")
        return result.data
