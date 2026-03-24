"""
COCO annotation visualizer.

Visualizes COCO format annotation files.
Supports both polygon and RLE segmentation formats.
"""

from pathlib import Path
from typing import Union

from dataflow.label.coco_handler import CocoAnnotationHandler

from .base import BaseVisualizer


class COCOVisualizer(BaseVisualizer):
    """COCO format visualizer."""

    def __init__(
        self,
        annotation_file: Union[str, Path],
        image_dir: Union[str, Path],
        verbose: bool = False,  # New: verbose parameter
        **kwargs,
    ):
        """
        Initialize COCO visualizer.

        Args:
            annotation_file: COCO annotation file path (JSON)
            image_dir: Image directory
            verbose: Whether to enable verbose logging (new)
            **kwargs: Additional arguments for BaseVisualizer
        """
        super().__init__(annotation_file, image_dir, verbose=verbose, **kwargs)
        self.annotation_file = Path(annotation_file)
        self.handler = CocoAnnotationHandler(
            annotation_file=str(annotation_file),
            strict_mode=self.strict_mode,
            logger=self.logger,
        )

        if verbose:
            self.logger.debug(
                f"COCO visualizer initialization complete, annotation file: {annotation_file}"
            )

    def load_annotations(self):
        """Load COCO annotation data."""
        result = self.handler.read()
        if not result.success:
            raise ValueError(f"Failed to load COCO annotations: {result.message}")
        return result.data
