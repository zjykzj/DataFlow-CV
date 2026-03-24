"""
LabelMe annotation visualizer.

Visualizes LabelMe JSON annotation format.
"""

from pathlib import Path
from typing import Optional, Union

from dataflow.label.labelme_handler import LabelMeAnnotationHandler

from .base import BaseVisualizer


class LabelMeVisualizer(BaseVisualizer):
    """LabelMe format visualizer."""

    def __init__(
        self,
        label_dir: Union[str, Path],
        image_dir: Union[str, Path],
        class_file: Optional[Union[str, Path]] = None,
        verbose: bool = False,  # New: verbose parameter
        **kwargs,
    ):
        """
        Initialize LabelMe visualizer.

        Args:
            label_dir: LabelMe label directory (contains JSON files)
            image_dir: Image directory
            class_file: Optional class file path
            verbose: Whether to enable verbose logging (new)
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

    def load_annotations(self):
        """Load LabelMe annotation data."""
        result = self.handler.read()
        if not result.success:
            raise ValueError(f"Failed to load LabelMe annotations: {result.message}")
        return result.data
