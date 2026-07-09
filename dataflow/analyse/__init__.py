"""
Analyse module — dataset introspection and preparation.

Provides dataset statistics and train/test splitting for all supported
annotation formats (YOLO, LabelMe, COCO).  Format is auto-detected from
the label path.

Key features:
- **Dataset statistics**: Count total files, total annotations, and
  per-class distribution
- **Train/test split**: Deterministic dataset splitting with configurable
  ratio and seed

Example::

    from dataflow.analyse import StatsAnalyser

    analyser = StatsAnalyser()
    result = analyser.analyse(
        label_path=Path("yolo_labels/"),
        class_file=Path("classes.txt"),
    )
    if result.success:
        stats = result.data
        print(f"Total: {stats.total_annotations} annotations")
"""

from . import utils
from .base import AnalysisResult, BaseAnalyser, SplitResult, StatsResult
from .split import SplitAnalyser
from .stats import StatsAnalyser

__all__ = [
    "BaseAnalyser",
    "AnalysisResult",
    "StatsResult",
    "SplitResult",
    "StatsAnalyser",
    "SplitAnalyser",
    "utils",
]
