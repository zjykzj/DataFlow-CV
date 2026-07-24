"""
Analyse module — dataset introspection and preparation.

Provides dataset statistics, train/test splitting, category filtering,
N-way partitioning, and file sampling for all supported annotation
formats (YOLO, LabelMe, COCO).  Format is auto-detected from the
label path.

Key features:
- **Dataset statistics**: Count total files, total annotations, and
  per-class distribution
- **Train/test split**: Deterministic dataset splitting with configurable
  ratio and seed
- **Category filter**: Keep only specified categories and remap class IDs
- **N-way partition**: Split into N roughly-equal subsets
- **File sampling**: Collect N files randomly or sequentially

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
from .base import (
    AnalysisResult,
    BaseAnalyser,
    CategoryMapping,
    FilterResult,
    PartitionResult,
    RemovedCategory,
    SampleResult,
    SplitResult,
    StatsResult,
)
from .filter import FilterAnalyser
from .partition import PartitionAnalyser
from .sample import SampleAnalyser
from .split import SplitAnalyser
from .stats import StatsAnalyser

__all__ = [
    "BaseAnalyser",
    "AnalysisResult",
    "CategoryMapping",
    "FilterResult",
    "PartitionResult",
    "RemovedCategory",
    "SampleResult",
    "SplitResult",
    "StatsResult",
    "StatsAnalyser",
    "SplitAnalyser",
    "FilterAnalyser",
    "PartitionAnalyser",
    "SampleAnalyser",
    "utils",
]
