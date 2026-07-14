"""
Base classes and data models for the Analyse module.

Provides shared logging infrastructure (``BaseAnalyser``) and result
containers (``AnalysisResult``, ``StatsResult``, ``SplitResult``).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Result data models
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """Top-level return type shared by all analysers.

    Attributes:
        success: Whether the analysis completed successfully.
        data: Module-specific result data (``StatsResult`` or ``SplitResult``).
        errors: Accumulated error messages (non-empty if ``success=False``).
        warnings: Accumulated non-fatal warning messages.
        log_path: Log file path (verbose mode only, ``None`` otherwise).
    """

    success: bool = True
    data: Optional[Any] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    log_path: Optional[str] = None

    def add_error(self, error: str) -> None:
        """Add an error and mark the result as failed."""
        self.errors.append(error)
        self.success = False

    def add_warning(self, warning: str) -> None:
        """Add a non-fatal warning."""
        self.warnings.append(warning)


@dataclass
class StatsResult:
    """Container for dataset statistics.

    Attributes:
        total_files: Number of label files (or images for COCO).
        total_annotations: Total annotation objects across all images.
        per_class: ``{class_name: count}``, ordered by class_file or
            count descending.
        format: Detected format (``"yolo"`` | ``"labelme"`` | ``"coco"``).
        categories: ``{class_id: class_name}`` mapping.
        source_paths: Paths that contributed to this result
            (single element for single-path, multiple for multi-path).
    """

    total_files: int
    total_annotations: int
    per_class: Dict[str, int]
    format: str
    categories: Dict[int, str] = field(default_factory=dict)
    source_paths: List[Path] = field(default_factory=list)


@dataclass
class SplitResult:
    """Container for train/test split results.

    Attributes:
        train_count: Number of images in the training set.
        val_count: Number of images in the validation set.
        train_dir: Path to the train output directory.
        val_dir: Path to the validation output directory.
        ratio: Train ratio used (e.g., 0.8).
        seed: Random seed used for reproducibility.
        format: Detected format (``"yolo"`` | ``"labelme"`` | ``"coco"``).
    """

    train_count: int
    val_count: int
    train_dir: Path
    val_dir: Path
    ratio: float
    seed: int
    format: str


@dataclass
class CategoryMapping:
    """A category that was kept during filtering.

    Attributes:
        new_id: New class ID (line index in the target classes.txt).
        old_id: Original class ID in the source dataset.
        name: Class name.
    """

    new_id: int
    old_id: int
    name: str


@dataclass
class RemovedCategory:
    """A category that was removed during filtering.

    Attributes:
        old_id: Original class ID in the source dataset.
        name: Class name.
    """

    old_id: int
    name: str


@dataclass
class FilterResult:
    """Container for category-based annotation filtering results.

    Attributes:
        total_files: Total label files processed.
        total_files_with_annotations: Files that still have annotations
            after filtering.
        total_annotations_before: Annotation count before filtering.
        total_annotations_after: Annotation count after filtering.
        kept_categories: Categories retained, in new class file order.
        removed_categories: Categories removed during filtering.
        missing_categories: Categories in new class file but not found
            in the source data.
        output_dir: Output directory.
        format: Detected format (``"yolo"`` | ``"labelme"`` | ``"coco"``).
    """

    total_files: int
    total_files_with_annotations: int
    total_annotations_before: int
    total_annotations_after: int
    kept_categories: List[CategoryMapping] = field(default_factory=list)
    removed_categories: List[RemovedCategory] = field(default_factory=list)
    missing_categories: List[str] = field(default_factory=list)
    output_dir: Path = field(default_factory=Path)
    format: str = ""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseAnalyser:
    """Shared logging infrastructure for dataset analysers.

    Concrete analysers (``StatsAnalyser``, ``SplitAnalyser``) extend this
    class to inherit logging helpers.  Analyse operations are read-only —
    errors are logged and accumulated in the result, never raised.

    Constructor follows the project-wide ``LogConfig`` → ``LogManager``
    pattern (see ``dataflow/evaluate/base.py``).
    """

    def __init__(self, log_config: Optional[Any] = None):
        """Initialise the analyser.

        Args:
            log_config: Optional ``LogConfig`` instance.  If ``None``, a
                default ``LogConfig(name="analyse")`` is used.
        """
        from ..util.logging import LogConfig, LogManager

        if log_config is None:
            log_config = LogConfig(name="analyse")
        self._log_manager = LogManager(log_config)
        self.logger = self._log_manager.logger

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_info(self, message: str) -> None:
        """Log an informational message."""
        self.logger.info(message)

    def _log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def _log_error(self, message: str) -> None:
        """Log an error message.

        Unlike Convert (which raises in strict mode) and Evaluate (which
        always raises), Analyse is a read-only operation — errors are
        accumulated in ``AnalysisResult.errors`` rather than raised.
        """
        self.logger.error(message)

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_result(data: Any = None) -> AnalysisResult:
        """Create a new ``AnalysisResult`` with default success state.

        Args:
            data: Optional result data (``StatsResult`` or ``SplitResult``).

        Returns:
            A fresh ``AnalysisResult`` instance.
        """
        return AnalysisResult(success=True, data=data)
