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
    """

    total_files: int
    total_annotations: int
    per_class: Dict[str, int]
    format: str
    categories: Dict[int, str] = field(default_factory=dict)


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
        result = AnalysisResult(success=True, data=data)
        result.log_path = None  # set by subclass after analysis
        return result
