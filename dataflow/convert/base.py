"""
Base converter abstract class for format conversion.

Defines the interface and common functionality for all format converters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from ..label.base import AnnotationResult
from ..label.models import DatasetAnnotations, AnnotationFormat
from ..util.file_util import FileOperations


@dataclass
class ConversionResult:
    """Result of a format conversion operation."""

    success: bool
    source_format: str
    target_format: str
    source_path: str
    target_path: str
    num_images_converted: int = 0
    num_objects_converted: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)

    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.success = False

    def add_metadata(self, key: str, value: Any):
        """Add metadata key-value pair."""
        self.metadata[key] = value

    def get_summary(self) -> str:
        """Get a summary of the conversion result."""
        if self.success:
            return (f"Successfully converted {self.num_images_converted} images "
                    f"with {self.num_objects_converted} objects "
                    f"from {self.source_format} to {self.target_format}")
        else:
            return f"Conversion failed with {len(self.errors)} errors"


class BaseConverter(ABC):
    """Abstract base class for format converters."""

    def __init__(self,
                 source_format: str,
                 target_format: str,
                 strict_mode: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize base converter.

        Args:
            source_format: Source annotation format name
            target_format: Target annotation format name
            strict_mode: Whether to stop on errors (default True)
            logger: Optional logger instance
        """
        self.source_format = source_format
        self.target_format = target_format
        self.strict_mode = strict_mode
        self.logger = logger or logging.getLogger(__name__)
        self.file_ops = FileOperations(logger=self.logger)

    @abstractmethod
    def convert(self, source_path: str, target_path: str, **kwargs) -> ConversionResult:
        """
        Convert annotations from source format to target format.

        Args:
            source_path: Path to source annotations
            target_path: Path for target annotations
            **kwargs: Additional conversion parameters

        Returns:
            ConversionResult containing conversion status and details
        """
        pass

    def validate_inputs(self, source_path: str, target_path: str, kwargs: Dict) -> bool:
        """
        Validate conversion input parameters.

        Args:
            source_path: Path to source annotations
            target_path: Path for target annotations
            kwargs: Additional conversion parameters

        Returns:
            True if inputs are valid, False otherwise
        """
        # Check source path exists
        source_path_obj = Path(source_path)
        if not source_path_obj.exists():
            self.logger.error(f"Source path does not exist: {source_path}")
            return False

        # Check if we can create target directory
        target_path_obj = Path(target_path)
        try:
            target_path_obj.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Cannot create target directory {target_path_obj.parent}: {e}")
            return False

        # Additional format-specific validation should be implemented in subclasses
        return True

    @abstractmethod
    def create_source_handler(self, source_path: str, kwargs: Dict) -> Any:
        """
        Create source annotation handler.

        Args:
            source_path: Path to source annotations
            kwargs: Additional conversion parameters

        Returns:
            BaseAnnotationHandler subclass instance
        """
        pass

    @abstractmethod
    def create_target_handler(self, target_path: str, kwargs: Dict) -> Any:
        """
        Create target annotation handler.

        Args:
            target_path: Path for target annotations
            kwargs: Additional conversion parameters

        Returns:
            BaseAnnotationHandler subclass instance
        """
        pass

    def convert_annotations(self,
                           source_annotations: DatasetAnnotations,
                           kwargs: Dict) -> DatasetAnnotations:
        """
        Convert annotation data (format-specific conversion).

        Args:
            source_annotations: Annotations read from source format
            kwargs: Additional conversion parameters

        Returns:
            Converted DatasetAnnotations ready for writing to target format
        """
        # Default implementation: return as-is (assuming categories and data are already correct)
        # Subclasses should override for format-specific conversions like category mapping
        return source_annotations

    def _create_conversion_result(self,
                                 success: bool,
                                 source_path: str,
                                 target_path: str,
                                 annotations: Optional[DatasetAnnotations] = None,
                                 write_result: Optional[AnnotationResult] = None,
                                 errors: Optional[List[str]] = None) -> ConversionResult:
        """
        Create a ConversionResult instance with appropriate data.

        Args:
            success: Whether conversion was successful
            source_path: Source annotation path
            target_path: Target annotation path
            annotations: Converted annotations (optional)
            write_result: Result from handler.write() (optional)
            errors: List of error messages (optional)

        Returns:
            ConversionResult instance
        """
        result = ConversionResult(
            success=success,
            source_format=self.source_format,
            target_format=self.target_format,
            source_path=source_path,
            target_path=target_path
        )

        if errors:
            for error in errors:
                result.add_error(error)

        if annotations:
            result.num_images_converted = annotations.num_images
            result.num_objects_converted = annotations.num_objects

        if write_result and write_result.errors:
            for error in write_result.errors:
                result.add_error(error)

        return result

    def _log_info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def _log_warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def _log_error(self, message: str):
        """Log error message, raise exception in strict mode."""
        self.logger.error(message)
        if self.strict_mode:
            raise ValueError(message)