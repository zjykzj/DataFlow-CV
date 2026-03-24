"""
Base converter abstract class for format conversion.

Defines the interface and common functionality for all format converters.
"""

import datetime
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..label.base import AnnotationResult
from ..label.models import AnnotationFormat, DatasetAnnotations
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
    verbose_log: List[str] = field(default_factory=list)  # New: detailed log entries

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

    def add_verbose_log(self, entry: str):
        """Add detailed log entry (for verbose mode)."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.verbose_log.append(f"[{timestamp}] {entry}")

    def get_summary(self) -> str:
        """Get a summary of the conversion result."""
        if self.success:
            return (
                f"Successfully converted {self.num_images_converted} images "
                f"with {self.num_objects_converted} objects "
                f"from {self.source_format} to {self.target_format}"
            )
        else:
            return f"Conversion failed with {len(self.errors)} errors"

    def get_verbose_summary(self) -> str:
        """Get detailed summary (including verbose log)."""
        if not self.verbose_log:
            return self.get_summary()

        summary = self.get_summary()
        log_entries = "\n".join(f"  {entry}" for entry in self.verbose_log)

        return f"""
{summary}

Detailed processing log:
{'-'*50}
{log_entries}
{'-'*50}
Warnings: {len(self.warnings)}
Errors: {len(self.errors)}
"""


class BaseConverter(ABC):
    """Abstract base class for format converters."""

    def __init__(
        self,
        source_format: str,
        target_format: str,
        strict_mode: bool = True,
        verbose: bool = False,  # New: verbose parameter
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize base converter.

        Args:
            source_format: Source annotation format name
            target_format: Target annotation format name
            strict_mode: Whether to stop on errors (default True)
            verbose: Whether to enable verbose logging mode (new)
            logger: Optional logger instance
        """
        self.source_format = source_format
        self.target_format = target_format
        self.strict_mode = strict_mode
        self.verbose = verbose

        # Configure logger based on verbose
        if verbose and logger is None:
            from ..util.logging_util import VerboseLoggingOperations

            logging_ops = VerboseLoggingOperations()
            self.logger = logging_ops.get_verbose_logger(
                name=f"convert.{source_format}_to_{target_format}", verbose=verbose
            )
            self.progress_logger = logging_ops.create_progress_logger()
        else:
            self.logger = logger or logging.getLogger(__name__)
            self.progress_logger = None

        self.file_ops = FileOperations(logger=self.logger)

        # Conversion statistics
        self.conversion_stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "objects_converted": 0,
            "conversion_errors": 0,
            "start_time": None,
            "end_time": None,
        }

    def convert(self, source_path: str, target_path: str, **kwargs) -> ConversionResult:
        """
        Convert annotations from source format to target format (enhanced with verbose logging).

        Args:
            source_path: Path to source annotations
            target_path: Path for target annotations
            **kwargs: Additional conversion parameters

        Returns:
            ConversionResult containing conversion status and details
        """
        start_time = datetime.datetime.now()
        self.conversion_stats["start_time"] = start_time

        # Log start information
        if self.verbose:
            self.logger.debug(f"Starting conversion: {source_path} -> {target_path}")
            self.logger.debug(f"Conversion parameters: {kwargs}")

        # 1. Validate input parameters
        if not self.validate_inputs(source_path, target_path, kwargs):
            if self.verbose:
                self.logger.error("Input parameter validation failed")
            return self._create_conversion_result(
                success=False,
                source_path=source_path,
                target_path=target_path,
                errors=["Input parameter validation failed"],
            )

        # 2. Use source handler to read data
        source_handler = self.create_source_handler(source_path, kwargs)
        if self.verbose:
            self.logger.debug(
                f"Created source handler: {source_handler.__class__.__name__}"
            )

        read_result = source_handler.read()
        if not read_result.success:
            if self.verbose:
                self.logger.error(f"Failed to read source data: {read_result.errors}")
            return self._create_conversion_result(
                success=False,
                source_path=source_path,
                target_path=target_path,
                errors=read_result.errors,
            )

        # Record read results
        annotations = read_result.data
        if self.verbose:
            self.logger.info(f"Read annotations for {annotations.num_images} images")
            self.logger.debug(f"Category count: {len(annotations.categories)}")

        # 3. Convert data (format-specific conversion)
        if self.verbose:
            self.logger.debug("Starting format-specific conversion")

        converted_annotations = self.convert_annotations(annotations, kwargs)
        self.conversion_stats["objects_converted"] = converted_annotations.num_objects

        if self.verbose:
            self.logger.debug(
                f"Conversion completed, object count: {converted_annotations.num_objects}"
            )

        # 4. Use target handler to write data
        target_handler = self.create_target_handler(target_path, kwargs)
        if self.verbose:
            self.logger.debug(
                f"Created target handler: {target_handler.__class__.__name__}"
            )

        write_result = target_handler.write(converted_annotations, target_path)

        # 5. Create result
        result = self._create_conversion_result(
            success=write_result.success,
            source_path=source_path,
            target_path=target_path,
            annotations=converted_annotations,
            write_result=write_result,
        )

        # Add verbose log entries
        if self.verbose:
            result.add_verbose_log(f"Source format: {self.source_format}")
            result.add_verbose_log(f"Target format: {self.target_format}")
            result.add_verbose_log(f"Source path: {source_path}")
            result.add_verbose_log(f"Target path: {target_path}")
            result.add_verbose_log(f"Images processed: {annotations.num_images}")
            result.add_verbose_log(
                f"Objects converted: {converted_annotations.num_objects}"
            )

            if write_result.warnings:
                for warning in write_result.warnings:
                    result.add_verbose_log(f"Warning: {warning}")

            # Record conversion statistics
            self.conversion_stats["end_time"] = datetime.datetime.now()
            duration = (
                self.conversion_stats["end_time"] - self.conversion_stats["start_time"]
            )
            result.add_verbose_log(
                f"Total duration: {duration.total_seconds():.2f} seconds"
            )

        return result

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
            self.logger.error(
                f"Cannot create target directory {target_path_obj.parent}: {e}"
            )
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

    def convert_annotations(
        self, source_annotations: DatasetAnnotations, kwargs: Dict
    ) -> DatasetAnnotations:
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

    def _create_conversion_result(
        self,
        success: bool,
        source_path: str,
        target_path: str,
        annotations: Optional[DatasetAnnotations] = None,
        write_result: Optional[AnnotationResult] = None,
        errors: Optional[List[str]] = None,
    ) -> ConversionResult:
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
            target_path=target_path,
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
