"""
Base converter abstract class for format conversion.

Defines the interface and common functionality for all format converters.
"""

import datetime
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..label.base import AnnotationResult, BaseAnnotationHandler
from ..label.models import (AnnotationFormat, DatasetAnnotations,
                              ImageAnnotation)
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
    log_file_path: Optional[str] = None  # Log file path when verbose=True

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
            self.logger, self.log_file_path = logging_ops.get_verbose_logger(
                name=f"convert.{source_format}_to_{target_format}", verbose=verbose
            )
        else:
            self.logger = logger or logging.getLogger(__name__)
            self.log_file_path = None

        self.file_ops = FileOperations(logger=self.logger)

        # Initialize conversion stats for batch convert()
        self.conversion_stats: Dict[str, Any] = {}

        # Source annotations retained for target handler creation
        # (category extraction, image copying). Cleared in try/finally.
        self._source_annotations_for_target: Optional[DatasetAnnotations] = None

    def _ensure_categories_for_streaming(
        self,
        source_handler: BaseAnnotationHandler,
        source_path: str,
        kwargs: Dict,
    ) -> None:
        """Ensure categories are available before streaming iteration.

        Called by ``stream_convert()`` before ``create_target_handler()``.
        Subclasses may override to pre-load categories from source files
        that donʼt expose them until ``read()`` / ``iter_images()`` runs.

        Default: try ``source_handler.categories`` if available and it is
        a genuine dict (not a mock).
        """
        cats = getattr(source_handler, "categories", None)
        if isinstance(cats, dict) and cats:
            self._source_annotations_for_target = DatasetAnnotations(
                format=AnnotationFormat.UNKNOWN,
                categories=cats.copy(),
            )

    @abstractmethod
    def _convert_single_image(
        self, image_ann: ImageAnnotation, **kwargs
    ) -> ImageAnnotation:
        """Convert a single ImageAnnotation from source to target format.

        Operates on one image at a time for the streaming pipeline.
        The coordinate transformation must match the batch
        ``convert_annotations()`` for the same direction.

        Args:
            image_ann: Single ImageAnnotation with source-native coordinates.
            **kwargs: Additional conversion parameters.

        Returns:
            New ImageAnnotation with target-native coordinates.
        """
        pass

    def stream_convert(
        self, source_path: str, target_path: str, **kwargs
    ) -> ConversionResult:
        """Convert annotations using streaming (per-image) pipeline.

        Sources images one at a time via ``handler.iter_images()``, converts
        each via ``_convert_single_image()``, and writes immediately via
        ``target_handler.write_one()``.

        Applicable when the target format supports per-file output
        (YOLO .txt, LabelMe .json). Not applicable for COCO target.

        Args:
            source_path: Path to source annotations.
            target_path: Path to target output.
            **kwargs: Conversion parameters (class_file, image_dir, etc.).

        Returns:
            ConversionResult with conversion statistics.
        """
        start_time = datetime.datetime.now()
        result = ConversionResult(
            success=False,
            source_format=self.source_format,
            target_format=self.target_format,
            source_path=source_path,
            target_path=target_path,
        )

        # Store source_path for subclasses that need it in
        # create_target_handler() (e.g., image copying in LabelMe→YOLO)
        self._source_path = source_path

        num_images = 0
        num_objects = 0

        try:
            # 1. Validate inputs
            if not self.validate_inputs(source_path, target_path, kwargs):
                result.add_error("Input validation failed")
                return result

            # 2. Create source handler
            source_handler = self.create_source_handler(source_path, kwargs)

            # 3. Extract categories from source (needed for target handler)
            self._ensure_categories_for_streaming(
                source_handler, source_path, kwargs
            )

            # 4. Create target handler
            target_handler = self.create_target_handler(target_path, kwargs)

            # 5. Stream: iterate, convert, write per image
            write_dir = getattr(
                target_handler, "label_dir",
                Path(target_path)
            )

            for image_ann in source_handler.iter_images():
                target_ann = self._convert_single_image(
                    image_ann, **kwargs
                )
                write_result = target_handler.write_one(
                    target_ann, write_dir
                )
                if not write_result.success:
                    err = (
                        f"Failed to write {target_ann.image_id}: "
                        f"{write_result.message}"
                    )
                    if self.strict_mode:
                        result.add_error(err)
                        return result
                    else:
                        result.add_warning(err)
                        continue

                num_images += 1
                num_objects += len(target_ann.objects)

            result.success = True
            result.num_images_converted = num_images
            result.num_objects_converted = num_objects

            duration = (datetime.datetime.now() - start_time).total_seconds()
            result.add_metadata("duration_seconds", f"{duration:.2f}")

        except ValueError as e:
            result.add_error(str(e))
            result.num_images_converted = num_images
            result.num_objects_converted = num_objects
        except Exception as e:
            result.add_error(f"Unexpected error: {e}")
            if self.verbose:
                self.logger.exception("Streaming conversion failed")

        return result

    def convert(self, source_path: str, target_path: str, **kwargs) -> ConversionResult:
        """Convert annotations from source format to target format.

        Auto-dispatches to the correct pipeline based on target format:
        - COCO target (single JSON) → batch pipeline (``_batch_convert()``)
        - YOLO / LabelMe target (per-file output) → streaming pipeline
          (``stream_convert()``)

        Subclasses should NOT override this method. Instead, override the
        abstract hooks: ``create_source_handler()``, ``create_target_handler()``,
        ``_convert_single_image()``, ``convert_annotations()``, and optionally
        ``_post_batch_convert()``.

        Args:
            source_path: Path to source annotations.
            target_path: Path for target annotations.
            **kwargs: Additional conversion parameters.

        Returns:
            ConversionResult containing conversion status and details.
        """
        if self.target_format == "coco":
            return self._batch_convert(source_path, target_path, **kwargs)
        else:
            return self.stream_convert(source_path, target_path, **kwargs)

    def _batch_convert(
        self, source_path: str, target_path: str, **kwargs
    ) -> ConversionResult:
        """Batch pipeline: read ALL → convert ALL → write ALL.

        Used when the target format is a single file (COCO JSON).
        Subclasses should NOT override this — override the abstract
        hooks and the optional ``_post_batch_convert()`` hook instead.
        """
        self._source_annotations_for_target = None

        # 1. Validate inputs
        if not self.validate_inputs(source_path, target_path, kwargs):
            return self._create_conversion_result(
                success=False,
                source_path=source_path,
                target_path=target_path,
                errors=["Input validation failed"],
                log_file_path=self.log_file_path,
            )

        # 2. Read data using source handler
        source_handler = self.create_source_handler(source_path, kwargs)
        read_result = source_handler.read()
        if not read_result.success:
            return self._create_conversion_result(
                success=False,
                source_path=source_path,
                target_path=target_path,
                errors=read_result.errors,
                log_file_path=self.log_file_path,
            )

        # 3. Convert data
        annotations = read_result.data
        if self.verbose:
            self.logger.info(
                f"Read annotations for {annotations.num_images} images"
            )
            self.logger.debug(f"Category count: {len(annotations.categories)}")

        converted_annotations = self.convert_annotations(annotations, kwargs)

        if self.verbose:
            self.logger.debug(
                f"Conversion completed, object count: "
                f"{converted_annotations.num_objects}"
            )

        # 4. Write data (with state cleanup guarantee)
        self._source_annotations_for_target = converted_annotations
        try:
            target_handler = self.create_target_handler(target_path, kwargs)
            write_result = target_handler.write(
                converted_annotations, target_path
            )
        finally:
            self._source_annotations_for_target = None

        # 5. Build result
        result = self._create_conversion_result(
            success=write_result.success,
            source_path=source_path,
            target_path=target_path,
            annotations=converted_annotations,
            write_result=write_result,
            log_file_path=self.log_file_path,
        )

        if self.verbose:
            result.add_verbose_log(
                f"Images processed: {annotations.num_images}"
            )
            result.add_verbose_log(
                f"Objects converted: {converted_annotations.num_objects}"
            )
            if write_result.errors:
                for error in write_result.errors:
                    result.add_verbose_log(f"Error: {error}")

        # 6. Post-processing hook (e.g., RLE warnings)
        self._post_batch_convert(result, source_handler, kwargs)

        return result

    def _post_batch_convert(
        self,
        result: ConversionResult,
        source_handler: BaseAnnotationHandler,
        kwargs: Dict,
    ) -> None:
        """Optional post-processing hook for batch conversions.

        Called after a successful batch write. Subclasses override this
        to add format-specific warnings or metadata (e.g., RLE accuracy
        warnings).

        Default: no-op.
        """

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
        """Convert annotation data (format-specific transformation).

        Subclasses MUST override this to implement the coordinate
        transformation for their conversion direction.  There is no
        safe pass-through default because coordinate semantics differ
        between formats.

        The canonical implementation delegates to
        ``_convert_single_image()`` per image:

        .. code-block:: python

            target = DatasetAnnotations(format=target_format)
            target.categories = source_annotations.categories.copy()
            for img in source_annotations.images:
                target.add_image(self._convert_single_image(img, **kwargs))
            return target

        Args:
            source_annotations: Annotations in source-native coordinates.
            kwargs: Additional conversion parameters.

        Returns:
            Converted DatasetAnnotations in target-native coordinates.

        Raises:
            NotImplementedError: Always — subclasses must implement.
        """
        raise NotImplementedError(
            "Subclass must implement convert_annotations() "
            "for this conversion direction"
        )

    def _create_conversion_result(
        self,
        success: bool,
        source_path: str,
        target_path: str,
        annotations: Optional[DatasetAnnotations] = None,
        write_result: Optional[AnnotationResult] = None,
        errors: Optional[List[str]] = None,
        log_file_path: Optional[str] = None,
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
            log_file_path: Log file path for verbose mode (optional)

        Returns:
            ConversionResult instance
        """
        result = ConversionResult(
            success=success,
            source_format=self.source_format,
            target_format=self.target_format,
            source_path=source_path,
            target_path=target_path,
            log_file_path=log_file_path,
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
        from dataflow.util.logging_util import logging_error_or_raise

        logging_error_or_raise(message, self.logger, self.strict_mode)
