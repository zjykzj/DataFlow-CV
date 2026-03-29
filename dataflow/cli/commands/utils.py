"""CLI utility functions."""

from pathlib import Path
from typing import Optional, Tuple
import click
from functools import wraps

from dataflow.util.logging_util import LoggingOperations, VerboseLoggingOperations


def add_common_options(func):
    """Decorator: add common options to subcommands (--verbose only)"""
    @click.option(
        "--verbose",
        is_flag=True,
        help="Enable verbose log output",
    )
    @click.pass_context
    @wraps(func)
    def wrapper(ctx, verbose, *args, **kwargs):
        # Update options in context object
        ctx.obj["verbose"] = verbose
        # Set default strict value (True, strict mode)
        ctx.obj["strict"] = True

        # Reconfigure logging (based on verbose flag)
        if verbose:
            # Use default log directory ./logs
            logger = VerboseLoggingOperations().get_verbose_logger(
                name=ctx.command.name,
                verbose=True,
                log_dir=Path("./logs"),
            )
        else:
            logger = LoggingOperations().get_logger(ctx.command.name)
        ctx.obj["logger"] = logger

        logger.debug(f"Subcommand context updated: verbose={verbose}")
        # Call original function, passing ctx as first argument
        return func(ctx, *args, **kwargs)
    return wrapper


def validate_path_exists(path: Path, name: str = "path") -> Path:
    """Validate if path exists"""
    if not path.exists():
        from dataflow.cli.exceptions import InputError
        raise InputError(f"{name} does not exist: {path}")
    return path


def validate_visualize_params(
    input_path: Path,
    image_dir: Optional[Path],
    output_dir: Optional[Path],
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Validate visualization parameters"""
    input_path = validate_path_exists(input_path, "input path")

    if image_dir:
        image_dir = validate_path_exists(image_dir, "image directory")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    return input_path, image_dir, output_dir


def validate_convert_params(
    source_format: str,
    target_format: str,
    input_path: Path,
    output_path: Path,
    image_dir: Optional[Path],
    class_file: Optional[Path],
) -> Tuple[Path, Path, Optional[Path], Optional[Path]]:
    """Validate conversion parameters"""
    from dataflow.cli.exceptions import InputError

    input_path = validate_path_exists(input_path, "input path")

    # Ensure output directory exists
    if output_path.suffix:  # Is a file
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:  # Is a directory
        output_path.mkdir(parents=True, exist_ok=True)

    # Check required parameters based on conversion direction
    if source_format == "yolo" and target_format == "coco":
        if not image_dir:
            raise InputError("--image-dir is required for YOLO→COCO conversion")
        if not class_file:
            raise InputError("--class-file is required for YOLO→COCO conversion")
    elif source_format == "yolo" and target_format == "labelme":
        if not image_dir:
            raise InputError("--image-dir is required for YOLO→LabelMe conversion")
        if not class_file:
            raise InputError("--class-file is required for YOLO→LabelMe conversion")
    elif source_format == "labelme" and target_format == "coco":
        if not class_file:
            raise InputError("--class-file is required for LabelMe→COCO conversion")
    elif source_format == "labelme" and target_format == "yolo":
        if not class_file:
            raise InputError("--class-file is required for LabelMe→YOLO conversion")
    # For coco→yolo: both optional
    # For coco→labelme: both optional

    if image_dir:
        image_dir = validate_path_exists(image_dir, "image directory")

    if class_file:
        class_file = validate_path_exists(class_file, "class file")

    return input_path, output_path, image_dir, class_file