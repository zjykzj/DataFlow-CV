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


class FormattedCommand(click.Command):
    """自定义Command类，提供格式化的Arguments显示"""

    def format_help(self, ctx, formatter):
        """重写帮助输出格式"""
        # 写入用法
        self.format_usage(ctx, formatter)

        # 写入命令描述
        if self.help:
            formatter.write_paragraph()
            with formatter.indentation():
                formatter.write_text(self.help)

        # 写入Arguments（自定义格式）
        self._format_arguments(ctx, formatter)

        # 写入Options
        self.format_options(ctx, formatter)

        # 写入epilog
        if self.epilog:
            formatter.write_paragraph()
            formatter.write_text(self.epilog)

    def _format_arguments(self, ctx, formatter):
        """格式化Arguments部分，模仿Options的格式"""
        args = [param for param in self.params
                if isinstance(param, click.Argument) and param.expose_value]
        if not args:
            return

        with formatter.section("Arguments"):
            # 创建参数名和帮助文本的列表，用于formatter.write_dl
            # write_dl会自动对齐，与Options使用相同的机制
            rows = []
            for param in args:
                param_name = param.make_metavar()
                help_text = self._get_argument_help(param.name) if hasattr(param, 'name') else ""
                rows.append((param_name, help_text))

            # 使用write_dl获得与Options一致的对齐效果
            formatter.write_dl(rows)

    def _get_argument_help(self, param_name):
        """根据参数名获取帮助文本"""
        # 参数名到帮助文本的映射
        help_map = {
            "image_dir": "Image file directory (for obtaining image dimensions)",
            "label_dir": "YOLO label directory",
            "class_file": "Class file path",
            "output_file": "Output COCO JSON file path",
            "output_dir": "Output directory (will contain classes.txt and labels/)",
            "output_path": "Output directory (will contain classes.txt and labels/)",
            "labelme_dir": "LabelMe annotation directory",
            "input_path": "Input COCO JSON annotation file",
        }
        return help_map.get(param_name, "")