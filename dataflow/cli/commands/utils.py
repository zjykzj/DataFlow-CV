"""CLI utility functions."""

import inspect
from pathlib import Path
from typing import Optional, Tuple
import click
from functools import wraps

# Click 8.2+ requires ctx in make_metavar(); older versions don't accept it.
# We detect at import time so the code works with both.
_MAKE_METAVAR_NEEDS_CTX = (
    "ctx" in inspect.signature(click.Argument.make_metavar).parameters
)


def add_common_options(func):
    """Decorator: add common options to subcommands"""
    @click.option(
        "--verbose",
        is_flag=True,
        help="Enable verbose log output",
    )
    @click.option(
        "--no-strict",
        is_flag=True,
        help="Disable strict mode (skip invalid annotations instead of aborting)",
    )
    @click.option(
        "--log-dir",
        type=click.Path(path_type=Path),
        default="./logs",
        show_default=True,
        help="Log file output directory",
    )
    @click.pass_context
    @wraps(func)
    def wrapper(ctx, verbose, no_strict, log_dir, *args, **kwargs):
        # Update options in context object
        ctx.obj["verbose"] = verbose
        # strict_mode default True; --no-strict sets it to False
        ctx.obj["strict"] = not no_strict
        ctx.obj["log_dir"] = Path(log_dir)
        # No logger creation — logging is module-owned
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

    # Ensure output directory exists.
    # If the path already exists and is a directory, treat as directory.
    # Otherwise use the suffix heuristic: no suffix → directory,
    # has suffix → file (create parent dirs).
    if output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
    elif output_path.suffix:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    # Check required parameters based on conversion direction
    if source_format == "yolo" and target_format == "coco":
        if not image_dir:
            raise InputError("IMAGE_DIR is required for YOLO→COCO conversion")
        if not class_file:
            raise InputError("CLASS_FILE is required for YOLO→COCO conversion")
    elif source_format == "yolo" and target_format == "labelme":
        if not image_dir:
            raise InputError("IMAGE_DIR is required for YOLO→LabelMe conversion")
        if not class_file:
            raise InputError("CLASS_FILE is required for YOLO→LabelMe conversion")
    elif source_format == "labelme" and target_format == "coco":
        if not class_file:
            raise InputError("CLASS_FILE is required for LabelMe→COCO conversion")
    elif source_format == "labelme" and target_format == "yolo":
        if not class_file:
            raise InputError("CLASS_FILE is required for LabelMe→YOLO conversion")
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

        # 写入命令描述（使用 Click 标准格式，与 main CLI 对齐）
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
                param_name = (
                    param.make_metavar(ctx)
                    if _MAKE_METAVAR_NEEDS_CTX
                    else param.make_metavar()
                )
                help_text = self._get_argument_help(param.name) if hasattr(param, 'name') else ""
                rows.append((param_name, help_text))

            # 使用write_dl获得与Options一致的对齐效果
            formatter.write_dl(rows)

    def _get_argument_help(self, param_name):
        """Get help text by parameter name."""
        help_map = {
            "image_dir": "Image file directory (for obtaining image dimensions)",
            "label_dir": "YOLO label directory",
            "label_path": "Path to labels — directory (YOLO/LabelMe) or COCO .json file",
            "label_paths": "One or more paths to labels",
            "class_file": "Class file path (classes.txt)",
            "output_file": "Output COCO JSON file path",
            "output_dir": "Output directory for converted annotations",
            "output_path": "Output directory for converted annotations",
            "labelme_dir": "LabelMe annotation directory",
            "input_path": "Input COCO JSON annotation file",
            "gt_json": "COCO format Ground Truth JSON file",
            "dt_json": "COCO format Detection/Prediction JSON file",
            "coco_file": "COCO JSON annotation file",
            "annotation_file": "COCO JSON annotation file",
            "original_class_file": "Source classes.txt (all categories in dataset)",
            "new_class_file": "Target classes.txt (categories to keep, new order/IDs)",
        }
        return help_map.get(param_name, "")