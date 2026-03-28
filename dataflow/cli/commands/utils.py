"""CLI utility functions."""

from pathlib import Path
from typing import Optional, Tuple
import click
from functools import wraps

from dataflow.util.logging_util import LoggingOperations, VerboseLoggingOperations


def add_common_options(func):
    """装饰器：为子命令添加通用选项（--verbose, --log-dir, --strict）"""
    @click.option(
        "--verbose",
        is_flag=True,
        help="启用详细日志输出",
    )
    @click.option(
        "--log-dir",
        type=click.Path(path_type=Path),
        default="./logs",
        help="日志文件保存目录",
    )
    @click.option(
        "--strict",
        is_flag=True,
        default=True,
        help="严格模式（遇错停止）",
    )
    @wraps(func)
    def wrapper(ctx, verbose, log_dir, strict, *args, **kwargs):
        # 更新上下文对象中的选项
        ctx.obj["verbose"] = verbose
        ctx.obj["log_dir"] = log_dir
        ctx.obj["strict"] = strict

        # 重新配置日志（基于verbose标志）
        if verbose:
            logger = VerboseLoggingOperations().get_verbose_logger(
                name=ctx.command.name,
                verbose=True,
                log_dir=log_dir,
            )
        else:
            logger = LoggingOperations().get_logger(ctx.command.name)
        ctx.obj["logger"] = logger

        logger.debug(f"子命令上下文更新: verbose={verbose}, log_dir={log_dir}, strict={strict}")
        return func(ctx, *args, **kwargs)
    return wrapper


def validate_path_exists(path: Path, name: str = "路径") -> Path:
    """验证路径是否存在"""
    if not path.exists():
        from dataflow.cli.exceptions import InputError
        raise InputError(f"{name}不存在: {path}")
    return path


def validate_visualize_params(
    input_path: Path,
    image_dir: Optional[Path],
    output_dir: Optional[Path],
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """验证可视化参数"""
    input_path = validate_path_exists(input_path, "输入路径")

    if image_dir:
        image_dir = validate_path_exists(image_dir, "图像目录")

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
    """验证转换参数"""
    input_path = validate_path_exists(input_path, "输入路径")

    # 确保输出目录存在
    if output_path.suffix:  # 是文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:  # 是目录
        output_path.mkdir(parents=True, exist_ok=True)

    if image_dir:
        image_dir = validate_path_exists(image_dir, "图像目录")

    if class_file:
        class_file = validate_path_exists(class_file, "类别文件")

    return input_path, output_path, image_dir, class_file