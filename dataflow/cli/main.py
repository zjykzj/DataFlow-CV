"""CLI entry point for DataFlow-CV."""

import click
from pathlib import Path

from dataflow.util.logging_util import LoggingOperations, VerboseLoggingOperations


@click.group()
@click.version_option()
@click.option(
    "--verbose",
    "-v",
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
@click.pass_context
def cli(ctx, verbose, log_dir, strict):
    """DataFlow-CV命令行工具 - 计算机视觉数据处理工具集"""
    # 初始化上下文对象
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["log_dir"] = log_dir
    ctx.obj["strict"] = strict

    # 配置日志
    if verbose:
        logger = VerboseLoggingOperations().get_verbose_logger(
            name="dataflow.cli",
            verbose=True,
            log_dir=log_dir,
        )
    else:
        logger = LoggingOperations().get_logger("dataflow.cli")

    ctx.obj["logger"] = logger
    logger.debug(f"CLI上下文初始化完成: verbose={verbose}, log_dir={log_dir}, strict={strict}")


# 注册子命令组
from .commands import visualize, convert

cli.add_command(visualize.visualize_group)
cli.add_command(convert.convert_group)

if __name__ == "__main__":
    cli()