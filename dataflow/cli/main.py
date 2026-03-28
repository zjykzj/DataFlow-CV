"""CLI entry point for DataFlow-CV."""

import click
from pathlib import Path

from dataflow.util.logging_util import LoggingOperations, VerboseLoggingOperations


def print_version(ctx, param, value):
    """回调函数：显示版本信息"""
    if not value or ctx.resilient_parsing:
        return
    from dataflow import __version__
    click.echo(f"dataflow-cv version {__version__}")
    ctx.exit()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--version",
    "-v",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=print_version,
    help="显示版本信息",
)
@click.pass_context
def cli(ctx):
    """DataFlow-CV命令行工具 - 计算机视觉数据处理工具集"""
    # 初始化上下文对象（默认值）
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = False
    ctx.obj["log_dir"] = Path("./logs")
    ctx.obj["strict"] = True

    # 配置默认日志（非详细模式）
    logger = LoggingOperations().get_logger("dataflow.cli")
    ctx.obj["logger"] = logger
    logger.debug("CLI上下文初始化完成")


# 注册子命令组
from .commands import visualize, convert

cli.add_command(visualize.visualize_group)
cli.add_command(convert.convert_group)

if __name__ == "__main__":
    cli()