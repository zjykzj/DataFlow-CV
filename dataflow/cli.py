# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:41
@File    : cli.py
@Author  : zj
@Description: Command-line interface for DataFlow-CV
"""

import os
import click
from dataflow import __version__
from dataflow.config import Config

# 动态导入模块CLI
try:
    from dataflow.convert.cli import create_convert_group
    CONVERT_AVAILABLE = True
except ImportError as e:
    CONVERT_AVAILABLE = False
    convert_error = str(e)

try:
    from dataflow.visualize.cli import create_visualize_group
    VISUALIZE_AVAILABLE = True
except ImportError as e:
    VISUALIZE_AVAILABLE = False
    visualize_error = str(e)


@click.group(context_settings={'help_option_names': ['-h', '--help']}, invoke_without_command=True)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--overwrite', is_flag=True, help='Overwrite existing files')
@click.version_option(version=__version__, prog_name='DataFlow-CV')
@click.pass_context
def cli(ctx, verbose, overwrite):
    """DataFlow-CV: Computer vision dataset processing tool."""
    # 处理-v单独使用的情况
    if verbose and ctx.invoked_subcommand is None:
        click.echo(f"DataFlow-CV, version {__version__}")
        ctx.exit()

    # 存储配置到上下文
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['overwrite'] = overwrite

    # 更新全局配置
    if verbose:
        Config.VERBOSE = True
    if overwrite:
        Config.OVERWRITE_EXISTING = True

    # 如果没有子命令，显示帮助
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        return

    if verbose:
        click.echo(f"Verbose mode: {'ON' if verbose else 'OFF'}")
        click.echo(f"Overwrite mode: {'ON' if overwrite else 'OFF'}")


# 动态注册模块命令组
if CONVERT_AVAILABLE:
    cli.add_command(create_convert_group(), name='convert')
else:
    @cli.group(name='convert')
    @click.pass_context
    def convert_disabled(ctx):
        """Convert module is not available."""
        click.echo(f"❌ Convert module is disabled: {convert_error}")
        ctx.exit(1)


if VISUALIZE_AVAILABLE:
    cli.add_command(create_visualize_group(), name='visualize')
else:
    @cli.group(name='visualize')
    @click.pass_context
    def visualize_disabled(ctx):
        """Visualize module is not available."""
        click.echo(f"❌ Visualize module is disabled: {visualize_error}")
        ctx.exit(1)


@cli.command()
@click.pass_context
def config(ctx):
    """Show current configuration."""
    click.echo("DataFlow-CV Configuration")
    click.echo("="*50)

    # 显示全局配置
    click.echo("\nGlobal Configuration:")
    global_config_vars = [
        ("YOLO_CLASSES_FILENAME", Config.YOLO_CLASSES_FILENAME),
        ("YOLO_LABELS_DIRNAME", Config.YOLO_LABELS_DIRNAME),
        ("IMAGE_EXTENSIONS", Config.IMAGE_EXTENSIONS),
        ("YOLO_LABEL_EXTENSION", Config.YOLO_LABEL_EXTENSION),
        ("COCO_JSON_EXTENSION", Config.COCO_JSON_EXTENSION),
        ("OVERWRITE_EXISTING", Config.OVERWRITE_EXISTING),
        ("VERBOSE", Config.VERBOSE),
        ("CREATE_DIRS", Config.CREATE_DIRS),
        ("YOLO_NORMALIZE", Config.YOLO_NORMALIZE),
        ("YOLO_SEGMENTATION", Config.YOLO_SEGMENTATION),
    ]

    for name, value in global_config_vars:
        click.echo(f"  {name:30} = {value}")

    # 显示模块配置（如果可用）
    if CONVERT_AVAILABLE:
        from dataflow.convert.config import ConvertConfig
        click.echo("\nConvert Module Configuration:")
        convert_config_vars = [
            ("DEFAULT_SEGMENTATION", ConvertConfig.DEFAULT_SEGMENTATION),
            ("VALIDATE_ANNOTATIONS", ConvertConfig.VALIDATE_ANNOTATIONS),
            ("BATCH_SIZE", ConvertConfig.BATCH_SIZE),
        ]
        for name, value in convert_config_vars:
            click.echo(f"  {name:30} = {value}")

    if VISUALIZE_AVAILABLE:
        from dataflow.visualize.config import VisualizeConfig
        click.echo("\nVisualize Module Configuration:")
        visualize_config_vars = [
            ("DEFAULT_WINDOW_SIZE", VisualizeConfig.DEFAULT_WINDOW_SIZE),
            ("DEFAULT_COLOR_SCHEME", VisualizeConfig.DEFAULT_COLOR_SCHEME),
            ("SHOW_CONFIDENCE", VisualizeConfig.SHOW_CONFIDENCE),
            ("FONT_SCALE", VisualizeConfig.FONT_SCALE),
            ("LINE_THICKNESS", VisualizeConfig.LINE_THICKNESS),
        ]
        for name, value in visualize_config_vars:
            click.echo(f"  {name:30} = {value}")

    # 显示CLI上下文
    click.echo("\nCLI Context:")
    for key, value in ctx.obj.items():
        click.echo(f"  {key}: {value}")


def main():
    """Main entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()