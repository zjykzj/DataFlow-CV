"""Visualization commands."""

import click
from pathlib import Path
from typing import Optional
from functools import wraps

from dataflow.cli.commands.utils import FormattedCommand, validate_visualize_params
from dataflow.cli.exceptions import RuntimeCLIError
from dataflow.util.logging import LogConfig


def add_visualize_options(func):
    """Decorator: add visualize-specific options"""
    @click.option(
        "--verbose",
        is_flag=True,
        help="Enable verbose log output and save to logs/ directory",
    )
    @click.option(
        "--display/--no-display",
        default=True,
        help="Show visualization window (--no-display for headless servers)",
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
    def wrapper(ctx, verbose, display, log_dir, *args, **kwargs):
        # Update options in context object
        ctx.obj["verbose"] = verbose
        ctx.obj["is_show"] = display
        ctx.obj["log_dir"] = Path(log_dir)
        # No logger creation — logging is module-owned
        return func(ctx, *args, **kwargs)
    return wrapper


@click.group(name="visualize")
def visualize_group():
    """Visualization command group - supports visualization of multiple label formats"""
    pass


@visualize_group.command(cls=FormattedCommand)
@add_visualize_options
@click.argument("image_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("label_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("class_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--save",
    "-s",
    type=click.Path(path_type=Path),
    help="Directory to save visualization results",
)
def yolo(
    ctx,
    image_dir: Path,
    label_dir: Path,
    class_file: Path,
    save: Optional[Path],
):
    """Visualize YOLO format labels"""
    from dataflow.visualize.yolo_visualizer import YOLOVisualizer

    # Parameter validation
    validate_visualize_params(label_dir, image_dir, save)

    # Build log config
    log_config = LogConfig(
        name=f"visualize.yolo",
        verbose=ctx.obj["verbose"],
        log_dir=ctx.obj["log_dir"],
    )

    # Call existing API
    visualizer = YOLOVisualizer(
        label_dir=label_dir,
        image_dir=image_dir,
        class_file=class_file,
        output_dir=save,
        is_show=ctx.obj["is_show"],
        is_save=save is not None,
        log_config=log_config,
    )
    result = visualizer.visualize()

    # Print terminal output
    if result.log_path:
        click.echo(f"Log saved to: {result.log_path}")

    if result.success:
        click.echo(f"Visualization completed: processed {result.data.get('processed_count', 0)} images")
    else:
        # Use result.message if available, otherwise fall back to errors list
        error_msg = result.message
        if not error_msg and result.errors:
            error_msg = result.errors[0] if result.errors else "Unknown error"
        click.echo(f"Visualization failed: {error_msg}", err=True)
        raise RuntimeCLIError(f"Visualization failed: {error_msg}")


@visualize_group.command(cls=FormattedCommand)
@add_visualize_options
@click.argument("image_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("label_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--save",
    "-s",
    type=click.Path(path_type=Path),
    help="Directory to save visualization results",
)
def labelme(
    ctx,
    image_dir: Path,
    label_dir: Path,
    save: Optional[Path],
):
    """Visualize LabelMe format labels"""
    from dataflow.visualize.labelme_visualizer import LabelMeVisualizer

    # Parameter validation
    validate_visualize_params(label_dir, image_dir, save)

    # Build log config
    log_config = LogConfig(
        name=f"visualize.labelme",
        verbose=ctx.obj["verbose"],
        log_dir=ctx.obj["log_dir"],
    )

    # Call existing API
    visualizer = LabelMeVisualizer(
        label_dir=label_dir,
        image_dir=image_dir,
        output_dir=save,
        is_show=ctx.obj["is_show"],
        is_save=save is not None,
        log_config=log_config,
    )
    result = visualizer.visualize()

    # Print terminal output
    if result.log_path:
        click.echo(f"Log saved to: {result.log_path}")

    if result.success:
        click.echo(f"Visualization completed: processed {result.data.get('processed_count', 0)} images")
    else:
        error_msg = result.message
        if not error_msg and result.errors:
            error_msg = result.errors[0] if result.errors else "Unknown error"
        click.echo(f"Visualization failed: {error_msg}", err=True)
        raise RuntimeCLIError(f"Visualization failed: {error_msg}")


@visualize_group.command(cls=FormattedCommand)
@add_visualize_options
@click.argument("image_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("coco_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--save",
    "-s",
    type=click.Path(path_type=Path),
    help="Directory to save visualization results",
)
def coco(
    ctx,
    image_dir: Path,
    coco_file: Path,
    save: Optional[Path],
):
    """Visualize COCO format labels"""
    from dataflow.visualize.coco_visualizer import COCOVisualizer

    # Parameter validation
    validate_visualize_params(coco_file, image_dir, save)

    # Build log config
    log_config = LogConfig(
        name=f"visualize.coco",
        verbose=ctx.obj["verbose"],
        log_dir=ctx.obj["log_dir"],
    )

    # Call existing API
    visualizer = COCOVisualizer(
        annotation_file=coco_file,
        image_dir=image_dir,
        output_dir=save,
        is_show=ctx.obj["is_show"],
        is_save=save is not None,
        log_config=log_config,
    )
    result = visualizer.visualize()

    # Print terminal output
    if result.log_path:
        click.echo(f"Log saved to: {result.log_path}")

    if result.success:
        click.echo(f"Visualization completed: processed {result.data.get('processed_count', 0)} images")
    else:
        error_msg = result.message
        if not error_msg and result.errors:
            error_msg = result.errors[0] if result.errors else "Unknown error"
        click.echo(f"Visualization failed: {error_msg}", err=True)
        raise RuntimeCLIError(f"Visualization failed: {error_msg}")