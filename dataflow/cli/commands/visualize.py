"""Visualization commands."""

import click
from pathlib import Path
from typing import Optional
from functools import wraps

from dataflow.cli.commands.utils import validate_visualize_params
from dataflow.cli.exceptions import RuntimeCLIError
from dataflow.util.logging_util import LoggingOperations, VerboseLoggingOperations


def add_visualize_options(func):
    """Decorator: add visualize-specific options (--verbose only)"""
    @click.option(
        "--verbose",
        is_flag=True,
        help="Enable verbose log output and save to logs/ directory",
    )
    @click.pass_context
    @wraps(func)
    def wrapper(ctx, verbose, *args, **kwargs):
        import sys
        print(f"DEBUG wrapper called: ctx={ctx}, verbose={verbose}, args={args}, kwargs={kwargs}", file=sys.stderr)
        # Update options in context object
        ctx.obj["verbose"] = verbose
        # Use default log directory
        log_dir = Path("./logs")
        ctx.obj["log_dir"] = log_dir
        # Strict mode is always True (no option)
        ctx.obj["strict"] = True

        # Reconfigure logging (based on verbose flag)
        if verbose:
            logger = VerboseLoggingOperations().get_verbose_logger(
                name=ctx.command.name,
                verbose=True,
                log_dir=log_dir,
            )
        else:
            logger = LoggingOperations().get_logger(ctx.command.name)
        ctx.obj["logger"] = logger

        logger.debug(f"Subcommand context updated: verbose={verbose}, log_dir={log_dir}")
        # Call original function, passing ctx as first argument
        return func(ctx, *args, **kwargs)
    return wrapper


@click.group(name="visualize")
def visualize_group():
    """Visualization command group - supports visualization of multiple label formats"""
    pass


@visualize_group.command()
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

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"]

    logger.info(f"Starting visualization of YOLO labels: image_dir={image_dir}, label_dir={label_dir}")

    # Parameter validation
    validate_visualize_params(label_dir, image_dir, save)

    # Call existing API
    visualizer = YOLOVisualizer(
        label_dir=label_dir,
        image_dir=image_dir,
        class_file=class_file,
        output_dir=save,
        is_show=False,  # Don't display, only save if output directory provided
        is_save=save is not None,
        strict_mode=strict,
        verbose=verbose,
        logger=logger,
    )
    result = visualizer.visualize()

    if result.success:
        logger.info(f"Visualization completed: processed {result.data.get('processed_images', 0)} images")
    else:
        logger.error(f"Visualization failed: {result.message}")
        raise RuntimeCLIError(f"Visualization failed: {result.message}")


@visualize_group.command()
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

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"]

    logger.info(f"Starting visualization of COCO labels: {coco_file}")

    # Parameter validation
    validate_visualize_params(coco_file, image_dir, save)

    # Call existing API
    visualizer = COCOVisualizer(
        annotation_file=coco_file,
        image_dir=image_dir,
        output_dir=save,
        is_show=False,  # Don't display, only save if output directory provided
        is_save=save is not None,
        strict_mode=strict,
        verbose=verbose,
        logger=logger,
    )
    result = visualizer.visualize()

    if result.success:
        logger.info(f"Visualization completed: processed {result.data.get('processed_images', 0)} images")
    else:
        logger.error(f"Visualization failed: {result.message}")
        raise RuntimeCLIError(f"Visualization failed: {result.message}")


@visualize_group.command()
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

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"]

    logger.info(f"Starting visualization of LabelMe labels: {label_dir}")

    # Parameter validation
    validate_visualize_params(label_dir, image_dir, save)

    # Call existing API
    visualizer = LabelMeVisualizer(
        label_dir=label_dir,
        image_dir=image_dir,
        output_dir=save,
        is_show=False,  # Don't display, only save if output directory provided
        is_save=save is not None,
        strict_mode=strict,
        verbose=verbose,
        logger=logger,
    )
    result = visualizer.visualize()

    if result.success:
        logger.info(f"Visualization completed: processed {result.data.get('processed_images', 0)} images")
    else:
        logger.error(f"Visualization failed: {result.message}")
        raise RuntimeCLIError(f"Visualization failed: {result.message}")