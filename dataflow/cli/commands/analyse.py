"""
CLI analyse subcommands for DataFlow-CV.

Provides ``stats`` and ``split`` subcommands under the ``analyse``
command group.
"""

from pathlib import Path

import click

from dataflow.cli.commands.utils import FormattedCommand, add_common_options
from dataflow.cli.exceptions import RuntimeCLIError
from dataflow.util.logging import LogConfig


# ---------------------------------------------------------------------------
# Command group
# ---------------------------------------------------------------------------


@click.group(name="analyse")
def analyse_group():
    """Dataset analysis and preparation commands."""
    pass


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------


def _add_analyse_options(func):
    """Decorator: add analyse-specific options shared by all subcommands."""
    from functools import wraps

    @click.option(
        "-c",
        "--class-file",
        type=click.Path(exists=True, path_type=Path),
        default=None,
        help="Classes.txt file for class name mapping and output ordering",
    )
    @click.option(
        "--image-dir",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        default=None,
        help="Image directory (auto-detected for YOLO if omitted: "
             "looks for sibling 'images/' directory of LABEL_PATH)",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Subcommand: stats
# ---------------------------------------------------------------------------


@analyse_group.command(cls=FormattedCommand)
@add_common_options
@_add_analyse_options
@click.option(
    "--sort-by",
    type=click.Choice(["id", "count"], case_sensitive=False),
    default="id",
    show_default=True,
    help="Sort per-class output by class ID or annotation count "
         "(ignored when --class-file is provided)",
)
@click.option(
    "--descending/--ascending",
    default=False,
    show_default=True,
    help="Sort direction",
)
@click.argument(
    "label_path",
    type=click.Path(exists=True, path_type=Path),
)
def stats(
    ctx: click.Context,
    label_path: Path,
    class_file: Path,
    image_dir: Path,
    sort_by: str,
    descending: bool,
):
    """Compute dataset statistics for annotations at LABEL_PATH.

    LABEL_PATH is a directory (YOLO .txt or LabelMe .json files) or
    a single COCO .json file.  The format is auto-detected.
    """
    from dataflow.analyse import StatsAnalyser

    verbose = ctx.obj["verbose"]
    log_dir = ctx.obj["log_dir"]

    log_config = LogConfig(
        name="analyse.stats",
        verbose=verbose,
        log_dir=log_dir,
    )
    analyser = StatsAnalyser(log_config=log_config)
    result = analyser.analyse(
        label_path,
        class_file=class_file,
        image_dir=image_dir,
        sort_by=sort_by,
        descending=descending,
    )

    if result.success and result.data is not None:
        if result.log_path:
            click.echo(f"\nLog saved to: {result.log_path}")
        for warning in result.warnings:
            click.echo(f"Warning: {warning}", err=True)
    else:
        for warning in result.warnings:
            click.echo(f"Warning: {warning}", err=True)
        error_msg = result.errors[0] if result.errors else "Statistics analysis failed"
        if len(result.errors) > 1:
            error_msg += "\n" + "\n".join(result.errors[1:])
        raise RuntimeCLIError(error_msg)


# ---------------------------------------------------------------------------
# Subcommand: split
# ---------------------------------------------------------------------------


@analyse_group.command(cls=FormattedCommand)
@add_common_options
@click.argument(
    "label_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "output_dir",
    type=click.Path(path_type=Path),
)
@click.option(
    "-r",
    "--ratio",
    type=float,
    default=0.8,
    show_default=True,
    help="Proportion of data for training set",
)
@click.option(
    "-s",
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducible shuffling",
)
@click.option(
    "-c",
    "--class-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Classes.txt file (required for YOLO format, copied to output dirs)",
)
@click.option(
    "--image-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Image directory (auto-detected for YOLO if omitted)",
)
def split(
    ctx: click.Context,
    label_path: Path,
    output_dir: Path,
    ratio: float,
    seed: int,
    class_file: Path,
    image_dir: Path,
):
    """Split dataset at LABEL_PATH into train/val at OUTPUT_DIR.

    LABEL_PATH is a directory (YOLO .txt or LabelMe .json files) or
    a single COCO .json file.  The format is auto-detected.

    For COCO: creates ``train.json`` and ``val.json`` in OUTPUT_DIR.
    For YOLO/LabelMe: creates OUTPUT_DIR/train/ and OUTPUT_DIR/val/.
    """
    from dataflow.analyse import SplitAnalyser

    verbose = ctx.obj["verbose"]
    log_dir = ctx.obj["log_dir"]

    log_config = LogConfig(
        name="analyse.split",
        verbose=verbose,
        log_dir=log_dir,
    )
    analyser = SplitAnalyser(log_config=log_config)
    result = analyser.analyse(
        label_path,
        output_dir,
        ratio=ratio,
        seed=seed,
        class_file=class_file,
        image_dir=image_dir,
    )

    if result.success and result.data is not None:
        if result.log_path:
            click.echo(f"\nLog saved to: {result.log_path}")
        for warning in result.warnings:
            click.echo(f"Warning: {warning}", err=True)
    else:
        for warning in result.warnings:
            click.echo(f"Warning: {warning}", err=True)
        error_msg = result.errors[0] if result.errors else "Dataset split failed"
        if len(result.errors) > 1:
            error_msg += "\n" + "\n".join(result.errors[1:])
        raise RuntimeCLIError(error_msg)
