"""
CLI analyse subcommands for DataFlow-CV.

Provides ``stats``, ``split``, and ``filter`` subcommands under the
``analyse`` command group.
"""

from pathlib import Path

import click

from dataflow.cli.commands.utils import FormattedCommand
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
    """Decorator: add analyse-specific options shared by all subcommands.

    Includes --verbose, --log-dir, --class-file, --image-dir.
    Does NOT include --no-strict (analyse is always read-only / non-strict).
    """
    from functools import wraps

    @click.option(
        "--verbose",
        is_flag=True,
        help="Enable verbose log output",
    )
    @click.option(
        "--log-dir",
        type=click.Path(path_type=Path),
        default="./logs",
        show_default=True,
        help="Log file output directory",
    )
    @click.option(
        "-c",
        "--class-file",
        type=click.Path(exists=True, path_type=Path),
        default=None,
        help="Classes.txt for name mapping and output ordering",
    )
    @click.option(
        "--image-dir",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        default=None,
        help="Image directory for YOLO format (auto-detected if omitted)",
    )
    @click.pass_context
    @wraps(func)
    def wrapper(ctx, verbose, log_dir, class_file, image_dir, *args, **kwargs):
        ctx.obj["verbose"] = verbose
        ctx.obj["log_dir"] = Path(log_dir)
        return func(ctx, class_file=class_file, image_dir=image_dir, *args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Subcommand: stats
# ---------------------------------------------------------------------------


@analyse_group.command(cls=FormattedCommand)
@_add_analyse_options
@click.option(
    "--sort-by",
    type=click.Choice(["id", "count"], case_sensitive=False),
    default="id",
    show_default=True,
    help="Sort by class ID (0-indexed) or annotation count "
         "(overridden by --class-file)",
)
@click.option(
    "--descending/--ascending",
    default=False,
    show_default=True,
    help="Sort direction",
)
@click.option(
    "--recursive", "-R",
    is_flag=True,
    default=False,
    help="Recursively traverse subdirectories for label files "
         "(YOLO/LabelMe only)",
)
@click.argument(
    "label_paths",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
    metavar="LABEL_PATH [LABEL_PATH ...]",
)
def stats(
    ctx: click.Context,
    label_paths: tuple,
    class_file: Path,
    image_dir: Path,
    sort_by: str,
    descending: bool,
    recursive: bool,
):
    """Compute dataset statistics for annotations at one or more LABEL_PATHs.

    LABEL_PATH is a directory (YOLO .txt or LabelMe .json) or a single
    COCO JSON file. The format is auto-detected.

    Multiple LABEL_PATHs can be specified — statistics are merged into a
    single result. All paths must be the same format.

    When --class-file / -c is provided, strict validation is enforced:
    any category in the data not listed in the class file causes an error.
    """
    from dataflow.analyse import StatsAnalyser

    if not label_paths:
        raise click.UsageError("At least one LABEL_PATH is required.")

    verbose = ctx.obj["verbose"]
    log_dir = ctx.obj["log_dir"]

    log_config = LogConfig(
        name="analyse.stats",
        verbose=verbose,
        log_dir=log_dir,
    )
    analyser = StatsAnalyser(log_config=log_config)
    result = analyser.analyse(
        label_paths=list(label_paths),
        class_file=class_file,
        image_dir=image_dir,
        sort_by=sort_by,
        descending=descending,
        recursive=recursive,
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
@_add_analyse_options
@click.argument(
    "label_path",
    type=click.Path(exists=True, path_type=Path),
    metavar="LABEL_PATH",
)
@click.argument(
    "output_dir",
    type=click.Path(path_type=Path),
    metavar="OUTPUT_DIR",
)
@click.option(
    "-r",
    "--ratio",
    type=float,
    default=0.8,
    show_default=True,
    help="Train proportion",
)
@click.option(
    "-s",
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducible shuffling",
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
    """Split dataset at LABEL_PATH into train/val subsets.

    LABEL_PATH is a directory (YOLO .txt or LabelMe .json) or a single COCO JSON file. The format is auto-detected.

    COCO creates train.json and val.json in OUTPUT_DIR. YOLO/LabelMe create train/ and val/ subdirectories in OUTPUT_DIR.
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


# ---------------------------------------------------------------------------
# Subcommand: filter
# ---------------------------------------------------------------------------


@analyse_group.command(cls=FormattedCommand)
@_add_analyse_options
@click.argument(
    "label_path",
    type=click.Path(exists=True, path_type=Path),
    metavar="LABEL_PATH",
)
@click.argument(
    "original_class_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    metavar="ORIGINAL_CLASS_FILE",
)
@click.argument(
    "new_class_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    metavar="NEW_CLASS_FILE",
)
@click.argument(
    "output_dir",
    type=click.Path(path_type=Path),
    metavar="OUTPUT_DIR",
)
def filter(
    ctx: click.Context,
    label_path: Path,
    original_class_file: Path,
    new_class_file: Path,
    output_dir: Path,
    class_file: Path,
    image_dir: Path,
):
    """Filter dataset at LABEL_PATH by category.

    LABEL_PATH is a directory (YOLO .txt or LabelMe .json) or a single COCO JSON file. The format is auto-detected.

    ORIGINAL_CLASS_FILE is the source classes.txt defining all categories in the source dataset. NEW_CLASS_FILE is the target classes.txt defining which categories to keep and their new order/IDs.

    Filtered labels are written to OUTPUT_DIR. The new classes.txt is copied there as well.
    """
    from dataflow.analyse import FilterAnalyser

    verbose = ctx.obj["verbose"]
    log_dir = ctx.obj["log_dir"]

    log_config = LogConfig(
        name="analyse.filter",
        verbose=verbose,
        log_dir=log_dir,
    )
    analyser = FilterAnalyser(log_config=log_config)
    result = analyser.analyse(
        label_path,
        original_class_file=original_class_file,
        new_class_file=new_class_file,
        output_dir=output_dir,
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
        error_msg = result.errors[0] if result.errors else "Category filter failed"
        if len(result.errors) > 1:
            error_msg += "\n" + "\n".join(result.errors[1:])
        raise RuntimeCLIError(error_msg)


# ---------------------------------------------------------------------------
# Subcommand: partition
# ---------------------------------------------------------------------------


@analyse_group.command(cls=FormattedCommand)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose log output",
)
@click.option(
    "--log-dir",
    type=click.Path(path_type=Path),
    default="./logs",
    show_default=True,
    help="Log file output directory",
)
@click.option(
    "-c",
    "--class-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Classes.txt for name mapping (label mode only)",
)
@click.argument(
    "output_dir",
    type=click.Path(path_type=Path),
    metavar="OUTPUT_DIR",
)
@click.option(
    "-n",
    "--num",
    type=int,
    required=True,
    help="Number of partitions (>= 2)",
)
@click.option(
    "-l",
    "--label-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Label directory (YOLO or LabelMe). "
         "At least one of --label-dir / --image-dir required.",
)
@click.option(
    "-i",
    "--image-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Image directory. "
         "At least one of --label-dir / --image-dir required.",
)
@click.option(
    "--shuffle",
    is_flag=True,
    default=False,
    help="Randomly shuffle before partitioning (default: sequential)",
)
@click.option(
    "-s",
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for shuffle reproducibility",
)
@click.option(
    "--move",
    is_flag=True,
    default=False,
    help="Move source files instead of copying "
         "(destructive — requires confirmation)",
)
@click.pass_context
def partition(
    ctx: click.Context,
    output_dir: Path,
    num: int,
    label_dir: Path,
    image_dir: Path,
    shuffle: bool,
    seed: int,
    move: bool,
    verbose: bool,
    log_dir: Path,
    class_file: Path,
):
    """Partition dataset into N roughly-equal subsets.

    Supports three modes:
    \b
      --label-dir only     Partition label files
      --image-dir only     Partition image files
      both specified       Labels drive partition; images follow by stem

    OUTPUT_DIR receives part_1/ through part_N/ subdirectories.

    Only YOLO and LabelMe label formats are supported (not COCO).
    """
    from dataflow.analyse import PartitionAnalyser

    # Validate at least one input source
    if label_dir is None and image_dir is None:
        raise click.UsageError(
            "At least one of --label-dir / --image-dir is required."
        )

    # Validate num
    if num < 2:
        raise click.BadParameter(
            f"Number of partitions must be at least 2, got: {num}",
            param_hint="--num",
        )

    # Move confirmation
    if move:
        click.echo(
            f"\nWARNING: --move will permanently relocate source files.\n"
            f"  Source label dir:  {label_dir if label_dir else 'N/A'}\n"
            f"  Source image dir:  {image_dir if image_dir else 'N/A'}\n"
            f"  Target:            {output_dir}/\n"
        )
        if not click.confirm("Continue?", default=False):
            raise click.Abort()

    log_config = LogConfig(
        name="analyse.partition",
        verbose=verbose,
        log_dir=log_dir,
    )
    analyser = PartitionAnalyser(log_config=log_config)
    result = analyser.analyse(
        output_dir=output_dir,
        num=num,
        label_dir=label_dir,
        image_dir=image_dir,
        shuffle=shuffle,
        seed=seed,
        class_file=class_file,
        move=move,
    )

    if result.success and result.data is not None:
        if result.log_path:
            click.echo(f"\nLog saved to: {result.log_path}")
        for warning in result.warnings:
            click.echo(f"Warning: {warning}", err=True)
    else:
        for warning in result.warnings:
            click.echo(f"Warning: {warning}", err=True)
        error_msg = result.errors[0] if result.errors else "Dataset partition failed"
        if len(result.errors) > 1:
            error_msg += "\n" + "\n".join(result.errors[1:])
        raise RuntimeCLIError(error_msg)
