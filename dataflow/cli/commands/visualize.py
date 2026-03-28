"""Visualization commands."""

import click
from pathlib import Path
from typing import Optional

from dataflow.cli.commands.utils import validate_visualize_params, add_common_options
from dataflow.cli.exceptions import RuntimeCLIError


@click.group(name="visualize")
def visualize_group():
    """Visualization command group - supports visualization of multiple label formats"""
    pass


@visualize_group.command()
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    required=True,
    help="Image file directory (required)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Directory to save visualization results",
)
@click.option(
    "--class-file",
    "-c",
    type=click.Path(path_type=Path),
    required=True,
    help="Class file path (required for YOLO format)",
)
@click.option("--display", "-d", is_flag=True, help="Display results interactively")
@click.option(
    "--save", "-s", is_flag=True, default=True, help="Save visualization results to files"
)
@click.option(
    "--color-scheme",
    type=click.Choice(["random", "category", "consistent"]),
    default="random",
    help="Color scheme: random/category/consistent",
)
@click.option("--thickness", type=int, default=2, help="Bounding box/polygon line thickness")
def yolo(
    ctx,
    input_path: Path,
    image_dir: Path,
    output_dir: Optional[Path],
    class_file: Path,
    display: bool,
    save: bool,
    color_scheme: str,
    thickness: int,
):
    """Visualize YOLO format labels"""
    from dataflow.visualize.yolo_visualizer import YOLOVisualizer

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"]

    logger.info(f"Starting visualization of YOLO labels: {input_path}")

    # Parameter validation
    validate_visualize_params(input_path, image_dir, output_dir)

    # Call existing API
    visualizer = YOLOVisualizer(
        label_dir=input_path,
        image_dir=image_dir,
        class_file=class_file,
        output_dir=output_dir,
        is_show=display,
        is_save=save,
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
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    required=True,
    help="Image file directory (required)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Directory to save visualization results",
)
@click.option("--display", "-d", is_flag=True, help="Display results interactively")
@click.option(
    "--save", "-s", is_flag=True, default=True, help="Save visualization results to files",
)
@click.option(
    "--color-scheme",
    type=click.Choice(["random", "category", "consistent"]),
    default="random",
    help="Color scheme: random/category/consistent",
)
@click.option("--thickness", type=int, default=2, help="Bounding box/polygon line thickness")
def coco(
    ctx,
    input_path: Path,
    image_dir: Path,
    output_dir: Optional[Path],
    display: bool,
    save: bool,
    color_scheme: str,
    thickness: int,
):
    """Visualize COCO format labels"""
    from dataflow.visualize.coco_visualizer import COCOVisualizer

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"]

    logger.info(f"Starting visualization of COCO labels: {input_path}")

    # Parameter validation
    validate_visualize_params(input_path, image_dir, output_dir)

    # Call existing API
    visualizer = COCOVisualizer(
        annotation_file=input_path,
        image_dir=image_dir,
        output_dir=output_dir,
        is_show=display,
        is_save=save,
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
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    required=True,
    help="Image file directory (required)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Directory to save visualization results",
)
@click.option("--display", "-d", is_flag=True, help="Display results interactively")
@click.option(
    "--save", "-s", is_flag=True, default=True, help="Save visualization results to files",
)
@click.option(
    "--color-scheme",
    type=click.Choice(["random", "category", "consistent"]),
    default="random",
    help="Color scheme: random/category/consistent",
)
@click.option("--thickness", type=int, default=2, help="Bounding box/polygon line thickness")
def labelme(
    ctx,
    input_path: Path,
    image_dir: Path,
    output_dir: Optional[Path],
    display: bool,
    save: bool,
    color_scheme: str,
    thickness: int,
):
    """Visualize LabelMe format labels"""
    from dataflow.visualize.labelme_visualizer import LabelMeVisualizer

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"]

    logger.info(f"Starting visualization of LabelMe labels: {input_path}")

    # Parameter validation
    validate_visualize_params(input_path, image_dir, output_dir)

    # Call existing API
    visualizer = LabelMeVisualizer(
        label_dir=input_path,
        image_dir=image_dir,
        output_dir=output_dir,
        is_show=display,
        is_save=save,
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