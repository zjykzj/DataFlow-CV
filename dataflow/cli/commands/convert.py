"""Conversion commands."""

import click
from pathlib import Path
from typing import Optional

from dataflow.cli.commands.utils import (
    validate_convert_params,
    add_common_options,
    FormattedCommand,
)
from dataflow.cli.exceptions import RuntimeCLIError
from dataflow.util.logging import LogConfig


@click.group(name="convert")
def convert_group():
    """Format conversion command group - supports conversion between multiple label formats"""
    pass


@convert_group.command(cls=FormattedCommand)
@add_common_options
@click.argument("image_dir", type=click.Path(exists=True, path_type=Path), metavar="IMAGE_DIR")
@click.argument("label_dir", type=click.Path(exists=True, path_type=Path), metavar="LABEL_DIR")
@click.argument("class_file", type=click.Path(exists=True, path_type=Path), metavar="CLASS_FILE")
@click.argument("output_file", type=click.Path(path_type=Path), metavar="OUTPUT_FILE")
@click.option(
    "--do-rle",
    is_flag=True,
    help="Use RLE encoding for COCO format",
)
@click.option(
    "--prediction",
    is_flag=True,
    help="Treat input as prediction format (YOLO: 6 tokens for detection, "
    "even tokens for segmentation). Output COCO as plain JSON list of "
    "annotation dicts (prediction format, Variant B).",
)
@click.option(
    "--pose",
    is_flag=True,
    help="Enable pose conversion (parse 5+3k pose rows; write COCO keypoints). "
    "Without it, keypoint data is dropped with a warning.",
)
@click.option(
    "--kpts-config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Keypoint-definition YAML (keypoint names + skeleton for COCO categories)",
)
def yolo2coco(
    ctx,
    image_dir: Path,
    label_dir: Path,
    class_file: Path,
    output_file: Path,
    do_rle: bool,
    prediction: bool,
    pose: bool,
    kpts_config: Optional[Path],
):
    """Convert YOLO format to COCO format"""
    from dataflow.convert import YoloAndCocoConverter

    strict = ctx.obj["strict"]
    log_config = LogConfig(
        name="convert.yolo_to_coco",
        verbose=ctx.obj["verbose"],
        log_dir=ctx.obj["log_dir"],
    )

    # Parameter validation
    validate_convert_params("yolo", "coco", label_dir, output_file, image_dir, class_file)

    # Call existing API
    converter = YoloAndCocoConverter(
        source_to_target=True, prediction=prediction, log_config=log_config, strict_mode=strict
    )
    result = converter.convert(
        source_path=str(label_dir),
        target_path=str(output_file),
        class_file=str(class_file),
        image_dir=str(image_dir),
        do_rle=do_rle,
        pose=pose,
        kpts_config=kpts_config,
        category_mapping=None,
    )

    # Print terminal output
    if result.log_path:
        click.echo(f"Log saved to: {result.log_path}")

    if result.success:
        click.echo(f"Conversion completed: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        click.echo(f"Conversion failed: {error_msg}", err=True)
        raise RuntimeCLIError(f"Conversion failed: {error_msg}")


@convert_group.command(cls=FormattedCommand)
@add_common_options
@click.argument("image_dir", type=click.Path(exists=True, path_type=Path), metavar="IMAGE_DIR")
@click.argument("label_dir", type=click.Path(exists=True, path_type=Path), metavar="LABEL_DIR")
@click.argument("class_file", type=click.Path(exists=True, path_type=Path), metavar="CLASS_FILE")
@click.argument("output_dir", type=click.Path(path_type=Path), metavar="OUTPUT_DIR")
@click.option(
    "--pose",
    is_flag=True,
    help="Enable pose conversion (parse 5+3k pose rows; write point shapes + rectangle)",
)
@click.option(
    "--kpts-config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Keypoint-definition YAML (point-shape labels; fallback: keypoint_N placeholders)",
)
def yolo2labelme(
    ctx,
    image_dir: Path,
    label_dir: Path,
    class_file: Path,
    output_dir: Path,
    pose: bool,
    kpts_config: Optional[Path],
):
    """Convert YOLO format to LabelMe format"""
    from dataflow.convert import LabelMeAndYoloConverter

    strict = ctx.obj["strict"]
    log_config = LogConfig(
        name="convert.yolo_to_labelme",
        verbose=ctx.obj["verbose"],
        log_dir=ctx.obj["log_dir"],
    )

    # Parameter validation
    validate_convert_params("yolo", "labelme", label_dir, output_dir, image_dir, class_file)

    # Call existing API
    converter = LabelMeAndYoloConverter(
        source_to_target=False, log_config=log_config, strict_mode=strict
    )
    result = converter.convert(
        source_path=str(label_dir),
        target_path=str(output_dir),
        class_file=str(class_file),
        image_dir=str(image_dir),
        pose=pose,
        kpts_config=kpts_config,
        category_mapping=None,
    )

    # Print terminal output
    if result.log_path:
        click.echo(f"Log saved to: {result.log_path}")

    if result.success:
        click.echo(f"Conversion completed: {result.get_summary()}")
    else:
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        click.echo(f"Conversion failed: {error_msg}", err=True)
        raise RuntimeCLIError(f"Conversion failed: {error_msg}")


@convert_group.command(cls=FormattedCommand)
@add_common_options
@click.argument("labelme_dir", type=click.Path(exists=True, path_type=Path), metavar="LABELME_DIR")
@click.argument("class_file", type=click.Path(exists=True, path_type=Path), metavar="CLASS_FILE")
@click.argument("output_dir", type=click.Path(path_type=Path), metavar="OUTPUT_DIR")
@click.option(
    "--pose",
    is_flag=True,
    help="Enable pose conversion (write 5+3k pose rows)",
)
@click.option(
    "--kpts-config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Keypoint-definition YAML (keypoint ordering; fallback: file order with a warning)",
)
def labelme2yolo(
    ctx,
    labelme_dir: Path,
    class_file: Path,
    output_dir: Path,
    pose: bool,
    kpts_config: Optional[Path],
):
    """Convert LabelMe format to YOLO format"""
    from dataflow.convert import LabelMeAndYoloConverter

    strict = ctx.obj["strict"]
    log_config = LogConfig(
        name="convert.labelme_to_yolo",
        verbose=ctx.obj["verbose"],
        log_dir=ctx.obj["log_dir"],
    )

    # Parameter validation
    validate_convert_params("labelme", "yolo", labelme_dir, output_dir, None, class_file)

    # Call existing API
    converter = LabelMeAndYoloConverter(
        source_to_target=True, log_config=log_config, strict_mode=strict
    )
    result = converter.convert(
        source_path=str(labelme_dir),
        target_path=str(output_dir),
        class_file=str(class_file),
        image_dir=None,
        pose=pose,
        kpts_config=kpts_config,
        category_mapping=None,
    )

    # Print terminal output
    if result.log_path:
        click.echo(f"Log saved to: {result.log_path}")

    if result.success:
        click.echo(f"Conversion completed: {result.get_summary()}")
    else:
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        click.echo(f"Conversion failed: {error_msg}", err=True)
        raise RuntimeCLIError(f"Conversion failed: {error_msg}")


@convert_group.command(cls=FormattedCommand)
@add_common_options
@click.argument("labelme_dir", type=click.Path(exists=True, path_type=Path), metavar="LABELME_DIR")
@click.argument("class_file", type=click.Path(exists=True, path_type=Path), metavar="CLASS_FILE")
@click.argument("output_file", type=click.Path(path_type=Path), metavar="OUTPUT_FILE")
@click.option(
    "--do-rle",
    is_flag=True,
    help="Use RLE encoding for COCO format",
)
@click.option(
    "--pose",
    is_flag=True,
    help="Enable pose conversion (write COCO keypoints + num_keypoints)",
)
@click.option(
    "--kpts-config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Keypoint-definition YAML (keypoint ordering + COCO category names/skeleton)",
)
def labelme2coco(
    ctx,
    labelme_dir: Path,
    class_file: Path,
    output_file: Path,
    do_rle: bool,
    pose: bool,
    kpts_config: Optional[Path],
):
    """Convert LabelMe format to COCO format"""
    from dataflow.convert import CocoAndLabelMeConverter

    strict = ctx.obj["strict"]
    log_config = LogConfig(
        name="convert.labelme_to_coco",
        verbose=ctx.obj["verbose"],
        log_dir=ctx.obj["log_dir"],
    )

    # Parameter validation
    validate_convert_params("labelme", "coco", labelme_dir, output_file, None, class_file)

    # Call existing API
    converter = CocoAndLabelMeConverter(
        source_to_target=False, log_config=log_config, strict_mode=strict
    )
    result = converter.convert(
        source_path=str(labelme_dir),
        target_path=str(output_file),
        class_file=str(class_file),
        image_dir=None,
        category_mapping=None,
        do_rle=do_rle,
        pose=pose,
        kpts_config=kpts_config,
    )

    # Print terminal output
    if result.log_path:
        click.echo(f"Log saved to: {result.log_path}")

    if result.success:
        click.echo(f"Conversion completed: {result.get_summary()}")
    else:
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        click.echo(f"Conversion failed: {error_msg}", err=True)
        raise RuntimeCLIError(f"Conversion failed: {error_msg}")


@convert_group.command(cls=FormattedCommand)
@add_common_options
@click.option(
    "--save-crowd",
    is_flag=True,
    default=False,
    help="Write iscrowd=1 annotations as ordinary rows (default: skip them)",
)
@click.option(
    "--pose",
    is_flag=True,
    help="Enable pose conversion (write 5+3k pose rows; segmentation is dropped "
    "with a warning). Without it, keypoint data is dropped with a warning.",
)
@click.option(
    "--save-kpts-config",
    type=click.Path(path_type=Path),
    default=None,
    help="Write a keypoint-definition YAML sidecar generated from COCO categories",
)
@click.argument("input_path", type=click.Path(exists=True, path_type=Path), metavar="COCO_FILE")
@click.argument("output_path", type=click.Path(path_type=Path), metavar="OUTPUT_DIR")
def coco2yolo(
    ctx,
    input_path: Path,
    output_path: Path,
    save_crowd: bool,
    pose: bool,
    save_kpts_config: Optional[Path],
):
    """Convert COCO format to YOLO format"""
    from dataflow.convert import YoloAndCocoConverter

    strict = ctx.obj["strict"]
    log_config = LogConfig(
        name="convert.coco_to_yolo",
        verbose=ctx.obj["verbose"],
        log_dir=ctx.obj["log_dir"],
    )

    # Parameter validation
    validate_convert_params("coco", "yolo", input_path, output_path, None, None)

    # Call existing API
    converter = YoloAndCocoConverter(
        source_to_target=False, log_config=log_config, strict_mode=strict
    )
    result = converter.convert(
        source_path=str(input_path),
        target_path=str(output_path),
        class_file=None,
        image_dir=None,
        category_mapping=None,
        save_crowd=save_crowd,
        pose=pose,
        save_kpts_config=save_kpts_config,
    )

    # Print terminal output
    if result.log_path:
        click.echo(f"Log saved to: {result.log_path}")

    if result.success:
        click.echo(f"Conversion completed: {result.get_summary()}")
    else:
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        click.echo(f"Conversion failed: {error_msg}", err=True)
        raise RuntimeCLIError(f"Conversion failed: {error_msg}")


@convert_group.command(cls=FormattedCommand)
@add_common_options
@click.option(
    "--save-crowd",
    is_flag=True,
    default=False,
    help="Write iscrowd=1 annotations as ordinary shapes (default: skip them)",
)
@click.option(
    "--pose",
    is_flag=True,
    help="Enable pose conversion (write point shapes + rectangle; v=1 degrades "
    "to v=2 with a warning). Without it, keypoint data is dropped with a warning.",
)
@click.option(
    "--save-kpts-config",
    type=click.Path(path_type=Path),
    default=None,
    help="Write a keypoint-definition YAML sidecar generated from COCO categories",
)
@click.argument("input_path", type=click.Path(exists=True, path_type=Path), metavar="COCO_FILE")
@click.argument("output_path", type=click.Path(path_type=Path), metavar="OUTPUT_DIR")
def coco2labelme(
    ctx,
    input_path: Path,
    output_path: Path,
    save_crowd: bool,
    pose: bool,
    save_kpts_config: Optional[Path],
):
    """Convert COCO format to LabelMe format"""
    from dataflow.convert import CocoAndLabelMeConverter

    strict = ctx.obj["strict"]
    log_config = LogConfig(
        name="convert.coco_to_labelme",
        verbose=ctx.obj["verbose"],
        log_dir=ctx.obj["log_dir"],
    )

    # Parameter validation
    validate_convert_params("coco", "labelme", input_path, output_path, None, None)

    # Call existing API
    converter = CocoAndLabelMeConverter(
        source_to_target=True, log_config=log_config, strict_mode=strict
    )
    result = converter.convert(
        source_path=str(input_path),
        target_path=str(output_path),
        class_file=None,
        image_dir=None,
        category_mapping=None,
        save_crowd=save_crowd,
        pose=pose,
        save_kpts_config=save_kpts_config,
    )

    # Print terminal output
    if result.log_path:
        click.echo(f"Log saved to: {result.log_path}")

    if result.success:
        click.echo(f"Conversion completed: {result.get_summary()}")
    else:
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        click.echo(f"Conversion failed: {error_msg}", err=True)
        raise RuntimeCLIError(f"Conversion failed: {error_msg}")
