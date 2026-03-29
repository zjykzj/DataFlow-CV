"""Conversion commands."""

import json
import click
from pathlib import Path
from typing import Optional

from dataflow.cli.commands.utils import validate_convert_params, add_common_options, FormattedCommand
from dataflow.cli.exceptions import RuntimeCLIError


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
def yolo2coco(
    ctx,
    image_dir: Path,
    label_dir: Path,
    class_file: Path,
    output_file: Path,
    do_rle: bool,
):
    """Convert YOLO format to COCO format"""
    from dataflow.convert.yolo_and_coco import YoloAndCocoConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"]  # Use global strict mode (default True)

    logger.info(f"Starting conversion ofYOLOtoCOCO: {label_dir} -> {output_file}")

    # Parameter validation
    validate_convert_params("yolo", "coco", label_dir, output_file, image_dir, class_file)

    # Call existing API
    converter = YoloAndCocoConverter(source_to_target=True, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(label_dir),
        target_path=str(output_file),
        class_file=str(class_file),
        image_dir=str(image_dir),
        do_rle=do_rle,
        category_mapping=None,
    )

    if result.success:
        logger.info(f"Conversion completed: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        logger.error(f"Conversion failed: {error_msg}")
        raise RuntimeCLIError(f"Conversion failed: {error_msg}")


@convert_group.command(cls=FormattedCommand)
@add_common_options
@click.argument("image_dir", type=click.Path(exists=True, path_type=Path), metavar="IMAGE_DIR")
@click.argument("label_dir", type=click.Path(exists=True, path_type=Path), metavar="LABEL_DIR")
@click.argument("class_file", type=click.Path(exists=True, path_type=Path), metavar="CLASS_FILE")
@click.argument("output_dir", type=click.Path(path_type=Path), metavar="OUTPUT_DIR")
def yolo2labelme(
    ctx,
    image_dir: Path,
    label_dir: Path,
    class_file: Path,
    output_dir: Path,
):
    """Convert YOLO format to LabelMe format"""
    from dataflow.convert.labelme_and_yolo import LabelMeAndYoloConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"]

    logger.info(f"Starting conversion ofYOLOtoLabelMe: {label_dir} -> {output_dir}")

    # Parameter validation
    validate_convert_params("yolo", "labelme", label_dir, output_dir, image_dir, class_file)

    # Call existing API
    converter = LabelMeAndYoloConverter(source_to_target=False, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(label_dir),
        target_path=str(output_dir),
        class_file=str(class_file),
        image_dir=str(image_dir),
        category_mapping=None,
    )

    if result.success:
        logger.info(f"Conversion completed: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        logger.error(f"Conversion failed: {error_msg}")
        raise RuntimeCLIError(f"Conversion failed: {error_msg}")


@convert_group.command(cls=FormattedCommand)
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path), metavar="COCO_FILE")
@click.argument("output_path", type=click.Path(path_type=Path), metavar="OUTPUT_DIR")
def coco2yolo(
    ctx,
    input_path: Path,
    output_path: Path,
):
    """Convert COCO format to YOLO format"""
    from dataflow.convert.yolo_and_coco import YoloAndCocoConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"]

    logger.info(f"Starting conversion ofCOCOtoYOLO: {input_path} -> {output_path}")

    # Parameter validation
    validate_convert_params("coco", "yolo", input_path, output_path, None, None)

    # Call existing API
    converter = YoloAndCocoConverter(source_to_target=False, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(input_path),
        target_path=str(output_path),
        class_file=None,
        image_dir=None,
        category_mapping=None,
    )

    if result.success:
        logger.info(f"Conversion completed: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        logger.error(f"Conversion failed: {error_msg}")
        raise RuntimeCLIError(f"Conversion failed: {error_msg}")


@convert_group.command(cls=FormattedCommand)
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path), metavar="COCO_FILE")
@click.argument("output_path", type=click.Path(path_type=Path), metavar="OUTPUT_DIR")
def coco2labelme(
    ctx,
    input_path: Path,
    output_path: Path,
):
    """Convert COCO format to LabelMe format"""
    from dataflow.convert.coco_and_labelme import CocoAndLabelMeConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"]

    logger.info(f"Starting conversion ofCOCOtoLabelMe: {input_path} -> {output_path}")

    # Parameter validation
    validate_convert_params("coco", "labelme", input_path, output_path, None, None)

    # Call existing API
    converter = CocoAndLabelMeConverter(source_to_target=True, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(input_path),
        target_path=str(output_path),
        class_file=None,
        image_dir=None,
        category_mapping=None,
    )

    if result.success:
        logger.info(f"Conversion completed: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        logger.error(f"Conversion failed: {error_msg}")
        raise RuntimeCLIError(f"Conversion failed: {error_msg}")


@convert_group.command(cls=FormattedCommand)
@add_common_options
@click.argument("labelme_dir", type=click.Path(exists=True, path_type=Path), metavar="LABELME_DIR")
@click.argument("class_file", type=click.Path(exists=True, path_type=Path), metavar="CLASS_FILE")
@click.argument("output_dir", type=click.Path(path_type=Path), metavar="OUTPUT_DIR")
def labelme2yolo(
    ctx,
    labelme_dir: Path,
    class_file: Path,
    output_dir: Path,
):
    """Convert LabelMe format to YOLO format"""
    from dataflow.convert.labelme_and_yolo import LabelMeAndYoloConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"]

    logger.info(f"Starting conversion ofLabelMetoYOLO: {labelme_dir} -> {output_dir}")

    # Parameter validation
    validate_convert_params("labelme", "yolo", labelme_dir, output_dir, None, class_file)

    # Call existing API
    converter = LabelMeAndYoloConverter(source_to_target=True, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(labelme_dir),
        target_path=str(output_dir),
        class_file=str(class_file),
        image_dir=None,
        category_mapping=None,
    )

    if result.success:
        logger.info(f"Conversion completed: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        logger.error(f"Conversion failed: {error_msg}")
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
def labelme2coco(
    ctx,
    labelme_dir: Path,
    class_file: Path,
    output_file: Path,
    do_rle: bool,
):
    """Convert LabelMe format to COCO format"""
    from dataflow.convert.coco_and_labelme import CocoAndLabelMeConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"]

    logger.info(f"Starting conversion ofLabelMetoCOCO: {labelme_dir} -> {output_file}")

    # Parameter validation
    validate_convert_params("labelme", "coco", labelme_dir, output_file, None, class_file)

    # Call existing API
    converter = CocoAndLabelMeConverter(source_to_target=False, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(labelme_dir),
        target_path=str(output_file),
        class_file=str(class_file),
        image_dir=None,
        category_mapping=None,
        do_rle=do_rle,
    )

    if result.success:
        logger.info(f"Conversion completed: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        logger.error(f"Conversion failed: {error_msg}")
        raise RuntimeCLIError(f"Conversion failed: {error_msg}")


def validate_path_exists(path: Path, name: str = "path") -> Path:
    """Validate if path exists"""
    if not path.exists():
        from dataflow.cli.exceptions import InputError
        raise InputError(f"{name} does not exist: {path}")
    return path