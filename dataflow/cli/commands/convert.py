"""Conversion commands."""

import json
import click
from pathlib import Path
from typing import Optional

from dataflow.cli.commands.utils import validate_convert_params, add_common_options
from dataflow.cli.exceptions import RuntimeCLIError


@click.group(name="convert")
def convert_group():
    """Format conversion command group - supports conversion between multiple label formats"""
    pass


@convert_group.command()
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="Image file directory (for obtaining image dimensions)",
)
@click.option(
    "--class-file",
    "-c",
    type=click.Path(path_type=Path),
    help="Class file path (required for YOLO format)",
)
@click.option(
    "--do-rle",
    is_flag=True,
    help="Use RLE encoding for COCO format",
)
@click.option(
    "--category-mapping",
    type=click.Path(path_type=Path),
    help="Custom category mapping file (JSON format)",
)
@click.option(
    "--skip-errors",
    is_flag=True,
    help="Skip errors and continue processing (lenient mode)",
)
def yolo2coco(
    ctx,
    input_path: Path,
    output_path: Path,
    image_dir: Optional[Path],
    class_file: Optional[Path],
    do_rle: bool,
    category_mapping: Optional[Path],
    skip_errors: bool,
):
    """Convert YOLO format to COCO format"""
    from dataflow.convert.yolo_and_coco import YoloAndCocoConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"] and not skip_errors  # Combine global strict and local skip-errors

    logger.info(f"Starting conversion ofYOLOtoCOCO: {input_path} -> {output_path}")

    # Parameter validation
    validate_convert_params("yolo", "coco", input_path, output_path, image_dir, class_file)

    # Load category mapping
    category_mapping_dict = None
    if category_mapping:
        category_mapping = validate_path_exists(category_mapping, "category mapping file")
        with open(category_mapping, "r", encoding="utf-8") as f:
            category_mapping_dict = json.load(f)

    # Call existing API
    converter = YoloAndCocoConverter(source_to_target=True, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(input_path),
        target_path=str(output_path),
        class_file=str(class_file) if class_file else None,
        image_dir=str(image_dir) if image_dir else None,
        do_rle=do_rle,
        category_mapping=category_mapping_dict,
    )

    if result.success:
        logger.info(f"Conversion completed: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        logger.error(f"Conversion failed: {error_msg}")
        raise RuntimeCLIError(f"Conversion failed: {error_msg}")


@convert_group.command()
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="Image file directory (for obtaining image dimensions)",
)
@click.option(
    "--class-file",
    "-c",
    type=click.Path(path_type=Path),
    help="Class file path (required for YOLO format)",
)
@click.option(
    "--category-mapping",
    type=click.Path(path_type=Path),
    help="Custom category mapping file (JSON format)",
)
@click.option(
    "--skip-errors",
    is_flag=True,
    help="Skip errors and continue processing (lenient mode)",
)
def yolo2labelme(
    ctx,
    input_path: Path,
    output_path: Path,
    image_dir: Optional[Path],
    class_file: Optional[Path],
    category_mapping: Optional[Path],
    skip_errors: bool,
):
    """Convert YOLO format to LabelMe format"""
    from dataflow.convert.labelme_and_yolo import LabelMeAndYoloConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"] and not skip_errors

    logger.info(f"Starting conversion ofYOLOtoLabelMe: {input_path} -> {output_path}")

    # Parameter validation
    validate_convert_params("yolo", "labelme", input_path, output_path, image_dir, class_file)

    # Load category mapping
    category_mapping_dict = None
    if category_mapping:
        category_mapping = validate_path_exists(category_mapping, "category mapping file")
        with open(category_mapping, "r", encoding="utf-8") as f:
            category_mapping_dict = json.load(f)

    # Call existing API
    converter = LabelMeAndYoloConverter(source_to_target=True, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(input_path),
        target_path=str(output_path),
        class_file=str(class_file) if class_file else None,
        image_dir=str(image_dir) if image_dir else None,
        category_mapping=category_mapping_dict,
    )

    if result.success:
        logger.info(f"Conversion completed: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        logger.error(f"Conversion failed: {error_msg}")
        raise RuntimeCLIError(f"Conversion failed: {error_msg}")


@convert_group.command()
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="Image file directory (for obtaining image dimensions)",
)
@click.option(
    "--class-file",
    "-c",
    type=click.Path(path_type=Path),
    help="Class file path (required for YOLO format)",
)
@click.option(
    "--category-mapping",
    type=click.Path(path_type=Path),
    help="Custom category mapping file (JSON format)",
)
@click.option(
    "--skip-errors",
    is_flag=True,
    help="Skip errors and continue processing (lenient mode)",
)
def coco2yolo(
    ctx,
    input_path: Path,
    output_path: Path,
    image_dir: Optional[Path],
    class_file: Optional[Path],
    category_mapping: Optional[Path],
    skip_errors: bool,
):
    """Convert COCO format to YOLO format"""
    from dataflow.convert.yolo_and_coco import YoloAndCocoConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"] and not skip_errors

    logger.info(f"Starting conversion ofCOCOtoYOLO: {input_path} -> {output_path}")

    # Parameter validation
    validate_convert_params("coco", "yolo", input_path, output_path, image_dir, class_file)

    # Load category mapping
    category_mapping_dict = None
    if category_mapping:
        category_mapping = validate_path_exists(category_mapping, "category mapping file")
        with open(category_mapping, "r", encoding="utf-8") as f:
            category_mapping_dict = json.load(f)

    # Call existing API
    converter = YoloAndCocoConverter(source_to_target=False, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(input_path),
        target_path=str(output_path),
        class_file=str(class_file) if class_file else None,
        image_dir=str(image_dir) if image_dir else None,
        category_mapping=category_mapping_dict,
    )

    if result.success:
        logger.info(f"Conversion completed: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        logger.error(f"Conversion failed: {error_msg}")
        raise RuntimeCLIError(f"Conversion failed: {error_msg}")


@convert_group.command()
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="Image file directory (for obtaining image dimensions)",
)
@click.option(
    "--category-mapping",
    type=click.Path(path_type=Path),
    help="Custom category mapping file (JSON format)",
)
@click.option(
    "--skip-errors",
    is_flag=True,
    help="Skip errors and continue processing (lenient mode)",
)
def coco2labelme(
    ctx,
    input_path: Path,
    output_path: Path,
    image_dir: Optional[Path],
    category_mapping: Optional[Path],
    skip_errors: bool,
):
    """Convert COCO format to LabelMe format"""
    from dataflow.convert.coco_and_labelme import CocoAndLabelMeConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"] and not skip_errors

    logger.info(f"Starting conversion ofCOCOtoLabelMe: {input_path} -> {output_path}")

    # Parameter validation
    validate_convert_params("coco", "labelme", input_path, output_path, image_dir, None)

    # Load category mapping
    category_mapping_dict = None
    if category_mapping:
        category_mapping = validate_path_exists(category_mapping, "category mapping file")
        with open(category_mapping, "r", encoding="utf-8") as f:
            category_mapping_dict = json.load(f)

    # Call existing API
    converter = CocoAndLabelMeConverter(source_to_target=True, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(input_path),
        target_path=str(output_path),
        image_dir=str(image_dir) if image_dir else None,
        category_mapping=category_mapping_dict,
    )

    if result.success:
        logger.info(f"Conversion completed: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        logger.error(f"Conversion failed: {error_msg}")
        raise RuntimeCLIError(f"Conversion failed: {error_msg}")


@convert_group.command()
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="Image file directory (for obtaining image dimensions)",
)
@click.option(
    "--class-file",
    "-c",
    type=click.Path(path_type=Path),
    help="Class file path (required for YOLO format)",
)
@click.option(
    "--category-mapping",
    type=click.Path(path_type=Path),
    help="Custom category mapping file (JSON format)",
)
@click.option(
    "--skip-errors",
    is_flag=True,
    help="Skip errors and continue processing (lenient mode)",
)
def labelme2yolo(
    ctx,
    input_path: Path,
    output_path: Path,
    image_dir: Optional[Path],
    class_file: Optional[Path],
    category_mapping: Optional[Path],
    skip_errors: bool,
):
    """Convert LabelMe format to YOLO format"""
    from dataflow.convert.labelme_and_yolo import LabelMeAndYoloConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"] and not skip_errors

    logger.info(f"Starting conversion ofLabelMetoYOLO: {input_path} -> {output_path}")

    # Parameter validation
    validate_convert_params("labelme", "yolo", input_path, output_path, image_dir, class_file)

    # Load category mapping
    category_mapping_dict = None
    if category_mapping:
        category_mapping = validate_path_exists(category_mapping, "category mapping file")
        with open(category_mapping, "r", encoding="utf-8") as f:
            category_mapping_dict = json.load(f)

    # Call existing API
    converter = LabelMeAndYoloConverter(source_to_target=False, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(input_path),
        target_path=str(output_path),
        class_file=str(class_file) if class_file else None,
        image_dir=str(image_dir) if image_dir else None,
        category_mapping=category_mapping_dict,
    )

    if result.success:
        logger.info(f"Conversion completed: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "Conversion failed"
        logger.error(f"Conversion failed: {error_msg}")
        raise RuntimeCLIError(f"Conversion failed: {error_msg}")


@convert_group.command()
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="Image file directory (for obtaining image dimensions)",
)
@click.option(
    "--category-mapping",
    type=click.Path(path_type=Path),
    help="Custom category mapping file (JSON format)",
)
@click.option(
    "--skip-errors",
    is_flag=True,
    help="Skip errors and continue processing (lenient mode)",
)
def labelme2coco(
    ctx,
    input_path: Path,
    output_path: Path,
    image_dir: Optional[Path],
    category_mapping: Optional[Path],
    skip_errors: bool,
):
    """Convert LabelMe format to COCO format"""
    from dataflow.convert.coco_and_labelme import CocoAndLabelMeConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"] and not skip_errors

    logger.info(f"Starting conversion ofLabelMetoCOCO: {input_path} -> {output_path}")

    # Parameter validation
    validate_convert_params("labelme", "coco", input_path, output_path, image_dir, None)

    # Load category mapping
    category_mapping_dict = None
    if category_mapping:
        category_mapping = validate_path_exists(category_mapping, "category mapping file")
        with open(category_mapping, "r", encoding="utf-8") as f:
            category_mapping_dict = json.load(f)

    # Call existing API
    converter = CocoAndLabelMeConverter(source_to_target=False, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(input_path),
        target_path=str(output_path),
        image_dir=str(image_dir) if image_dir else None,
        category_mapping=category_mapping_dict,
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