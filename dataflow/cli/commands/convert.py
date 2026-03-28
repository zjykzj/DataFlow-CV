"""Conversion commands."""

import json
import click
from pathlib import Path
from typing import Optional

from dataflow.cli.commands.utils import validate_convert_params, add_common_options
from dataflow.cli.exceptions import RuntimeCLIError


@click.group(name="convert")
def convert_group():
    """格式转换命令组 - 支持多种标签格式间的转换"""
    pass


@convert_group.command()
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="图像文件目录（用于获取图像尺寸）",
)
@click.option(
    "--class-file",
    "-c",
    type=click.Path(path_type=Path),
    help="类别文件路径（YOLO格式需要）",
)
@click.option(
    "--do-rle",
    is_flag=True,
    help="对COCO格式使用RLE编码",
)
@click.option(
    "--category-mapping",
    type=click.Path(path_type=Path),
    help="自定义类别映射文件（JSON格式）",
)
@click.option(
    "--skip-errors",
    is_flag=True,
    help="跳过错误继续处理（宽松模式）",
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
    """YOLO格式转COCO格式"""
    from dataflow.convert.yolo_and_coco import YoloAndCocoConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"] and not skip_errors  # 结合全局strict和本地skip-errors

    logger.info(f"开始转换YOLO到COCO: {input_path} -> {output_path}")

    # 参数验证
    validate_convert_params("yolo", "coco", input_path, output_path, image_dir, class_file)

    # 加载类别映射
    category_mapping_dict = None
    if category_mapping:
        category_mapping = validate_path_exists(category_mapping, "类别映射文件")
        with open(category_mapping, "r", encoding="utf-8") as f:
            category_mapping_dict = json.load(f)

    # 调用现有API
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
        logger.info(f"转换完成: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "转换失败"
        logger.error(f"转换失败: {error_msg}")
        raise RuntimeCLIError(f"转换失败: {error_msg}")


@convert_group.command()
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="图像文件目录（用于获取图像尺寸）",
)
@click.option(
    "--class-file",
    "-c",
    type=click.Path(path_type=Path),
    help="类别文件路径（YOLO格式需要）",
)
@click.option(
    "--category-mapping",
    type=click.Path(path_type=Path),
    help="自定义类别映射文件（JSON格式）",
)
@click.option(
    "--skip-errors",
    is_flag=True,
    help="跳过错误继续处理（宽松模式）",
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
    """YOLO格式转LabelMe格式"""
    from dataflow.convert.labelme_and_yolo import LabelMeAndYoloConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"] and not skip_errors

    logger.info(f"开始转换YOLO到LabelMe: {input_path} -> {output_path}")

    # 参数验证
    validate_convert_params("yolo", "labelme", input_path, output_path, image_dir, class_file)

    # 加载类别映射
    category_mapping_dict = None
    if category_mapping:
        category_mapping = validate_path_exists(category_mapping, "类别映射文件")
        with open(category_mapping, "r", encoding="utf-8") as f:
            category_mapping_dict = json.load(f)

    # 调用现有API
    converter = LabelMeAndYoloConverter(source_to_target=True, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(input_path),
        target_path=str(output_path),
        class_file=str(class_file) if class_file else None,
        image_dir=str(image_dir) if image_dir else None,
        category_mapping=category_mapping_dict,
    )

    if result.success:
        logger.info(f"转换完成: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "转换失败"
        logger.error(f"转换失败: {error_msg}")
        raise RuntimeCLIError(f"转换失败: {error_msg}")


@convert_group.command()
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="图像文件目录（用于获取图像尺寸）",
)
@click.option(
    "--class-file",
    "-c",
    type=click.Path(path_type=Path),
    help="类别文件路径（YOLO格式需要）",
)
@click.option(
    "--category-mapping",
    type=click.Path(path_type=Path),
    help="自定义类别映射文件（JSON格式）",
)
@click.option(
    "--skip-errors",
    is_flag=True,
    help="跳过错误继续处理（宽松模式）",
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
    """COCO格式转YOLO格式"""
    from dataflow.convert.yolo_and_coco import YoloAndCocoConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"] and not skip_errors

    logger.info(f"开始转换COCO到YOLO: {input_path} -> {output_path}")

    # 参数验证
    validate_convert_params("coco", "yolo", input_path, output_path, image_dir, class_file)

    # 加载类别映射
    category_mapping_dict = None
    if category_mapping:
        category_mapping = validate_path_exists(category_mapping, "类别映射文件")
        with open(category_mapping, "r", encoding="utf-8") as f:
            category_mapping_dict = json.load(f)

    # 调用现有API
    converter = YoloAndCocoConverter(source_to_target=False, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(input_path),
        target_path=str(output_path),
        class_file=str(class_file) if class_file else None,
        image_dir=str(image_dir) if image_dir else None,
        category_mapping=category_mapping_dict,
    )

    if result.success:
        logger.info(f"转换完成: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "转换失败"
        logger.error(f"转换失败: {error_msg}")
        raise RuntimeCLIError(f"转换失败: {error_msg}")


@convert_group.command()
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="图像文件目录（用于获取图像尺寸）",
)
@click.option(
    "--category-mapping",
    type=click.Path(path_type=Path),
    help="自定义类别映射文件（JSON格式）",
)
@click.option(
    "--skip-errors",
    is_flag=True,
    help="跳过错误继续处理（宽松模式）",
)
def coco2labelme(
    ctx,
    input_path: Path,
    output_path: Path,
    image_dir: Optional[Path],
    category_mapping: Optional[Path],
    skip_errors: bool,
):
    """COCO格式转LabelMe格式"""
    from dataflow.convert.coco_and_labelme import CocoAndLabelMeConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"] and not skip_errors

    logger.info(f"开始转换COCO到LabelMe: {input_path} -> {output_path}")

    # 参数验证
    validate_convert_params("coco", "labelme", input_path, output_path, image_dir, None)

    # 加载类别映射
    category_mapping_dict = None
    if category_mapping:
        category_mapping = validate_path_exists(category_mapping, "类别映射文件")
        with open(category_mapping, "r", encoding="utf-8") as f:
            category_mapping_dict = json.load(f)

    # 调用现有API
    converter = CocoAndLabelMeConverter(source_to_target=True, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(input_path),
        target_path=str(output_path),
        image_dir=str(image_dir) if image_dir else None,
        category_mapping=category_mapping_dict,
    )

    if result.success:
        logger.info(f"转换完成: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "转换失败"
        logger.error(f"转换失败: {error_msg}")
        raise RuntimeCLIError(f"转换失败: {error_msg}")


@convert_group.command()
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="图像文件目录（用于获取图像尺寸）",
)
@click.option(
    "--class-file",
    "-c",
    type=click.Path(path_type=Path),
    help="类别文件路径（YOLO格式需要）",
)
@click.option(
    "--category-mapping",
    type=click.Path(path_type=Path),
    help="自定义类别映射文件（JSON格式）",
)
@click.option(
    "--skip-errors",
    is_flag=True,
    help="跳过错误继续处理（宽松模式）",
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
    """LabelMe格式转YOLO格式"""
    from dataflow.convert.labelme_and_yolo import LabelMeAndYoloConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"] and not skip_errors

    logger.info(f"开始转换LabelMe到YOLO: {input_path} -> {output_path}")

    # 参数验证
    validate_convert_params("labelme", "yolo", input_path, output_path, image_dir, class_file)

    # 加载类别映射
    category_mapping_dict = None
    if category_mapping:
        category_mapping = validate_path_exists(category_mapping, "类别映射文件")
        with open(category_mapping, "r", encoding="utf-8") as f:
            category_mapping_dict = json.load(f)

    # 调用现有API
    converter = LabelMeAndYoloConverter(source_to_target=False, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(input_path),
        target_path=str(output_path),
        class_file=str(class_file) if class_file else None,
        image_dir=str(image_dir) if image_dir else None,
        category_mapping=category_mapping_dict,
    )

    if result.success:
        logger.info(f"转换完成: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "转换失败"
        logger.error(f"转换失败: {error_msg}")
        raise RuntimeCLIError(f"转换失败: {error_msg}")


@convert_group.command()
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="图像文件目录（用于获取图像尺寸）",
)
@click.option(
    "--category-mapping",
    type=click.Path(path_type=Path),
    help="自定义类别映射文件（JSON格式）",
)
@click.option(
    "--skip-errors",
    is_flag=True,
    help="跳过错误继续处理（宽松模式）",
)
def labelme2coco(
    ctx,
    input_path: Path,
    output_path: Path,
    image_dir: Optional[Path],
    category_mapping: Optional[Path],
    skip_errors: bool,
):
    """LabelMe格式转COCO格式"""
    from dataflow.convert.coco_and_labelme import CocoAndLabelMeConverter

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"] and not skip_errors

    logger.info(f"开始转换LabelMe到COCO: {input_path} -> {output_path}")

    # 参数验证
    validate_convert_params("labelme", "coco", input_path, output_path, image_dir, None)

    # 加载类别映射
    category_mapping_dict = None
    if category_mapping:
        category_mapping = validate_path_exists(category_mapping, "类别映射文件")
        with open(category_mapping, "r", encoding="utf-8") as f:
            category_mapping_dict = json.load(f)

    # 调用现有API
    converter = CocoAndLabelMeConverter(source_to_target=False, verbose=verbose, strict_mode=strict)
    result = converter.convert(
        source_path=str(input_path),
        target_path=str(output_path),
        image_dir=str(image_dir) if image_dir else None,
        category_mapping=category_mapping_dict,
    )

    if result.success:
        logger.info(f"转换完成: {result.get_summary()}")
    else:
        # Use first error message if available, otherwise generic message
        error_msg = result.errors[0] if result.errors else "转换失败"
        logger.error(f"转换失败: {error_msg}")
        raise RuntimeCLIError(f"转换失败: {error_msg}")


def validate_path_exists(path: Path, name: str = "路径") -> Path:
    """验证路径是否存在"""
    if not path.exists():
        from dataflow.cli.exceptions import InputError
        raise InputError(f"{name}不存在: {path}")
    return path