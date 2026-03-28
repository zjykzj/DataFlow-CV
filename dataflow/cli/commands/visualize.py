"""Visualization commands."""

import click
from pathlib import Path
from typing import Optional

from dataflow.cli.commands.utils import validate_visualize_params, add_common_options
from dataflow.cli.exceptions import RuntimeCLIError


@click.group(name="visualize")
def visualize_group():
    """可视化命令组 - 支持多种标签格式的可视化"""
    pass


@visualize_group.command()
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="图像文件目录（如与标签文件分离）",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="可视化结果保存目录",
)
@click.option(
    "--class-file",
    "-c",
    type=click.Path(path_type=Path),
    help="类别文件路径（YOLO格式需要）",
)
@click.option("--display", "-d", is_flag=True, help="交互式显示结果")
@click.option(
    "--save", "-s", is_flag=True, default=True, help="保存可视化结果到文件"
)
@click.option(
    "--color-scheme",
    type=click.Choice(["random", "category", "consistent"]),
    default="random",
    help="颜色方案：random/category/consistent",
)
@click.option("--thickness", type=int, default=2, help="边界框/多边形线宽")
@click.pass_context
def yolo(
    ctx,
    input_path: Path,
    image_dir: Optional[Path],
    output_dir: Optional[Path],
    class_file: Optional[Path],
    display: bool,
    save: bool,
    color_scheme: str,
    thickness: int,
):
    """可视化YOLO格式标签"""
    from dataflow.visualize.yolo_visualizer import YOLOVisualizer

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"]

    logger.info(f"开始可视化YOLO标签: {input_path}")

    # 参数验证
    validate_visualize_params(input_path, image_dir, output_dir)
    if class_file is None:
        from dataflow.cli.exceptions import ParameterError
        raise ParameterError("YOLO格式需要--class-file参数")

    # 调用现有API
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
        logger.info(f"可视化完成: 处理了{result.data.get('processed_images', 0)}张图像")
    else:
        logger.error(f"可视化失败: {result.message}")
        raise RuntimeCLIError(f"可视化失败: {result.message}")


@visualize_group.command()
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="图像文件目录（如与标签文件分离）",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="可视化结果保存目录",
)
@click.option("--display", "-d", is_flag=True, help="交互式显示结果")
@click.option(
    "--save", "-s", is_flag=True, default=True, help="保存可视化结果到文件"
)
@click.option(
    "--color-scheme",
    type=click.Choice(["random", "category", "consistent"]),
    default="random",
    help="颜色方案：random/category/consistent",
)
@click.option("--thickness", type=int, default=2, help="边界框/多边形线宽")
@click.pass_context
def coco(
    ctx,
    input_path: Path,
    image_dir: Optional[Path],
    output_dir: Optional[Path],
    display: bool,
    save: bool,
    color_scheme: str,
    thickness: int,
):
    """可视化COCO格式标签"""
    from dataflow.visualize.coco_visualizer import COCOVisualizer

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"]

    logger.info(f"开始可视化COCO标签: {input_path}")

    # 参数验证
    validate_visualize_params(input_path, image_dir, output_dir)

    # 调用现有API
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
        logger.info(f"可视化完成: 处理了{result.data.get('processed_images', 0)}张图像")
    else:
        logger.error(f"可视化失败: {result.message}")
        raise RuntimeCLIError(f"可视化失败: {result.message}")


@visualize_group.command()
@add_common_options
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--image-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="图像文件目录（如与标签文件分离）",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="可视化结果保存目录",
)
@click.option("--display", "-d", is_flag=True, help="交互式显示结果")
@click.option(
    "--save", "-s", is_flag=True, default=True, help="保存可视化结果到文件"
)
@click.option(
    "--color-scheme",
    type=click.Choice(["random", "category", "consistent"]),
    default="random",
    help="颜色方案：random/category/consistent",
)
@click.option("--thickness", type=int, default=2, help="边界框/多边形线宽")
@click.pass_context
def labelme(
    ctx,
    input_path: Path,
    image_dir: Optional[Path],
    output_dir: Optional[Path],
    display: bool,
    save: bool,
    color_scheme: str,
    thickness: int,
):
    """可视化LabelMe格式标签"""
    from dataflow.visualize.labelme_visualizer import LabelMeVisualizer

    logger = ctx.obj["logger"]
    verbose = ctx.obj["verbose"]
    strict = ctx.obj["strict"]

    logger.info(f"开始可视化LabelMe标签: {input_path}")

    # 参数验证
    validate_visualize_params(input_path, image_dir, output_dir)

    # 调用现有API
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
        logger.info(f"可视化完成: 处理了{result.data.get('processed_images', 0)}张图像")
    else:
        logger.error(f"可视化失败: {result.message}")
        raise RuntimeCLIError(f"可视化失败: {result.message}")