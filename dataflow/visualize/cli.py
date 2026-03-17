# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/14
@File    : cli.py
@Author  : DataFlow Team
@Description: Visualize module CLI for DataFlow-CV
"""

import click
from . import YoloVisualizer, CocoVisualizer, LabelMeVisualizer
from .config import VisualizeConfig


def create_visualize_group():
    """创建visualize命令组"""

    @click.group(name='visualize')
    @click.pass_context
    def visualize_group(ctx):
        """Visualize annotations on images."""
        # 传递全局CLI选项到模块
        VisualizeConfig.update_from_cli(
            verbose=ctx.parent.obj.get('verbose', False),
            overwrite=False
        )

    visualize_group.add_command(_create_yolo_command())
    visualize_group.add_command(_create_coco_command())
    visualize_group.add_command(_create_labelme_command())

    return visualize_group


def _create_yolo_command():
    """创建yolo可视化子命令"""
    @click.command(name='yolo')
    @click.argument('image_dir', type=click.Path(exists=True, file_okay=False))
    @click.argument('label_dir', type=click.Path(exists=True, file_okay=False))
    @click.argument('class_path', type=click.Path(exists=True, dir_okay=False))
    @click.option('--save', type=click.Path(file_okay=False), help='Directory to save visualized images')
    @click.option('--segmentation', '-s', is_flag=True, help='Force segmentation mode (strict validation)')
    @click.option('--fill/--no-fill', default=None, help='Fill polygons (default: False)')
    @click.option('--fill-alpha', type=float, default=None, help='Fill transparency (0.0-1.0, default: 0.3)')
    @click.option('--outline-alpha', type=float, default=None, help='Outline transparency (0.0-1.0, default: 1.0)')
    @click.option('--highlight-rle/--no-highlight-rle', default=None, help='Highlight RLE masks (default: True)')
    @click.option('--rle-color', type=str, default=None, help='RLE fill color as "R,G,B" (e.g., "255,0,0")')
    @click.option('-v', '--verbose', is_flag=True, help='Print detailed progress information')
    @click.pass_context
    def command(ctx, image_dir, label_dir, class_path, save, segmentation, fill, fill_alpha, outline_alpha, highlight_rle, rle_color, verbose):
        """
        Visualize YOLO format annotations.

        \b
        IMAGE_DIR: Directory containing image files
        LABEL_DIR: Directory containing YOLO label files
        CLASS_PATH: Path to class names file (e.g., class.names)
        """
        try:
            click.echo(f"Image directory: {image_dir}")
            click.echo(f"Label directory: {label_dir}")
            click.echo(f"Class file: {class_path}")
            if save:
                click.echo(f"Save directory: {save}")
            if segmentation:
                click.echo("Segmentation mode: ON (strict)")

            # Update config with local verbose flag
            VisualizeConfig.update_from_cli(
                verbose=verbose,
                overwrite=False
            )

            # Create visualizer and perform visualization
            visualizer = YoloVisualizer(verbose=verbose, segmentation=segmentation)

            # Set polygon fill and transparency options
            if fill is not None:
                visualizer.fill_polygons = fill
            if fill_alpha is not None:
                if not 0.0 <= fill_alpha <= 1.0:
                    raise ValueError("fill_alpha must be between 0.0 and 1.0")
                visualizer.fill_alpha = fill_alpha
            if outline_alpha is not None:
                if not 0.0 <= outline_alpha <= 1.0:
                    raise ValueError("outline_alpha must be between 0.0 and 1.0")
                visualizer.outline_alpha = outline_alpha
            if highlight_rle is not None:
                visualizer.highlight_rle = highlight_rle
            if rle_color is not None:
                # Parse R,G,B string
                try:
                    parts = rle_color.split(',')
                    if len(parts) != 3:
                        raise ValueError
                    r, g, b = map(int, parts)
                    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                        raise ValueError
                    visualizer.rle_fill_color = (b, g, r)  # OpenCV uses BGR
                except ValueError:
                    raise ValueError('rle_color must be "R,G,B" with integers 0-255')

            result = visualizer.visualize(image_dir, label_dir, class_path, save)

            _print_visualization_summary(result, segmentation)

        except Exception as e:
            click.echo(f"\n❌ Error: {e}", err=True)
            ctx.exit(1)

    return command


def _create_coco_command():
    """创建coco可视化子命令"""
    @click.command(name='coco')
    @click.argument('image_dir', type=click.Path(exists=True, file_okay=False))
    @click.argument('annotation_json', type=click.Path(exists=True, dir_okay=False))
    @click.option('--save', type=click.Path(file_okay=False), help='Directory to save visualized images')
    @click.option('--segmentation', '-s', is_flag=True, help='Force segmentation mode (strict validation)')
    @click.option('--fill/--no-fill', default=None, help='Fill polygons (default: False)')
    @click.option('--fill-alpha', type=float, default=None, help='Fill transparency (0.0-1.0, default: 0.3)')
    @click.option('--outline-alpha', type=float, default=None, help='Outline transparency (0.0-1.0, default: 1.0)')
    @click.option('--highlight-rle/--no-highlight-rle', default=None, help='Highlight RLE masks (default: True)')
    @click.option('--rle-color', type=str, default=None, help='RLE fill color as "R,G,B" (e.g., "255,0,0")')
    @click.option('-v', '--verbose', is_flag=True, help='Print detailed progress information')
    @click.pass_context
    def command(ctx, image_dir, annotation_json, save, segmentation, fill, fill_alpha, outline_alpha, highlight_rle, rle_color, verbose):
        """
        Visualize COCO format annotations.

        \b
        IMAGE_DIR: Directory containing image files
        ANNOTATION_JSON: Path to COCO JSON annotation file
        """
        try:
            click.echo(f"Image directory: {image_dir}")
            click.echo(f"Annotation JSON: {annotation_json}")
            if save:
                click.echo(f"Save directory: {save}")
            if segmentation:
                click.echo("Segmentation mode: ON (strict)")

            # Update config with local verbose flag
            VisualizeConfig.update_from_cli(
                verbose=verbose,
                overwrite=False
            )

            # Create visualizer and perform visualization
            visualizer = CocoVisualizer(verbose=verbose, segmentation=segmentation)

            # Set polygon fill and transparency options
            if fill is not None:
                visualizer.fill_polygons = fill
            if fill_alpha is not None:
                if not 0.0 <= fill_alpha <= 1.0:
                    raise ValueError("fill_alpha must be between 0.0 and 1.0")
                visualizer.fill_alpha = fill_alpha
            if outline_alpha is not None:
                if not 0.0 <= outline_alpha <= 1.0:
                    raise ValueError("outline_alpha must be between 0.0 and 1.0")
                visualizer.outline_alpha = outline_alpha
            if highlight_rle is not None:
                visualizer.highlight_rle = highlight_rle
            if rle_color is not None:
                # Parse R,G,B string
                try:
                    parts = rle_color.split(',')
                    if len(parts) != 3:
                        raise ValueError
                    r, g, b = map(int, parts)
                    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                        raise ValueError
                    visualizer.rle_fill_color = (b, g, r)  # OpenCV uses BGR
                except ValueError:
                    raise ValueError('rle_color must be "R,G,B" with integers 0-255')

            result = visualizer.visualize(image_dir, annotation_json, save)

            _print_visualization_summary(result, segmentation)

        except Exception as e:
            click.echo(f"\n❌ Error: {e}", err=True)
            ctx.exit(1)

    return command


def _create_labelme_command():
    """创建labelme可视化子命令"""
    @click.command(name='labelme')
    @click.argument('image_dir', type=click.Path(exists=True, file_okay=False))
    @click.argument('label_dir', type=click.Path(exists=True, file_okay=False))
    @click.option('--save', type=click.Path(file_okay=False), help='Directory to save visualized images')
    @click.option('--segmentation', '-s', is_flag=True, help='Force segmentation mode (strict validation)')
    @click.option('--fill/--no-fill', default=None, help='Fill polygons (default: False)')
    @click.option('--fill-alpha', type=float, default=None, help='Fill transparency (0.0-1.0, default: 0.3)')
    @click.option('--outline-alpha', type=float, default=None, help='Outline transparency (0.0-1.0, default: 1.0)')
    @click.option('--highlight-rle/--no-highlight-rle', default=None, help='Highlight RLE masks (default: True)')
    @click.option('--rle-color', type=str, default=None, help='RLE fill color as "R,G,B" (e.g., "255,0,0")')
    @click.option('-v', '--verbose', is_flag=True, help='Print detailed progress information')
    @click.pass_context
    def command(ctx, image_dir, label_dir, save, segmentation, fill, fill_alpha, outline_alpha, highlight_rle, rle_color, verbose):
        """
        Visualize LabelMe format annotations.

        \b
        IMAGE_DIR: Directory containing image files
        LABEL_DIR: Directory containing LabelMe JSON files
        """
        try:
            click.echo(f"Image directory: {image_dir}")
            click.echo(f"Label directory: {label_dir}")
            if save:
                click.echo(f"Save directory: {save}")
            if segmentation:
                click.echo("Segmentation mode: ON (strict)")

            # Update config with local verbose flag
            VisualizeConfig.update_from_cli(
                verbose=verbose,
                overwrite=False
            )

            # Create visualizer and perform visualization
            visualizer = LabelMeVisualizer(verbose=verbose, segmentation=segmentation)

            # Set polygon fill and transparency options
            if fill is not None:
                visualizer.fill_polygons = fill
            if fill_alpha is not None:
                if not 0.0 <= fill_alpha <= 1.0:
                    raise ValueError("fill_alpha must be between 0.0 and 1.0")
                visualizer.fill_alpha = fill_alpha
            if outline_alpha is not None:
                if not 0.0 <= outline_alpha <= 1.0:
                    raise ValueError("outline_alpha must be between 0.0 and 1.0")
                visualizer.outline_alpha = outline_alpha
            if highlight_rle is not None:
                visualizer.highlight_rle = highlight_rle
            if rle_color is not None:
                # Parse R,G,B string
                try:
                    parts = rle_color.split(',')
                    if len(parts) != 3:
                        raise ValueError
                    r, g, b = map(int, parts)
                    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                        raise ValueError
                    visualizer.rle_fill_color = (b, g, r)  # OpenCV uses BGR
                except ValueError:
                    raise ValueError('rle_color must be "R,G,B" with integers 0-255')

            result = visualizer.visualize(image_dir, label_dir, save)

            _print_visualization_summary(result, segmentation)

        except Exception as e:
            click.echo(f"\n❌ Error: {e}", err=True)
            ctx.exit(1)

    return command


def _print_visualization_summary(result, segmentation):
    """打印可视化结果摘要"""
    click.echo("\n" + "="*50)
    click.echo("VISUALIZATION SUMMARY")
    click.echo("="*50)

    # 输入信息组
    click.echo("\n📁 INPUT INFORMATION")
    input_fields = [
        ("Image directory", "image_dir"),
        ("Label directory", "label_dir"),
        ("Annotation JSON", "annotation_json"),
        ("Class file", "class_path"),
    ]
    for display_name, key in input_fields:
        if key in result and result[key] is not None:
            click.echo(f"  {display_name}: {result[key]}")

    # 处理统计组
    click.echo("\n📊 PROCESSING STATISTICS")
    stats_fields = [
        ("Total images", "total_images"),
        ("Images processed", "images_processed"),
        ("Images with annotations", "images_with_annotations"),
        ("Annotations processed", "annotations_processed"),
    ]
    for display_name, key in stats_fields:
        if key in result and result[key] is not None:
            click.echo(f"  {display_name}: {result[key]}")

    # 类别信息组
    if "classes_found" in result and result["classes_found"]:
        click.echo(f"  Classes found: {result['classes_found']}")
    if "categories_found" in result and result["categories_found"]:
        click.echo(f"  Categories found: {result['categories_found']}")

    # 输出信息组（如果保存了图像）
    if "save_dir" in result and result["save_dir"]:
        click.echo("\n💾 OUTPUT INFORMATION")
        click.echo(f"  Save directory: {result['save_dir']}")
        if "saved_images" in result:
            click.echo(f"  Saved images: {result['saved_images']}")

    # 配置信息组
    click.echo("\n⚙️ CONFIGURATION")
    click.echo(f"  Segmentation mode: {'ON (strict)' if segmentation else 'OFF'}")

    click.echo("\n✅ Visualization completed successfully!")