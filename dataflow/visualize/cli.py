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
            overwrite=ctx.parent.obj.get('overwrite', False)
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
    @click.option('-v', '--verbose', is_flag=True, help='Print detailed progress information')
    @click.pass_context
    def command(ctx, image_dir, label_dir, class_path, save, segmentation, verbose):
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
                overwrite=ctx.parent.obj.get('overwrite', False)
            )

            # Create visualizer and perform visualization
            visualizer = YoloVisualizer(verbose=verbose, segmentation=segmentation)
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
    @click.option('-v', '--verbose', is_flag=True, help='Print detailed progress information')
    @click.pass_context
    def command(ctx, image_dir, annotation_json, save, segmentation, verbose):
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
                overwrite=ctx.parent.obj.get('overwrite', False)
            )

            # Create visualizer and perform visualization
            visualizer = CocoVisualizer(verbose=verbose, segmentation=segmentation)
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
    @click.option('-v', '--verbose', is_flag=True, help='Print detailed progress information')
    @click.pass_context
    def command(ctx, image_dir, label_dir, save, segmentation, verbose):
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
                overwrite=ctx.parent.obj.get('overwrite', False)
            )

            # Create visualizer and perform visualization
            visualizer = LabelMeVisualizer(verbose=verbose, segmentation=segmentation)
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

    # 提取通用字段
    summary_fields = [
        ("Image directory", "image_dir"),
        ("Label directory", "label_dir"),
        ("Annotation JSON", "annotation_json"),
        ("Class file", "class_path"),
        ("Total images", "total_images"),
        ("Images processed", "images_processed"),
        ("Images with annotations", "images_with_annotations"),
        ("Annotations processed", "annotations_processed"),
        ("Classes found", "classes_found"),
        ("Categories found", "categories_found"),
        ("Saved images", "saved_images"),
        ("Save directory", "save_dir"),
    ]

    for display_name, key in summary_fields:
        if key in result and result[key] is not None:
            click.echo(f"{display_name}: {result[key]}")

    click.echo(f"Segmentation mode: {'ON' if segmentation else 'OFF'}")
    click.echo("\n✅ Visualization completed successfully!")