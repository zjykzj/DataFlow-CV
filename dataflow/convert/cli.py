# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/14
@File    : cli.py
@Author  : DataFlow Team
@Description: Convert module CLI for DataFlow-CV
"""

import os
import click
from . import (
    CocoToYoloConverter,
    YoloToCocoConverter,
    CocoToLabelMeConverter,
    LabelMeToCocoConverter,
    YoloToLabelMeConverter,
    LabelMeToYoloConverter
)
from .config import ConvertConfig


def create_convert_group():
    """创建convert命令组"""

    @click.group(name='convert')
    @click.pass_context
    def convert_group(ctx):
        """Convert between different annotation formats."""
        # 传递全局CLI选项到模块
        ConvertConfig.update_from_cli(
            verbose=ctx.parent.obj.get('verbose', False),
            overwrite=ctx.parent.obj.get('overwrite', False)
        )

    # 使用工厂函数创建子命令，避免过早绑定
    convert_group.add_command(_create_coco2yolo_command())
    convert_group.add_command(_create_yolo2coco_command())
    convert_group.add_command(_create_coco2labelme_command())
    convert_group.add_command(_create_labelme2coco_command())
    convert_group.add_command(_create_labelme2yolo_command())
    convert_group.add_command(_create_yolo2labelme_command())

    return convert_group


def _create_coco2yolo_command():
    """创建coco2yolo子命令"""
    @click.command(name='coco2yolo')
    @click.argument('coco_json_path', type=click.Path(exists=True, dir_okay=False))
    @click.argument('output_dir', type=click.Path(file_okay=False))
    @click.option('--segmentation', '-s', is_flag=True, help='Handle segmentation annotations')
    @click.pass_context
    def command(ctx, coco_json_path, output_dir, segmentation):
        """
        Convert COCO JSON to YOLO format.

        \b
        COCO_JSON_PATH: Path to COCO JSON annotation file
        OUTPUT_DIR: Directory where YOLO label files will be created (class.names will be auto-generated)
        """
        try:
            # Segmentation parameter is passed directly to converter

            click.echo(f"Converting COCO JSON: {coco_json_path}")
            click.echo(f"Output directory: {output_dir}")
            # Classes file will be auto-generated as {os.path.join(output_dir, ConvertConfig.YOLO_CLASSES_FILENAME)}

            # Create converter and perform conversion
            converter = CocoToYoloConverter(verbose=ctx.parent.obj.get('verbose', False))
            result = converter.convert(coco_json_path, output_dir, classes_path=None, segmentation=segmentation)

            # Print summary
            _print_conversion_summary(result, segmentation)

        except Exception as e:
            click.echo(f"\n❌ Error: {e}", err=True)
            ctx.exit(1)

    return command


def _create_yolo2coco_command():
    """创建yolo2coco子命令"""
    @click.command(name='yolo2coco')
    @click.argument('image_dir', type=click.Path(exists=True, file_okay=False))
    @click.argument('yolo_labels_dir', type=click.Path(exists=True, file_okay=False))
    @click.argument('yolo_class_path', type=click.Path(exists=True, dir_okay=False))
    @click.argument('coco_json_path', type=click.Path())
    @click.option('--segmentation', '-s', is_flag=True, help='Handle segmentation annotations')
    @click.pass_context
    def command(ctx, image_dir, yolo_labels_dir, yolo_class_path, coco_json_path, segmentation=False):
        """
        Convert YOLO format to COCO JSON.

        \b
        IMAGE_DIR: Directory containing image files
        YOLO_LABELS_DIR: Directory containing YOLO label files
        YOLO_CLASS_PATH: Path to YOLO class names file (e.g., class.names)
        COCO_JSON_PATH: Path to save COCO JSON file
        """
        try:
            click.echo(f"Image directory: {image_dir}")
            click.echo(f"YOLO labels directory: {yolo_labels_dir}")
            click.echo(f"YOLO classes file: {yolo_class_path}")
            click.echo(f"COCO JSON output: {coco_json_path}")

            # Create converter and perform conversion
            converter = YoloToCocoConverter(verbose=ctx.parent.obj.get('verbose', False))
            result = converter.convert(
                image_dir, yolo_labels_dir, yolo_class_path, coco_json_path, segmentation=segmentation
            )

            # Print summary
            _print_conversion_summary(result, segmentation)

        except Exception as e:
            click.echo(f"\n❌ Error: {e}", err=True)
            ctx.exit(1)

    return command


def _create_coco2labelme_command():
    """创建coco2labelme子命令"""
    @click.command(name='coco2labelme')
    @click.argument('coco_json_path', type=click.Path(exists=True, dir_okay=False))
    @click.argument('output_dir', type=click.Path(file_okay=False))
    @click.option('--segmentation', '-s', is_flag=True, help='Handle segmentation annotations')
    @click.pass_context
    def command(ctx, coco_json_path, output_dir, segmentation):
        """
        Convert COCO JSON to LabelMe format.

        \b
        COCO_JSON_PATH: Path to COCO JSON annotation file
        OUTPUT_DIR: Directory where LabelMe JSON files will be created
        """
        try:
            click.echo(f"Converting COCO JSON: {coco_json_path}")
            click.echo(f"Output directory: {output_dir}")
            if segmentation:
                click.echo("Segmentation mode: ON (strict)")

            # Create converter and perform conversion
            converter = CocoToLabelMeConverter(verbose=ctx.parent.obj.get('verbose', False))
            result = converter.convert(coco_json_path, output_dir, segmentation=segmentation)

            # Print summary
            _print_conversion_summary(result, segmentation)

        except Exception as e:
            click.echo(f"\n❌ Error: {e}", err=True)
            ctx.exit(1)

    return command


def _create_labelme2coco_command():
    """创建labelme2coco子命令"""
    @click.command(name='labelme2coco')
    @click.argument('label_dir', type=click.Path(exists=True, file_okay=False))
    @click.argument('classes_path', type=click.Path(exists=True, dir_okay=False))
    @click.argument('output_json_path', type=click.Path())
    @click.option('--segmentation', '-s', is_flag=True, help='Handle segmentation annotations')
    @click.pass_context
    def command(ctx, label_dir, classes_path, output_json_path, segmentation):
        """
        Convert LabelMe format to COCO JSON.

        \b
        LABEL_DIR: Directory containing LabelMe JSON files
        CLASSES_PATH: Path to class names file (e.g., class.names)
        OUTPUT_JSON_PATH: Path to save COCO JSON file
        """
        try:
            click.echo(f"Label directory: {label_dir}")
            click.echo(f"Classes file: {classes_path}")
            click.echo(f"COCO JSON output: {output_json_path}")
            if segmentation:
                click.echo("Segmentation mode: ON (strict)")

            # Create converter and perform conversion
            converter = LabelMeToCocoConverter(verbose=ctx.parent.obj.get('verbose', False))
            result = converter.convert(label_dir, classes_path, output_json_path, segmentation=segmentation)

            # Print summary
            _print_conversion_summary(result, segmentation)

        except Exception as e:
            click.echo(f"\n❌ Error: {e}", err=True)
            ctx.exit(1)

    return command


def _create_labelme2yolo_command():
    """创建labelme2yolo子命令"""
    @click.command(name='labelme2yolo')
    @click.argument('label_dir', type=click.Path(exists=True, file_okay=False))
    @click.argument('classes_path', type=click.Path(exists=True, dir_okay=False))
    @click.argument('output_dir', type=click.Path(file_okay=False))
    @click.option('--segmentation', '-s', is_flag=True, help='Handle segmentation annotations')
    @click.pass_context
    def command(ctx, label_dir, classes_path, output_dir, segmentation):
        """
        Convert LabelMe format to YOLO format.

        \b
        LABEL_DIR: Directory containing LabelMe JSON files
        CLASSES_PATH: Path to class names file (e.g., class.names)
        OUTPUT_DIR: Directory where YOLO label files will be created
        """
        try:
            click.echo(f"Label directory: {label_dir}")
            click.echo(f"Classes file: {classes_path}")
            click.echo(f"Output directory: {output_dir}")
            if segmentation:
                click.echo("Segmentation mode: ON (strict)")

            # Create converter and perform conversion
            converter = LabelMeToYoloConverter(verbose=ctx.parent.obj.get('verbose', False))
            result = converter.convert(label_dir, classes_path, output_dir, segmentation=segmentation)

            # Print summary
            _print_conversion_summary(result, segmentation)

        except Exception as e:
            click.echo(f"\n❌ Error: {e}", err=True)
            ctx.exit(1)

    return command


def _create_yolo2labelme_command():
    """创建yolo2labelme子命令"""
    @click.command(name='yolo2labelme')
    @click.argument('image_dir', type=click.Path(exists=True, file_okay=False))
    @click.argument('label_dir', type=click.Path(exists=True, file_okay=False))
    @click.argument('classes_path', type=click.Path(exists=True, dir_okay=False))
    @click.argument('output_dir', type=click.Path(file_okay=False))
    @click.option('--segmentation', '-s', is_flag=True, help='Handle segmentation annotations')
    @click.pass_context
    def command(ctx, image_dir, label_dir, classes_path, output_dir, segmentation):
        """
        Convert YOLO format to LabelMe format.

        \b
        IMAGE_DIR: Directory containing image files
        LABEL_DIR: Directory containing YOLO label files (.txt)
        CLASSES_PATH: Path to YOLO class names file (e.g., class.names)
        OUTPUT_DIR: Directory where LabelMe JSON files will be created
        """
        try:
            click.echo(f"Image directory: {image_dir}")
            click.echo(f"Label directory: {label_dir}")
            click.echo(f"Classes file: {classes_path}")
            click.echo(f"Output directory: {output_dir}")
            if segmentation:
                click.echo("Segmentation mode: ON (strict)")

            # Create converter and perform conversion
            converter = YoloToLabelMeConverter(verbose=ctx.parent.obj.get('verbose', False))
            result = converter.convert(image_dir, label_dir, classes_path, output_dir, segmentation=segmentation)

            # Print summary
            _print_conversion_summary(result, segmentation)

        except Exception as e:
            click.echo(f"\n❌ Error: {e}", err=True)
            ctx.exit(1)

    return command


def _print_conversion_summary(result, segmentation):
    """打印转换结果摘要"""
    click.echo("\n" + "="*50)
    click.echo("CONVERSION SUMMARY")
    click.echo("="*50)

    # 提取通用字段
    summary_fields = [
        ("COCO JSON", "coco_json_path"),
        ("Classes file", "classes_file"),
        ("Output directory", "output_dir"),
        ("Image directory", "image_dir"),
        ("Label directory", "label_dir"),
        ("YOLO labels directory", "yolo_labels_dir"),
        ("YOLO classes file", "yolo_class_path"),
        ("Images processed", "images_processed"),
        ("Annotations processed", "annotations_processed"),
        ("Categories found", "categories_found"),
        ("Categories in classes file", "categories_found"),
        ("Categories in data", "categories_in_data"),
    ]

    for display_name, key in summary_fields:
        if key in result and result[key] is not None:
            click.echo(f"{display_name}: {result[key]}")

    click.echo(f"Segmentation mode: {'ON' if segmentation else 'OFF'}")
    click.echo("\n✅ Conversion completed successfully!")