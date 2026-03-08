# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:41
@File    : cli.py
@Author  : zj
@Description: Command-line interface for DataFlow-CV
"""

import os
import click
from dataflow.convert.coco_to_yolo import CocoToYoloConverter
from dataflow.convert.yolo_to_coco import YoloToCocoConverter
from dataflow.config import Config


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--overwrite', is_flag=True, help='Overwrite existing files')
@click.pass_context
def cli(ctx, verbose, overwrite):
    """DataFlow-CV: Computer vision dataset processing tool."""
    # Store configuration in context
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['overwrite'] = overwrite

    # Update global config based on CLI options
    if verbose:
        Config.VERBOSE = True
    if overwrite:
        Config.OVERWRITE_EXISTING = True

    if verbose:
        click.echo(f"Verbose mode: {'ON' if verbose else 'OFF'}")
        click.echo(f"Overwrite mode: {'ON' if overwrite else 'OFF'}")


@cli.group()
@click.pass_context
def convert(ctx):
    """Convert between different annotation formats."""
    pass


@convert.command(name='coco2yolo')
@click.argument('coco_json_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.option('--segmentation', '-s', is_flag=True, help='Handle segmentation annotations')
@click.pass_context
def coco2yolo(ctx, coco_json_path, output_dir, segmentation):
    """
    Convert COCO JSON to YOLO format.

    \b
    COCO_JSON_PATH: Path to COCO JSON annotation file
    OUTPUT_DIR: Directory where labels/ and class.names will be created
    """
    try:
        # Update config for segmentation if needed
        if segmentation:
            Config.YOLO_SEGMENTATION = True

        click.echo(f"Converting COCO JSON: {coco_json_path}")
        click.echo(f"Output directory: {output_dir}")

        # Create converter and perform conversion
        converter = CocoToYoloConverter(verbose=ctx.obj['verbose'])
        result = converter.convert(coco_json_path, output_dir)

        # Print summary
        click.echo("\n" + "="*50)
        click.echo("CONVERSION SUMMARY")
        click.echo("="*50)
        click.echo(f"COCO JSON: {result.get('coco_json_path')}")
        click.echo(f"Output directory: {result.get('output_dir')}")
        click.echo(f"Labels directory: {result.get('labels_dir')}")
        click.echo(f"Classes file: {result.get('classes_file')}")
        click.echo(f"Total images: {result.get('total_images', 0)}")
        click.echo(f"Images processed: {result.get('images_processed', 0)}")
        click.echo(f"Images with annotations: {result.get('images_with_annotations', 0)}")
        click.echo(f"Annotations processed: {result.get('annotations_processed', 0)}")
        click.echo(f"Total categories: {result.get('total_categories', 0)}")

        click.echo("\n✅ Conversion completed successfully!")

    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        ctx.exit(1)


@convert.command(name='yolo2coco')
@click.argument('image_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('yolo_labels_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('yolo_class_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('coco_json_path', type=click.Path())
@click.pass_context
def yolo2coco(ctx, image_dir, yolo_labels_dir, yolo_class_path, coco_json_path):
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
        converter = YoloToCocoConverter(verbose=ctx.obj['verbose'])
        result = converter.convert(
            image_dir, yolo_labels_dir, yolo_class_path, coco_json_path
        )

        # Print summary
        click.echo("\n" + "="*50)
        click.echo("CONVERSION SUMMARY")
        click.echo("="*50)
        click.echo(f"Image directory: {result.get('image_dir')}")
        click.echo(f"YOLO labels directory: {result.get('yolo_labels_dir')}")
        click.echo(f"YOLO classes file: {result.get('yolo_class_path')}")
        click.echo(f"COCO JSON: {result.get('coco_json_path')}")
        click.echo(f"Total images: {len(result.get('image_files', []))}")
        click.echo(f"Images processed: {result.get('images_processed', 0)}")
        click.echo(f"Images with annotations: {result.get('images_with_annotations', 0)}")
        click.echo(f"Annotations processed: {result.get('annotations_processed', 0)}")
        click.echo(f"Total classes: {result.get('total_classes', 0)}")
        click.echo(f"Images without labels: {result.get('images_without_labels', 0)}")

        click.echo("\n✅ Conversion completed successfully!")

    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.pass_context
def config(ctx):
    """Show current configuration."""
    click.echo("DataFlow-CV Configuration")
    click.echo("="*50)

    config_vars = [
        ("YOLO_CLASSES_FILENAME", Config.YOLO_CLASSES_FILENAME),
        ("YOLO_LABELS_DIRNAME", Config.YOLO_LABELS_DIRNAME),
        ("IMAGE_EXTENSIONS", Config.IMAGE_EXTENSIONS),
        ("YOLO_LABEL_EXTENSION", Config.YOLO_LABEL_EXTENSION),
        ("COCO_JSON_EXTENSION", Config.COCO_JSON_EXTENSION),
        ("OVERWRITE_EXISTING", Config.OVERWRITE_EXISTING),
        ("VERBOSE", Config.VERBOSE),
        ("CREATE_DIRS", Config.CREATE_DIRS),
        ("YOLO_NORMALIZE", Config.YOLO_NORMALIZE),
        ("YOLO_SEGMENTATION", Config.YOLO_SEGMENTATION),
    ]

    for name, value in config_vars:
        click.echo(f"{name:30} = {value}")

    click.echo("\nCLI Context:")
    for key, value in ctx.obj.items():
        click.echo(f"  {key}: {value}")


def main():
    """Main entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
