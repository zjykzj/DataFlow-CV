# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 20:41
@File    : cli.py
@Author  : zj
@Description: Command-line interface for DataFlow-CV
"""

import os
import click
from dataflow.convert import (
    CocoToYoloConverter,
    YoloToCocoConverter,
    CocoToLabelMeConverter,
    LabelMeToCocoConverter,
    YoloToLabelMeConverter,
    LabelMeToYoloConverter
)
from dataflow.visualize.yolo import YoloVisualizer
from dataflow.visualize.coco import CocoVisualizer
from dataflow.visualize.labelme import LabelMeVisualizer
from dataflow.config import Config
from dataflow import __version__


@click.group(context_settings={'help_option_names': ['-h', '--help']}, invoke_without_command=True)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--overwrite', is_flag=True, help='Overwrite existing files')
@click.version_option(version=__version__, prog_name='DataFlow-CV')
@click.pass_context
def cli(ctx, verbose, overwrite):
    """DataFlow-CV: Computer vision dataset processing tool."""
    # If -v is used alone (no subcommand), show version and exit
    if verbose and ctx.invoked_subcommand is None:
        click.echo(f"DataFlow-CV, version {__version__}")
        ctx.exit()

    # Store configuration in context
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['overwrite'] = overwrite

    # Update global config based on CLI options
    if verbose:
        Config.VERBOSE = True
    if overwrite:
        Config.OVERWRITE_EXISTING = True

    # If no subcommand was invoked, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        return

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
        # Segmentation parameter is passed directly to converter

        click.echo(f"Converting COCO JSON: {coco_json_path}")
        click.echo(f"Output directory: {output_dir}")

        # Create converter and perform conversion
        converter = CocoToYoloConverter(verbose=ctx.obj['verbose'])
        result = converter.convert(coco_json_path, output_dir, segmentation=segmentation)

        # Print summary
        click.echo("\n" + "="*50)
        click.echo("CONVERSION SUMMARY")
        click.echo("="*50)
        click.echo(f"COCO JSON: {coco_json_path}")
        click.echo(f"Output directory: {result.get('output_dir')}")
        click.echo(f"Labels directory: {result.get('labels_dir')}")
        click.echo(f"Classes file: {result.get('classes_file')}")
        click.echo(f"Images processed: {result.get('images_processed', 0)}")
        click.echo(f"Annotations processed: {result.get('annotations_processed', 0)}")
        click.echo(f"Categories found: {result.get('categories_found', 0)}")
        click.echo(f"Segmentation mode: {'ON' if segmentation else 'OFF'}")

        click.echo("\n✅ Conversion completed successfully!")

    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        ctx.exit(1)


@convert.command(name='yolo2coco')
@click.argument('image_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('yolo_labels_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('yolo_class_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('coco_json_path', type=click.Path())
@click.option('--segmentation', '-s', is_flag=True, help='Handle segmentation annotations')
@click.pass_context
def yolo2coco(ctx, image_dir, yolo_labels_dir, yolo_class_path, coco_json_path, segmentation=False):
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
            image_dir, yolo_labels_dir, yolo_class_path, coco_json_path, segmentation=segmentation
        )

        # Print summary
        click.echo("\n" + "="*50)
        click.echo("CONVERSION SUMMARY")
        click.echo("="*50)
        click.echo(f"Image directory: {result.get('image_dir')}")
        click.echo(f"YOLO labels directory: {result.get('label_dir')}")
        click.echo(f"YOLO classes file: {result.get('classes_file')}")
        click.echo(f"COCO JSON: {result.get('coco_json_path')}")
        click.echo(f"Images processed: {result.get('images_processed', 0)}")
        click.echo(f"Annotations processed: {result.get('annotations_processed', 0)}")
        click.echo(f"Categories found: {result.get('categories_found', 0)}")
        click.echo(f"Segmentation mode: {'ON' if segmentation else 'OFF'}")

        click.echo("\n✅ Conversion completed successfully!")

    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        ctx.exit(1)


@convert.command(name='coco2labelme')
@click.argument('coco_json_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.option('--segmentation', '-s', is_flag=True, help='Handle segmentation annotations')
@click.pass_context
def coco2labelme(ctx, coco_json_path, output_dir, segmentation):
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
        converter = CocoToLabelMeConverter(verbose=ctx.obj['verbose'])
        result = converter.convert(coco_json_path, output_dir, segmentation=segmentation)

        # Print summary
        click.echo("\n" + "="*50)
        click.echo("CONVERSION SUMMARY")
        click.echo("="*50)
        click.echo(f"COCO JSON: {coco_json_path}")
        click.echo(f"Output directory: {result.get('output_dir')}")
        click.echo(f"Images processed: {result.get('images_processed', 0)}")
        click.echo(f"Annotations processed: {result.get('annotations_processed', 0)}")
        click.echo(f"Categories found: {result.get('categories_found', 0)}")
        click.echo(f"Classes file: {result.get('classes_file', 'Not created')}")
        click.echo(f"Segmentation mode: {'ON' if segmentation else 'OFF'}")

        click.echo("\n✅ Conversion completed successfully!")

    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        ctx.exit(1)


@convert.command(name='labelme2coco')
@click.argument('label_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('classes_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_json_path', type=click.Path())
@click.option('--segmentation', '-s', is_flag=True, help='Handle segmentation annotations')
@click.pass_context
def labelme2coco(ctx, label_dir, classes_path, output_json_path, segmentation):
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
        converter = LabelMeToCocoConverter(verbose=ctx.obj['verbose'])
        result = converter.convert(label_dir, classes_path, output_json_path, segmentation=segmentation)

        # Print summary
        click.echo("\n" + "="*50)
        click.echo("CONVERSION SUMMARY")
        click.echo("="*50)
        click.echo(f"Label directory: {result.get('label_dir')}")
        click.echo(f"Classes file: {result.get('classes_file')}")
        click.echo(f"COCO JSON: {result.get('coco_json_path')}")
        click.echo(f"Images processed: {result.get('images_processed', 0)}")
        click.echo(f"Annotations processed: {result.get('annotations_processed', 0)}")
        click.echo(f"Categories found: {result.get('categories_found', 0)}")
        click.echo(f"Segmentation mode: {'ON' if segmentation else 'OFF'}")

        click.echo("\n✅ Conversion completed successfully!")

    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        ctx.exit(1)


@convert.command(name='labelme2yolo')
@click.argument('label_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.option('--segmentation', '-s', is_flag=True, help='Handle segmentation annotations')
@click.pass_context
def labelme2yolo(ctx, label_dir, output_dir, segmentation):
    """
    Convert LabelMe format to YOLO format.

    \b
    LABEL_DIR: Directory containing LabelMe JSON files
    OUTPUT_DIR: Directory where labels/ and class.names will be created
    """
    try:
        click.echo(f"Label directory: {label_dir}")
        click.echo(f"Output directory: {output_dir}")
        if segmentation:
            click.echo("Segmentation mode: ON (strict)")

        # Create converter and perform conversion
        converter = LabelMeToYoloConverter(verbose=ctx.obj['verbose'])
        result = converter.convert(label_dir, output_dir, segmentation=segmentation)

        # Print summary
        click.echo("\n" + "="*50)
        click.echo("CONVERSION SUMMARY")
        click.echo("="*50)
        click.echo(f"Label directory: {result.get('label_dir')}")
        click.echo(f"Output directory: {result.get('output_dir')}")
        click.echo(f"Labels directory: {result.get('labels_dir')}")
        click.echo(f"Classes file: {result.get('classes_file', 'Not created')}")
        click.echo(f"Images processed: {result.get('images_processed', 0)}")
        click.echo(f"Annotations processed: {result.get('annotations_processed', 0)}")
        click.echo(f"Categories found: {result.get('categories_found', 0)}")
        click.echo(f"Segmentation mode: {'ON' if segmentation else 'OFF'}")

        click.echo("\n✅ Conversion completed successfully!")

    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        ctx.exit(1)


@convert.command(name='yolo2labelme')
@click.argument('image_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('label_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('classes_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.option('--segmentation', '-s', is_flag=True, help='Handle segmentation annotations')
@click.pass_context
def yolo2labelme(ctx, image_dir, label_dir, classes_path, output_dir, segmentation):
    """
    Convert YOLO format to LabelMe format.

    \b
    IMAGE_DIR: Directory containing image files
    LABEL_DIR: Directory containing YOLO label files
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
        converter = YoloToLabelMeConverter(verbose=ctx.obj['verbose'])
        result = converter.convert(image_dir, label_dir, classes_path, output_dir, segmentation=segmentation)

        # Print summary
        click.echo("\n" + "="*50)
        click.echo("CONVERSION SUMMARY")
        click.echo("="*50)
        click.echo(f"Image directory: {result.get('image_dir')}")
        click.echo(f"Label directory: {result.get('label_dir')}")
        click.echo(f"Classes file: {result.get('classes_file')}")
        click.echo(f"Output directory: {result.get('output_dir')}")
        click.echo(f"Images processed: {result.get('images_processed', 0)}")
        click.echo(f"Annotations processed: {result.get('annotations_processed', 0)}")
        click.echo(f"Segmentation mode: {'ON' if segmentation else 'OFF'}")

        click.echo("\n✅ Conversion completed successfully!")

    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        ctx.exit(1)


@cli.group()
@click.pass_context
def visualize(ctx):
    """Visualize annotations on images."""
    pass


@visualize.command(name='yolo')
@click.argument('image_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('label_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('class_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--save', type=click.Path(file_okay=False), help='Directory to save visualized images')
@click.option('--segmentation', '-s', is_flag=True, help='Force segmentation mode (strict validation)')
@click.pass_context
def visualize_yolo(ctx, image_dir, label_dir, class_path, save, segmentation):
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

        # Create visualizer and perform visualization
        visualizer = YoloVisualizer(verbose=ctx.obj['verbose'], segmentation=segmentation)
        result = visualizer.visualize(image_dir, label_dir, class_path, save)

        # Print summary
        click.echo("\n" + "="*50)
        click.echo("VISUALIZATION SUMMARY")
        click.echo("="*50)
        click.echo(f"Image directory: {result.get('image_dir')}")
        click.echo(f"Label directory: {result.get('label_dir')}")
        click.echo(f"Class file: {result.get('class_path')}")
        click.echo(f"Total images: {result.get('total_images', 0)}")
        click.echo(f"Images processed: {result.get('images_processed', 0)}")
        click.echo(f"Images with annotations: {result.get('images_with_annotations', 0)}")
        click.echo(f"Annotations processed: {result.get('annotations_processed', 0)}")
        click.echo(f"Classes found: {result.get('classes_found', [])}")
        if save:
            click.echo(f"Saved images: {result.get('saved_images', 0)}")
            click.echo(f"Save directory: {result.get('save_dir')}")

        click.echo("\n✅ Visualization completed successfully!")

    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        ctx.exit(1)


@visualize.command(name='coco')
@click.argument('image_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('annotation_json', type=click.Path(exists=True, dir_okay=False))
@click.option('--save', type=click.Path(file_okay=False), help='Directory to save visualized images')
@click.option('--segmentation', '-s', is_flag=True, help='Force segmentation mode (strict validation)')
@click.pass_context
def visualize_coco(ctx, image_dir, annotation_json, save, segmentation):
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

        # Create visualizer and perform visualization
        visualizer = CocoVisualizer(verbose=ctx.obj['verbose'], segmentation=segmentation)
        result = visualizer.visualize(image_dir, annotation_json, save)

        # Print summary
        click.echo("\n" + "="*50)
        click.echo("VISUALIZATION SUMMARY")
        click.echo("="*50)
        click.echo(f"Image directory: {result.get('image_dir')}")
        click.echo(f"Annotation JSON: {result.get('annotation_json')}")
        click.echo(f"Total images: {result.get('total_images', 0)}")
        click.echo(f"Images processed: {result.get('images_processed', 0)}")
        click.echo(f"Images with annotations: {result.get('images_with_annotations', 0)}")
        click.echo(f"Annotations processed: {result.get('annotations_processed', 0)}")
        click.echo(f"Categories found: {result.get('categories_found', [])}")
        if save:
            click.echo(f"Saved images: {result.get('saved_images', 0)}")
            click.echo(f"Save directory: {result.get('save_dir')}")

        click.echo("\n✅ Visualization completed successfully!")

    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        ctx.exit(1)


@visualize.command(name='labelme')
@click.argument('image_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('label_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--save', type=click.Path(file_okay=False), help='Directory to save visualized images')
@click.option('--segmentation', '-s', is_flag=True, help='Force segmentation mode (strict validation)')
@click.pass_context
def visualize_labelme(ctx, image_dir, label_dir, save, segmentation):
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

        # Create visualizer and perform visualization
        visualizer = LabelMeVisualizer(verbose=ctx.obj['verbose'], segmentation=segmentation)
        result = visualizer.visualize(image_dir, label_dir, save)

        # Print summary
        click.echo("\n" + "="*50)
        click.echo("VISUALIZATION SUMMARY")
        click.echo("="*50)
        click.echo(f"Image directory: {result.get('image_dir')}")
        click.echo(f"Label directory: {result.get('label_dir')}")
        click.echo(f"Total images: {result.get('total_images', 0)}")
        click.echo(f"Images processed: {result.get('images_processed', 0)}")
        click.echo(f"Images with annotations: {result.get('images_with_annotations', 0)}")
        click.echo(f"Annotations processed: {result.get('annotations_processed', 0)}")
        click.echo(f"Classes found: {result.get('classes_found', [])}")
        if save:
            click.echo(f"Saved images: {result.get('saved_images', 0)}")
            click.echo(f"Save directory: {result.get('save_dir')}")

        click.echo("\n✅ Visualization completed successfully!")

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
