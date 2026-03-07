"""
Command-line interface for DataFlow.
"""

import os
# Set Qt environment variables before any imports to suppress warnings
os.environ.setdefault('QT_QPA_FONTDIR', '/usr/share/fonts')
os.environ.setdefault('QT_LOGGING_RULES', '*.debug=false;qt.*=false')

import click
import sys
from pathlib import Path

# Import modules
try:
    from . import convert
    from . import visualize
    from .config import get_config
    from . import __version__
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're in the correct environment and dependencies are installed.")
    sys.exit(1)


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"dataflow-cv version {__version__}")
    ctx.exit()


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--version', '-v', is_flag=True, callback=print_version,
              expose_value=False, is_eager=True, help='Show the version and exit.')
def main():
    """
    DataFlow - A data processing library for computer vision datasets.

    Supports format conversion and visualization for LabelMe, COCO, and YOLO formats.
    """
    pass


# Conversion commands
@main.group(context_settings={'help_option_names': ['-h', '--help']})
def convert():
    """Convert between different annotation formats."""
    pass


@convert.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('coco_json_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--class-names', type=click.Path(exists=True),
              help='Path to class names file (one per line)')
@click.option('--batch', is_flag=True, help='Batch mode: process directories instead of single files')
def coco2yolo(image_path, coco_json_path, output_path, class_names, batch):
    """Convert COCO annotation(s) to YOLO format.

    In batch mode (--batch):
      - Creates separate YOLO files for each image (one .txt per image)
    """
    try:
        if class_names:
            with open(class_names, 'r') as f:
                classes = [line.strip() for line in f if line.strip()]
        else:
            classes = []

        from pathlib import Path

        if batch:
            # Batch mode: process directories
            from .visualize.batch import find_matching_pairs, validate_batch_directories
            from .convert.coco_to_yolo import coco_to_yolo, batch_coco_to_yolo

            # Validate directories
            validate_batch_directories(image_path, coco_json_path)

            # Find matching pairs
            pairs = find_matching_pairs(image_path, coco_json_path, '.json')

            if not pairs:
                click.echo("No matching image-annotation pairs found.")
                return

            click.echo(f"Found {len(pairs)} image-annotation pairs.")

            # Check output path
            output_path_obj = Path(output_path)
            if output_path_obj.exists() and not output_path_obj.is_dir():
                raise ValueError(f"Output path exists but is not a directory: {output_path}")

            # Create output directory if it doesn't exist
            output_path_obj.mkdir(parents=True, exist_ok=True)

            # Process each pair
            successful = 0
            for img_path, ann_path in pairs:
                # Generate output filename based on image name
                img_stem = Path(img_path).stem
                output_txt_path = output_path_obj / f"{img_stem}.txt"

                try:
                    coco_to_yolo(ann_path, img_path, str(output_txt_path), classes)
                    click.echo(f"  Converted: {Path(img_path).name} → {output_txt_path.name}")
                    successful += 1
                except Exception as e:
                    click.echo(f"  Error processing {Path(img_path).name}: {e}")
                    click.echo("  Skipping...")

            click.echo(f"Batch conversion complete. Successfully converted {successful}/{len(pairs)} files to {output_path}")
        else:
            # Single file mode
            from pathlib import Path
            img_path = Path(image_path)
            ann_path = Path(coco_json_path)

            # Check if inputs are directories (suggest using --batch)
            if img_path.is_dir() or ann_path.is_dir():
                click.echo("Error: Input paths are directories. Did you mean to use --batch flag?")
                click.echo("  For batch processing: dataflow convert coco2yolo <image_dir> <annotation_dir> <output_dir> --batch --class-names <class_names_file>")
                sys.exit(1)

            # Handle output path: if it's a directory, generate filename
            output_path_obj = Path(output_path)
            if output_path_obj.exists() and output_path_obj.is_dir():
                # Generate output filename based on image name
                img_stem = img_path.stem
                output_path = str(output_path_obj / f"{img_stem}.txt")
                click.echo(f"Output is a directory, saving to: {output_path}")

            from .convert.coco_to_yolo import coco_to_yolo
            coco_to_yolo(coco_json_path, image_path, output_path, classes)
            click.echo(f"Successfully converted to {output_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@convert.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('yolo_txt_path', type=click.Path(exists=True))
@click.argument('class_names_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--batch', is_flag=True, help='Batch mode: process directories instead of single files')
def yolo2coco(image_path, yolo_txt_path, class_names_path, output_path, batch):
    """Convert YOLO annotation(s) to COCO format.

    In batch mode (--batch):
      - Creates a single COCO file with all images and annotations
    """
    try:
        with open(class_names_path, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]

        from pathlib import Path

        if batch:
            # Batch mode: process directories
            from .visualize.batch import find_matching_pairs, validate_batch_directories
            from .convert.yolo_to_coco import yolo_to_coco, batch_yolo_to_coco
            from pathlib import Path

            # Validate directories
            validate_batch_directories(image_path, yolo_txt_path)

            # Find matching pairs
            pairs = find_matching_pairs(image_path, yolo_txt_path, '.txt')

            if not pairs:
                click.echo("No matching image-annotation pairs found.")
                return

            click.echo(f"Found {len(pairs)} image-annotation pairs.")

            # Output must be a single COCO file in batch mode
            output_path_obj = Path(output_path)

            # Determine output file path
            if output_path_obj.exists() and output_path_obj.is_dir():
                # If output is a directory, create default filename
                output_json_path = output_path_obj / "coco_annotations.json"
            else:
                # Ensure .json extension
                if not str(output_path).lower().endswith('.json'):
                    output_json_path = Path(str(output_path) + '.json')
                else:
                    output_json_path = output_path_obj

            # Ensure parent directory exists
            output_json_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                batch_yolo_to_coco(pairs, classes, str(output_json_path))
                click.echo(f"COCO file saved to: {output_json_path}")
            except Exception as e:
                click.echo(f"Error creating COCO file: {e}")
                sys.exit(1)
        else:
            # Single file mode
            from pathlib import Path
            img_path = Path(image_path)
            ann_path = Path(yolo_txt_path)

            # Check if inputs are directories (suggest using --batch)
            if img_path.is_dir() or ann_path.is_dir():
                click.echo("Error: Input paths are directories. Did you mean to use --batch flag?")
                click.echo("  For batch processing: dataflow convert yolo2coco <image_dir> <annotation_dir> <class_names> <output_dir> --batch")
                sys.exit(1)

            # Check if output path is a directory
            output_path_obj = Path(output_path)
            if output_path_obj.exists() and output_path_obj.is_dir():
                click.echo("Error: Output path is a directory. Please specify a JSON file path.")
                click.echo("  Example: dataflow convert yolo2coco image.jpg annotation.txt classes.txt output.json")
                sys.exit(1)

            # Ensure .json extension
            if not str(output_path).lower().endswith('.json'):
                click.echo("Warning: Output path does not have .json extension. Adding .json extension.")
                output_path = str(output_path) + '.json'

            from .convert.yolo_to_coco import yolo_to_coco
            yolo_to_coco(yolo_txt_path, image_path, classes, output_path)
            click.echo(f"Successfully converted to {output_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@convert.command()
@click.argument('labelme_json_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--batch', is_flag=True, help='Batch mode: process directory instead of single file')
def labelme2coco(labelme_json_path, output_path, batch):
    """Convert LabelMe annotation(s) to COCO format.

    In batch mode (--batch):
      - Creates a single COCO file with all images and annotations
    """
    try:
        from pathlib import Path

        if batch:
            # Batch mode: process directory
            from .convert.batch import find_matching_conversion_pairs, validate_conversion_directories
            from .convert.labelme_to_coco import labelme_to_coco, batch_labelme_to_coco

            # Validate directory (no input needed for LabelMe)
            labelme_path_obj = Path(labelme_json_path)
            if not labelme_path_obj.exists():
                raise FileNotFoundError(f"LabelMe path not found: {labelme_json_path}")

            if labelme_path_obj.is_file():
                raise ValueError(f"LabelMe path is a file, not a directory. Did you mean to omit --batch flag?")

            # Get all LabelMe JSON files in directory
            labelme_files = list(labelme_path_obj.glob("*.json")) + list(labelme_path_obj.glob("*.JSON"))
            if not labelme_files:
                raise ValueError(f"No LabelMe JSON files found in {labelme_json_path}")

            # Create pairs (same file for both input and annotation)
            pairs = [(str(f), str(f)) for f in labelme_files]

            click.echo(f"Found {len(pairs)} LabelMe annotation files.")

            # Output must be a single COCO file in batch mode
            output_path_obj = Path(output_path)

            # Determine output file path
            if output_path_obj.exists() and output_path_obj.is_dir():
                # If output is a directory, create default filename
                output_json_path = output_path_obj / "coco_annotations.json"
            else:
                # Ensure .json extension
                if not str(output_path).lower().endswith('.json'):
                    output_json_path = Path(str(output_path) + '.json')
                else:
                    output_json_path = output_path_obj

            # Ensure parent directory exists
            output_json_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                batch_labelme_to_coco(pairs, str(output_json_path))
                click.echo(f"COCO file saved to: {output_json_path}")
            except Exception as e:
                click.echo(f"Error creating COCO file: {e}")
                sys.exit(1)
        else:
            # Single file mode
            from pathlib import Path
            labelme_path_obj = Path(labelme_json_path)

            # Check if input is a directory (suggest using --batch)
            if labelme_path_obj.is_dir():
                click.echo("Error: Input path is a directory. Did you mean to use --batch flag?")
                click.echo("  For batch processing: dataflow convert labelme2coco <labelme_dir> <output_dir> --batch")
                sys.exit(1)

            # Check if output path is a directory
            output_path_obj = Path(output_path)
            if output_path_obj.exists() and output_path_obj.is_dir():
                click.echo("Error: Output path is a directory. Please specify a JSON file path.")
                click.echo("  Example: dataflow convert labelme2coco labelme.json output.json")
                sys.exit(1)

            # Ensure .json extension
            if not str(output_path).lower().endswith('.json'):
                click.echo("Warning: Output path does not have .json extension. Adding .json extension.")
                output_path = str(output_path) + '.json'

            from .convert.labelme_to_coco import labelme_to_coco
            labelme_to_coco(labelme_json_path, output_path)
            click.echo(f"Successfully converted to {output_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@convert.command()
@click.argument('coco_json_path', type=click.Path(exists=True))
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--batch', is_flag=True, help='Batch mode: process directories instead of single files')
def coco2labelme(coco_json_path, image_path, output_path, batch):
    """Convert COCO annotation(s) to LabelMe format.

    In batch mode (--batch):
      - Creates separate LabelMe files for each image (one .json per image)
    """
    try:
        from pathlib import Path

        if batch:
            # Batch mode: process directories
            from .visualize.batch import find_matching_pairs, validate_batch_directories
            from .convert.coco_to_labelme import coco_to_labelme, batch_coco_to_labelme

            # Validate directories
            validate_batch_directories(image_path, coco_json_path)

            # Find matching pairs
            pairs = find_matching_pairs(image_path, coco_json_path, '.json')

            if not pairs:
                click.echo("No matching image-annotation pairs found.")
                return

            click.echo(f"Found {len(pairs)} image-annotation pairs.")

            # Check output path
            output_path_obj = Path(output_path)
            if output_path_obj.exists() and not output_path_obj.is_dir():
                raise ValueError(f"Output path exists but is not a directory: {output_path}")

            # Create output directory if it doesn't exist
            output_path_obj.mkdir(parents=True, exist_ok=True)

            # Process each pair
            successful = 0
            for img_path, ann_path in pairs:
                # Generate output filename based on image name
                img_stem = Path(img_path).stem
                output_json_path = output_path_obj / f"{img_stem}.json"

                try:
                    coco_to_labelme(ann_path, img_path, str(output_json_path))
                    click.echo(f"  Converted: {Path(img_path).name} → {output_json_path.name}")
                    successful += 1
                except Exception as e:
                    click.echo(f"  Error processing {Path(img_path).name}: {e}")
                    click.echo("  Skipping...")

            click.echo(f"Batch conversion complete. Successfully converted {successful}/{len(pairs)} files to {output_path}")
        else:
            # Single file mode
            from pathlib import Path
            img_path = Path(image_path)
            ann_path = Path(coco_json_path)

            # Check if inputs are directories (suggest using --batch)
            if img_path.is_dir() or ann_path.is_dir():
                click.echo("Error: Input paths are directories. Did you mean to use --batch flag?")
                click.echo("  For batch processing: dataflow convert coco2labelme <image_dir> <annotation_dir> <output_dir> --batch")
                sys.exit(1)

            from .convert.coco_to_labelme import coco_to_labelme
            coco_to_labelme(coco_json_path, image_path, output_path)
            click.echo(f"Successfully converted to {output_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@convert.command()
@click.argument('labelme_json_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--class-names', type=click.Path(exists=True),
              help='Path to class names file (one per line)')
@click.option('--batch', is_flag=True, help='Batch mode: process directory instead of single file')
def labelme2yolo(labelme_json_path, output_path, class_names, batch):
    """Convert LabelMe annotation(s) to YOLO format.

    In batch mode (--batch):
      - Creates separate YOLO files for each LabelMe file (one .txt per LabelMe file)
    """
    try:
        if class_names:
            with open(class_names, 'r') as f:
                classes = [line.strip() for line in f if line.strip()]
        else:
            classes = []

        from pathlib import Path

        if batch:
            # Batch mode: process directory
            from .convert.labelme_to_yolo import labelme_to_yolo, batch_labelme_to_yolo

            # Validate directory
            labelme_path_obj = Path(labelme_json_path)
            if not labelme_path_obj.exists():
                raise FileNotFoundError(f"LabelMe path not found: {labelme_json_path}")

            if labelme_path_obj.is_file():
                raise ValueError(f"LabelMe path is a file, not a directory. Did you mean to omit --batch flag?")

            # Get all LabelMe JSON files in directory
            labelme_files = list(labelme_path_obj.glob("*.json")) + list(labelme_path_obj.glob("*.JSON"))
            if not labelme_files:
                raise ValueError(f"No LabelMe JSON files found in {labelme_json_path}")

            # Create pairs (same file for both input and annotation)
            pairs = [(str(f), str(f)) for f in labelme_files]

            click.echo(f"Found {len(pairs)} LabelMe annotation files.")

            # Check output path
            output_path_obj = Path(output_path)
            if output_path_obj.exists() and not output_path_obj.is_dir():
                raise ValueError(f"Output path exists but is not a directory: {output_path}")

            # Create output directory if it doesn't exist
            output_path_obj.mkdir(parents=True, exist_ok=True)

            # Process each LabelMe file
            successful = 0
            for ann_path in labelme_files:
                # Generate output filename based on annotation name
                ann_stem = ann_path.stem
                output_txt_path = output_path_obj / f"{ann_stem}.txt"

                try:
                    labelme_to_yolo(str(ann_path), str(output_txt_path), classes)
                    click.echo(f"  Converted: {ann_path.name} → {output_txt_path.name}")
                    successful += 1
                except Exception as e:
                    click.echo(f"  Error processing {ann_path.name}: {e}")
                    click.echo("  Skipping...")

            click.echo(f"Batch conversion complete. Successfully converted {successful}/{len(pairs)} files to {output_path}")
            if classes:
                click.echo(f"Class names used: {classes}")
        else:
            # Single file mode
            from pathlib import Path
            labelme_path_obj = Path(labelme_json_path)

            # Check if input is a directory (suggest using --batch)
            if labelme_path_obj.is_dir():
                click.echo("Error: Input path is a directory. Did you mean to use --batch flag?")
                click.echo("  For batch processing: dataflow convert labelme2yolo <labelme_dir> <output_dir> --batch --class-names <class_names_file>")
                sys.exit(1)

            from .convert.labelme_to_yolo import labelme_to_yolo
            labelme_to_yolo(labelme_json_path, output_path, classes)
            click.echo(f"Successfully converted to {output_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@convert.command()
@click.argument('yolo_txt_path', type=click.Path(exists=True))
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('class_names_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--batch', is_flag=True, help='Batch mode: process directories instead of single files')
def yolo2labelme(yolo_txt_path, image_path, class_names_path, output_path, batch):
    """Convert YOLO annotation(s) to LabelMe format.

    In batch mode (--batch):
      - Creates separate LabelMe files for each image (one .json per image)
    """
    try:
        with open(class_names_path, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]

        from pathlib import Path

        if batch:
            # Batch mode: process directories
            from .visualize.batch import find_matching_pairs, validate_batch_directories
            from .convert.yolo_to_labelme import yolo_to_labelme, batch_yolo_to_labelme

            # Validate directories
            validate_batch_directories(image_path, yolo_txt_path)

            # Find matching pairs
            pairs = find_matching_pairs(image_path, yolo_txt_path, '.txt')

            if not pairs:
                click.echo("No matching image-annotation pairs found.")
                return

            click.echo(f"Found {len(pairs)} image-annotation pairs.")

            # Check output path
            output_path_obj = Path(output_path)
            if output_path_obj.exists() and not output_path_obj.is_dir():
                raise ValueError(f"Output path exists but is not a directory: {output_path}")

            # Create output directory if it doesn't exist
            output_path_obj.mkdir(parents=True, exist_ok=True)

            # Process each pair
            successful = 0
            for img_path, ann_path in pairs:
                # Generate output filename based on image name
                img_stem = Path(img_path).stem
                output_json_path = output_path_obj / f"{img_stem}.json"

                try:
                    yolo_to_labelme(ann_path, img_path, classes, str(output_json_path))
                    click.echo(f"  Converted: {Path(img_path).name} → {output_json_path.name}")
                    successful += 1
                except Exception as e:
                    click.echo(f"  Error processing {Path(img_path).name}: {e}")
                    click.echo("  Skipping...")

            click.echo(f"Batch conversion complete. Successfully converted {successful}/{len(pairs)} files to {output_path}")
        else:
            # Single file mode
            from pathlib import Path
            img_path = Path(image_path)
            ann_path = Path(yolo_txt_path)

            # Check if inputs are directories (suggest using --batch)
            if img_path.is_dir() or ann_path.is_dir():
                click.echo("Error: Input paths are directories. Did you mean to use --batch flag?")
                click.echo("  For batch processing: dataflow convert yolo2labelme <image_dir> <annotation_dir> <class_names> <output_dir> --batch")
                sys.exit(1)

            from .convert.yolo_to_labelme import yolo_to_labelme
            yolo_to_labelme(yolo_txt_path, image_path, classes, output_path)
            click.echo(f"Successfully converted to {output_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# Visualization commands
@main.group(context_settings={'help_option_names': ['-h', '--help']})
def visualize():
    """Visualize annotations on images."""
    pass


@visualize.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('annotation_path', type=click.Path(exists=True))
@click.option('--save', type=click.Path(), help='Save visualization to file or directory')
@click.option('--show/--no-show', default=True, help='Show visualization window')
@click.option('--class-names', type=click.Path(exists=True),
              help='Path to class names file (required for YOLO format)')
@click.option('--batch', is_flag=True, help='Batch mode: process directories instead of single files')
def coco(image_path, annotation_path, save, show, class_names, batch):
    """Visualize COCO annotation(s)."""
    try:
        if batch:
            # Batch mode: process directories
            from pathlib import Path
            from .visualize.batch import find_matching_pairs, validate_batch_directories
            from .visualize.base import BaseVisualizer
            from .visualize.coco_vis import visualize_coco

            # Validate directories
            validate_batch_directories(image_path, annotation_path)

            # Find matching pairs
            pairs = find_matching_pairs(image_path, annotation_path, '.json')

            if not pairs:
                click.echo("No matching image-annotation pairs found.")
                return

            click.echo(f"Found {len(pairs)} image-annotation pairs.")

            # Process in batch mode
            current_idx = 0
            while current_idx < len(pairs):
                img_path, ann_path = pairs[current_idx]
                progress = f"[{current_idx + 1}/{len(pairs)}]"
                click.echo(f"{progress} Processing: {Path(img_path).name} ↔ {Path(ann_path).name}")

                result = visualize_coco(img_path, ann_path)

                if save:
                    # If save is a directory, save with original filename
                    save_path = Path(save)
                    if save_path.is_dir():
                        output_path = save_path / f"{Path(img_path).stem}_vis.jpg"
                    else:
                        output_path = save_path

                    BaseVisualizer.save_image(result, str(output_path))
                    click.echo(f"  Saved to: {output_path}")

                if show:
                    key = BaseVisualizer.show_batch_navigation(
                        result,
                        "COCO Visualization",
                        current_idx,
                        len(pairs)
                    )

                    if key == 'q':
                        click.echo("Batch visualization stopped by user.")
                        break
                    elif key == 'left' and current_idx > 0:
                        current_idx -= 1
                    elif key == 'right':
                        current_idx += 1
                    # 'other' key stays on current image
                else:
                    # Non-interactive mode, just process all
                    current_idx += 1

        else:
            # Original single-file mode
            from .visualize.coco_vis import visualize_coco
            result = visualize_coco(image_path, annotation_path)

            if save:
                import cv2
                cv2.imwrite(save, result)
                click.echo(f"Saved visualization to {save}")

            if show:
                import cv2
                cv2.imshow("COCO Visualization", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@visualize.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('annotation_path', type=click.Path(exists=True))
@click.argument('class_names_path', type=click.Path(exists=True))
@click.option('--save', type=click.Path(), help='Save visualization to file or directory')
@click.option('--show/--no-show', default=True, help='Show visualization window')
@click.option('--batch', is_flag=True, help='Batch mode: process directories instead of single files')
def yolo(image_path, annotation_path, class_names_path, save, show, batch):
    """Visualize YOLO annotation(s)."""
    try:
        import sys
        print(f"DEBUG yolo args: save={repr(save)}, show={show}, batch={batch}", file=sys.stderr, flush=True)
        # Common imports for both modes
        from pathlib import Path
        from .visualize.base import BaseVisualizer
        from .visualize.yolo_vis import visualize_yolo

        if batch:
            # Batch mode: process directories
            from .visualize.batch import find_matching_pairs, validate_batch_directories

            # Load class names (single file for entire batch)
            with open(class_names_path, 'r') as f:
                classes = [line.strip() for line in f if line.strip()]

            # Validate directories
            validate_batch_directories(image_path, annotation_path)

            # Find matching pairs
            pairs = find_matching_pairs(image_path, annotation_path, '.txt')

            if not pairs:
                click.echo("No matching image-annotation pairs found.")
                return

            click.echo(f"Found {len(pairs)} image-annotation pairs.")

            # Process in batch mode
            current_idx = 0
            while current_idx < len(pairs):
                img_path, ann_path = pairs[current_idx]
                progress = f"[{current_idx + 1}/{len(pairs)}]"
                click.echo(f"{progress} Processing: {Path(img_path).name} ↔ {Path(ann_path).name}")

                import sys
                print(f"DEBUG cli: before visualize_yolo", file=sys.stderr, flush=True)
                click.echo(f"DEBUG cli: before visualize_yolo")

                result = visualize_yolo(img_path, ann_path, classes)

                if save:
                    import sys
                    print(f"DEBUG cli: save={save}, result shape={result.shape if result is not None else 'None'}", file=sys.stderr, flush=True)
                    # Determine if save path is a directory or file
                    save_path = Path(save)
                    print(f"DEBUG cli: save_path={save_path}, exists={save_path.exists()}", file=sys.stderr, flush=True)

                    # Check if it's a directory or should be treated as one
                    is_directory = False
                    if save_path.exists():
                        is_directory = save_path.is_dir()
                    else:
                        # If path doesn't exist, check if it looks like a directory
                        # (ends with slash, has no extension, or parent doesn't exist)
                        save_str = str(save)
                        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
                        has_image_ext = save_path.suffix.lower() in valid_extensions
                        ends_with_slash = save_str.endswith('/') or save_str.endswith('\\')

                        if ends_with_slash or not has_image_ext:
                            # Treat as directory
                            is_directory = True
                            # Create the directory
                            save_path.mkdir(parents=True, exist_ok=True)

                    if is_directory:
                        output_path = save_path / f"{Path(img_path).stem}_vis.jpg"
                    else:
                        output_path = save_path
                        # Ensure parent directory exists
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Convert to absolute path to avoid issues with relative paths
                    output_path = output_path.absolute()
                    BaseVisualizer.save_image(result, str(output_path))
                    click.echo(f"  Saved to: {output_path}")

                if show:
                    key = BaseVisualizer.show_batch_navigation(
                        result,
                        "YOLO Visualization",
                        current_idx,
                        len(pairs)
                    )

                    if key == 'q':
                        click.echo("Batch visualization stopped by user.")
                        break
                    elif key == 'left' and current_idx > 0:
                        current_idx -= 1
                    elif key == 'right':
                        current_idx += 1
                    # 'other' key stays on current image
                else:
                    # Non-interactive mode, just process all
                    current_idx += 1

        else:
            # Original single-file mode
            with open(class_names_path, 'r') as f:
                classes = [line.strip() for line in f if line.strip()]

            from .visualize.yolo_vis import visualize_yolo
            result = visualize_yolo(image_path, annotation_path, classes)

            if save:
                BaseVisualizer.save_image(result, save)
                click.echo(f"Saved visualization to {save}")

            if show:
                import cv2
                cv2.imshow("YOLO Visualization", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@visualize.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('annotation_path', type=click.Path(exists=True))
@click.option('--save', type=click.Path(), help='Save visualization to file or directory')
@click.option('--show/--no-show', default=True, help='Show visualization window')
@click.option('--batch', is_flag=True, help='Batch mode: process directories instead of single files')
def labelme(image_path, annotation_path, save, show, batch):
    """Visualize LabelMe annotation(s)."""
    try:
        if batch:
            # Batch mode: process directories
            from pathlib import Path
            from .visualize.batch import find_matching_pairs, validate_batch_directories
            from .visualize.base import BaseVisualizer
            from .visualize.labelme_vis import visualize_labelme

            # Validate directories
            validate_batch_directories(image_path, annotation_path)

            # Find matching pairs
            pairs = find_matching_pairs(image_path, annotation_path, '.json')

            if not pairs:
                click.echo("No matching image-annotation pairs found.")
                return

            click.echo(f"Found {len(pairs)} image-annotation pairs.")

            # Process in batch mode
            current_idx = 0
            while current_idx < len(pairs):
                img_path, ann_path = pairs[current_idx]
                progress = f"[{current_idx + 1}/{len(pairs)}]"
                click.echo(f"{progress} Processing: {Path(img_path).name} ↔ {Path(ann_path).name}")

                result = visualize_labelme(img_path, ann_path)

                if save:
                    # If save is a directory, save with original filename
                    save_path = Path(save)
                    if save_path.is_dir():
                        output_path = save_path / f"{Path(img_path).stem}_vis.jpg"
                    else:
                        output_path = save_path

                    BaseVisualizer.save_image(result, str(output_path))
                    click.echo(f"  Saved to: {output_path}")

                if show:
                    key = BaseVisualizer.show_batch_navigation(
                        result,
                        "LabelMe Visualization",
                        current_idx,
                        len(pairs)
                    )

                    if key == 'q':
                        click.echo("Batch visualization stopped by user.")
                        break
                    elif key == 'left' and current_idx > 0:
                        current_idx -= 1
                    elif key == 'right':
                        current_idx += 1
                    # 'other' key stays on current image
                else:
                    # Non-interactive mode, just process all
                    current_idx += 1

        else:
            # Original single-file mode
            from .visualize.labelme_vis import visualize_labelme
            result = visualize_labelme(image_path, annotation_path)

            if save:
                import cv2
                cv2.imwrite(save, result)
                click.echo(f"Saved visualization to {save}")

            if show:
                import cv2
                cv2.imshow("LabelMe Visualization", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()