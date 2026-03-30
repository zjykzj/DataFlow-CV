#!/usr/bin/env python3
"""
YOLO annotation visualization example

Demonstrates how to use YOLOVisualizer to visualize YOLO format annotations.
Supports automatic detection of object detection and instance segmentation formats.

Usage:
    python yolo_demo.py [--task {det,seg}] [--verbose]

Examples:
    python yolo_demo.py                     # Visualize object detection annotations (default)
    python yolo_demo.py --task seg          # Visualize instance segmentation annotations
    python yolo_demo.py --verbose           # Enable verbose logging mode
    python yolo_demo.py --task seg --verbose # Instance segmentation + verbose logging
"""

import argparse
import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.util import LoggingOperations, VerboseLoggingOperations
from dataflow.visualize import YOLOVisualizer


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="YOLO annotation visualization example")
    parser.add_argument(
        "--task",
        choices=["det", "seg"],
        default="det",
        help="Task type: det=object detection, seg=instance segmentation (default: det)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging mode")
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        log_ops = VerboseLoggingOperations()
        logger = log_ops.get_verbose_logger(
            name="yolo_visualize_demo",
            verbose=True,
            log_dir=str(project_root / "logs")
        )
        logger.info("Verbose logging mode enabled")
    else:
        log_ops = LoggingOperations()
        logger = log_ops.get_logger("yolo_visualize_demo", level="INFO")

    # Select data path based on task type
    if args.task == "det":
        task_name = "object detection"
        data_dir = project_root / "assets" / "test_data" / "det" / "yolo"
    else:  # args.task == "seg"
        task_name = "instance segmentation"
        data_dir = project_root / "assets" / "test_data" / "seg" / "yolo"

    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"
    class_file = data_dir / "classes.txt"

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.info("Please ensure sample data is prepared")
        return

    logger.info("=" * 50)
    logger.info(f"YOLO {task_name}annotation visualization example")
    logger.info("=" * 50)

    # Create visualizer
    logger.info(f"Creating YOLO visualizer:")
    logger.info(f"  Task type: {task_name}")
    logger.info(f"  Label directory: {label_dir}")
    logger.info(f"  Image directory: {image_dir}")
    logger.info(f"  Class file: {class_file}")

    # Set different output directories based on task type to avoid conflicts
    output_dir = data_dir / "visualized_output"

    visualizer = YOLOVisualizer(
        label_dir=str(label_dir),
        image_dir=str(image_dir),
        class_file=str(class_file),
        verbose=args.verbose,  # Verbose logging mode
        is_show=True,  # Display window
        is_save=True,  # Also save
        output_dir=output_dir,
        strict_mode=True,
        logger=logger,
    )

    # Perform visualization
    logger.info("\nStarting visualization (press Enter for next image, press q to quit)...")
    result = visualizer.visualize()

    if result.success:
        logger.info(f"Visualization completed: {result.message}")
        logger.info(f"Results saved to: {output_dir}")
    else:
        logger.error(f"Visualization failed: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")


if __name__ == "__main__":
    main()
