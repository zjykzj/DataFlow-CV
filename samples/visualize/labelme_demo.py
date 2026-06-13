#!/usr/bin/env python3
"""
LabelMe annotation visualization example

Demonstrates how to use LabelMeVisualizer to visualize LabelMe format annotations.
Supports display and saving of object detection and instance segmentation annotations.

Usage:
    python labelme_demo.py [--task {det,seg}] [--verbose]

Examples:
    python labelme_demo.py                     # Visualize object detection annotations (default)
    python labelme_demo.py --task seg          # Visualize instance segmentation annotations
    python labelme_demo.py --verbose           # Enable verbose logging mode
    python labelme_demo.py --task seg --verbose # Instance segmentation + verbose logging
"""

import argparse
import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.util.logging import LogConfig, LogManager
from dataflow.visualize import LabelMeVisualizer


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LabelMe annotation visualization example")
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
        log_config = LogConfig(name="demo", verbose=True, log_dir=Path("logs"))
    log_manager = LogManager(log_config)
    logger = log_manager.logger
    log_path = log_manager.log_path
        )
        logger.info("Verbose logging mode enabled")
        logger.info(f"Verbose log saved to: {log_file_path}")
    else:
        log_config = LogConfig(name="demo", log_dir=Path("logs"))
    log_manager = LogManager(log_config)
    logger = log_manager.logger
        logger = log_manager.logger
        log_file_path = None

    # Select data path based on task type
    if args.task == "det":
        task_name = "object detection"
        data_dir = project_root / "assets" / "test_data" / "det" / "labelme"
    else:  # args.task == "seg"
        task_name = "instance segmentation"
        data_dir = project_root / "assets" / "test_data" / "seg" / "labelme"

    image_dir = data_dir  # LabelMe format stores JSON files and images in the same directory
    class_file = data_dir / "classes.txt"

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.info("Please ensure sample data is prepared")
        return

    logger.info("=" * 50)
    logger.info(f"LabelMe {task_name} annotation visualization example")
    logger.info("=" * 50)

    # Create visualizer
    logger.info(f"Creating LabelMe visualizer:")
    logger.info(f"  Task type: {task_name}")
    logger.info(f"  Label directory: {data_dir}")
    logger.info(f"  Image directory: {image_dir}")
    logger.info(f"  Class file: {class_file if class_file.exists() else 'None'}")

    visualizer = LabelMeVisualizer(
        label_dir=str(data_dir),
        image_dir=str(image_dir),
        class_file=str(class_file) if class_file.exists() else None,
        log_config=LogConfig(name="demo", log_dir=Path("logs")),  # Verbose logging mode
        is_show=True,  # Display window
        is_save=False,  # Do not save
        strict_mode=True,
        logger=logger,
         if args.verbose else None,
    )

    # Perform visualization
    logger.info("\nStarting visualization (press Enter for next image, press q to quit)...")
    result = visualizer.visualize()

    # Print log file path from result if verbose mode
    if args.verbose and result.log_file_path:
        logger.info(f"Verbose log file from result: {result.log_file_path}")

    if result.success:
        logger.info(f"Visualization completed: {result.message}")
    else:
        logger.error(f"Visualization failed: {result.message}")
        if result.errors:
            for error in result.errors:
                logger.error(f"  - {error}")


if __name__ == "__main__":
    main()
