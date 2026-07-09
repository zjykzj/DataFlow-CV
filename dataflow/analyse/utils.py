"""
Utility functions for the Analyse module.

Format auto-detection, handler factory, and class file parsing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional


def detect_format(label_path: Path) -> str:
    """Auto-detect annotation format from path.

    Detection rules (checked in order):
    1. If label_path is a file ending in .json → ``"coco"``
    2. If label_path is a directory:
       - Check first non-hidden file's extension:
         - ``.txt`` → ``"yolo"``
         - ``.json`` → read first .json, check for ``"shapes"`` key → ``"labelme"``;
           check for ``"images"`` key → ``"coco"``
       - Empty directory → ``ValueError``

    Args:
        label_path: Path to labels (directory or file).

    Returns:
        ``"yolo"`` | ``"labelme"`` | ``"coco"``

    Raises:
        ValueError: If format cannot be determined.
    """
    if not label_path.exists():
        raise ValueError(f"Label path does not exist: {label_path}")

    # 1. Single file → inspect contents to distinguish COCO vs LabelMe
    if label_path.is_file():
        if label_path.suffix != ".json":
            raise ValueError(
                f"Single-file label path must be a .json file, got: {label_path}"
            )
        # Read file to distinguish COCO vs LabelMe
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "images" in data:
                return "coco"
            if "shapes" in data:
                return "labelme"
            if "annotations" in data:
                # COCO-compatible: has annotations key but possibly
                # missing images (prediction list format)
                return "coco"
        raise ValueError(
            f"Cannot determine format from file: {label_path}. "
            f"Expected COCO ('images' / 'annotations' key) or "
            f"LabelMe ('shapes' key)."
        )

    # 2. Directory → inspect contents
    files = sorted(
        [f for f in label_path.iterdir() if f.is_file() and not f.name.startswith(".")]
    )

    if not files:
        raise ValueError(
            f"No annotation files found in directory: {label_path}"
        )

    # Check extensions (exclude auxiliary files like classes.txt)
    annotation_files = [
        f for f in files
        if f.name not in ("classes.txt",)
    ]
    extensions = {f.suffix for f in annotation_files}
    has_txt = ".txt" in extensions
    has_json = ".json" in extensions

    # Mixed extensions → ambiguous, refuse to guess
    if has_txt and has_json:
        raise ValueError(
            f"Cannot determine annotation format from: {label_path}. "
            f"Directory contains both .txt (YOLO) and .json "
            f"(LabelMe/COCO) files. Please separate them or specify "
            f"the format explicitly."
        )

    if has_txt:
        return "yolo"

    if has_json:
        # Read first .json to distinguish LabelMe from COCO
        json_files = [f for f in files if f.suffix == ".json"]
        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if "shapes" in data:
                return "labelme"
            if "images" in data:
                raise ValueError(
                    f"{label_path} appears to contain a COCO JSON file, "
                    f"but COCO annotations are a single file, not a "
                    f"directory. Point to the specific .json file instead."
                )

        raise ValueError(
            f"Cannot determine format from JSON files in: {label_path}. "
            f"Expected LabelMe ('shapes' key)."
        )

    raise ValueError(
        f"Cannot determine annotation format from: {label_path}. "
        f"Supported formats: .txt (YOLO), .json (LabelMe) or "
        f"a single COCO .json file."
    )


def create_handler(
    label_path: Path,
    format: str,
    class_file: Optional[Path] = None,
    image_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
):
    """Create the appropriate handler for the detected format.

    All handlers are created with ``strict_mode=False`` — analysis
    operations are read-only and should be lenient with imperfect data.

    Args:
        label_path: Path to labels — directory (YOLO/LabelMe) or JSON file (COCO).
        format: ``"yolo"`` | ``"labelme"`` | ``"coco"``.
        class_file: Classes.txt path (used for YOLO name mapping, optional for others).
        image_dir: Image directory path. Required for YOLO (to get image dimensions).
            If not provided for YOLO, attempts to auto-detect a sibling ``images/``
            directory.
        logger: Logger to pass to the handler.

    Returns:
        Configured BaseAnnotationHandler instance.

    Raises:
        ValueError: If format is unknown.
    """
    from dataflow.label.yolo_handler import YoloAnnotationHandler
    from dataflow.label.labelme_handler import LabelMeAnnotationHandler
    from dataflow.label.coco_handler import CocoAnnotationHandler

    handler_kwargs = {
        "strict_mode": False,
        "logger": logger,
    }

    if format == "yolo":
        if class_file is None:
            raise ValueError(
                "class_file is required for YOLO format. "
                "Provide a classes.txt file to map class IDs to names."
            )
        if image_dir is None:
            # Auto-detect: sibling "images" directory
            candidate = label_path.parent / "images"
            if candidate.is_dir():
                image_dir = candidate
            else:
                raise ValueError(
                    f"image_dir is required for YOLO format. "
                    f"No sibling 'images/' directory found at "
                    f"{candidate}. Use --image-dir to specify "
                    f"the image directory."
                )

        handler = YoloAnnotationHandler(
            label_dir=str(label_path),
            class_file=str(class_file),
            image_dir=str(image_dir) if image_dir else str(label_path.parent),
            **handler_kwargs,
        )
    elif format == "labelme":
        handler = LabelMeAnnotationHandler(
            label_dir=str(label_path),
            class_file=str(class_file) if class_file else None,
            **handler_kwargs,
        )
    elif format == "coco":
        annotation_file = str(label_path)
        handler = CocoAnnotationHandler(
            annotation_file=annotation_file,
            **handler_kwargs,
        )
    else:
        raise ValueError(f"Unknown format: {format}")

    return handler


def load_class_names(class_file: Path) -> Dict[int, str]:
    """Parse classes.txt → ``{class_id: class_name}``.

    Format: one class name per line, 0-indexed.
    Blank lines and lines starting with ``#`` are skipped.

    Args:
        class_file: Path to classes.txt file.

    Returns:
        Dict mapping class ID (int) to class name (str).

    Raises:
        FileNotFoundError: If class_file does not exist.
        ValueError: If class_file is empty (no valid class names found).
    """
    if not class_file.exists():
        raise FileNotFoundError(f"Class file not found: {class_file}")

    class_names: Dict[int, str] = {}
    class_id = 0
    with open(class_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            class_names[class_id] = stripped
            class_id += 1

    if not class_names:
        raise ValueError(f"No class names found in: {class_file}")

    return class_names
