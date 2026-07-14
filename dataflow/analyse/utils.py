"""
Utility functions for the Analyse module.

Format auto-detection, handler factory, and class file parsing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from dataflow.label.base import BaseAnnotationHandler


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


def _parse_class_id_token(token: str) -> Optional[int]:
    """Parse a class ID token, accepting integer-valued float strings.

    YOLO format uses ``class_id`` as an integer, but some tooling
    outputs float-formatted values like ``5.000000``.  This parser
    handles both cases gracefully.

    Returns:
        Integer class ID, or ``None`` if *token* is not a valid
        class ID.
    """
    try:
        return int(token)
    except ValueError:
        pass
    try:
        val = float(token)
        if val.is_integer() and val >= 0:
            return int(val)
    except (ValueError, OverflowError):
        pass
    return None


def _auto_generate_class_file(label_dir: Path, recursive: bool = False) -> Path:
    """Generate a temporary classes.txt from observed class IDs in label files.

    Scans all .txt files in ``label_dir``, collects unique class IDs, and
    creates a temporary ``classes.txt`` with ``class_<id>`` names.

    Args:
        label_dir: Directory containing YOLO .txt label files.
        recursive: If True, use ``rglob`` to scan subdirectories.

    Returns:
        Path to the generated temporary classes.txt file.
    """
    import tempfile

    pattern = label_dir.rglob if recursive else label_dir.glob
    class_ids: set[int] = set()
    for txt_file in sorted(pattern("*.txt")):
        if txt_file.name == "classes.txt":
            continue
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    tokens = stripped.split()
                    if tokens:
                        cid = _parse_class_id_token(tokens[0])
                        if cid is not None:
                            class_ids.add(cid)
        except Exception:
            continue

    if not class_ids:
        raise ValueError(
            f"No valid class IDs found in YOLO label files in: {label_dir}. "
            f"Provide a classes.txt file with --class-file."
        )

    max_id = max(class_ids)
    names = []
    for i in range(max_id + 1):
        if i in class_ids:
            names.append(f"class_{i}")
        else:
            names.append(f"class_{i}")  # placeholder for gaps

    # Write to a named temporary file (must survive beyond the function call)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="analyse_classes_", delete=False
    )
    tmp.write("\n".join(names) + "\n")
    tmp.close()

    return Path(tmp.name)


def create_handler(
    label_path: Path,
    format: str,
    class_file: Optional[Path] = None,
    image_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    skip_image_loading: bool = False,
    recursive: bool = False,
) -> "BaseAnnotationHandler":
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
        skip_image_loading: If True and format is YOLO, skip all image file
            I/O (use placeholder dimensions).  For read-only operations
            like stats that don't need real image dimensions.
        recursive: If True, handler uses ``rglob`` for file discovery,
            traversing subdirectories recursively.  Default False.

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
            class_file = _auto_generate_class_file(label_path, recursive=recursive)
        if image_dir is None:
            # Auto-detect: try common image directory layouts
            candidates = [
                label_path / "images",               # labels/images/
                label_path.parent / "images",        # dataset/images/ (labels/ sibling)
                label_path.parent.parent / "images", # dataset/images/ (labels/val/ → up 2)
            ]
            for candidate in candidates:
                if candidate.is_dir():
                    image_dir = candidate
                    break

        kwargs = {}
        if skip_image_loading:
            kwargs["skip_image_loading"] = True
        kwargs["recursive"] = recursive
        handler = YoloAnnotationHandler(
            label_dir=str(label_path),
            class_file=str(class_file),
            image_dir=str(image_dir) if image_dir else str(label_path),
            **kwargs,
            **handler_kwargs,
        )
    elif format == "labelme":
        handler = LabelMeAnnotationHandler(
            label_dir=str(label_path),
            class_file=str(class_file) if class_file else None,
            recursive=recursive,
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


def _scan_yolo_class_ids(label_dir: Path, recursive: bool = False) -> set:
    """Scan all .txt files in *label_dir* and return the set of class IDs.

    Used for strict class validation before the handler silently drops
    unknown IDs in non-strict mode.

    Args:
        label_dir: Directory containing YOLO .txt label files.
        recursive: If True, use ``rglob`` to scan subdirectories.

    Returns:
        Set of integer class IDs found across all .txt files.
    """
    ids: set = set()
    pattern = label_dir.rglob if recursive else label_dir.glob
    for txt_file in sorted(pattern("*.txt")):
        if txt_file.name == "classes.txt":
            continue
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    tokens = stripped.split()
                    if tokens:
                        cid = _parse_class_id_token(tokens[0])
                        if cid is not None:
                            ids.add(cid)
        except OSError:
            continue
    return ids


def _detect_format_recursive(root: Path) -> str:
    """Detect annotation format by looking at files recursively.

    Uses ``rglob`` to find files in all subdirectories, then applies the
    same detection rules as ``detect_format()`` for directory-based
    formats.  For use when the root directory contains only
    subdirectories (no annotation files directly at the top level).

    Args:
        root: Root directory to search recursively.

    Returns:
        ``"yolo"`` | ``"labelme"``

    Raises:
        ValueError: If format cannot be determined.
    """
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Path does not exist or is not a directory: {root}")

    all_files = sorted(
        [f for f in root.rglob("*") if f.is_file() and not f.name.startswith(".")]
    )

    if not all_files:
        raise ValueError(f"No files found recursively in: {root}")

    annotation_files = [
        f for f in all_files if f.name not in ("classes.txt",)
    ]
    extensions = {f.suffix for f in annotation_files}
    has_txt = ".txt" in extensions
    has_json = ".json" in extensions

    if has_txt and has_json:
        raise ValueError(
            f"Cannot determine annotation format recursively from: {root}. "
            f"Found both .txt (YOLO) and .json (LabelMe/COCO) files."
        )

    if has_txt:
        return "yolo"

    if has_json:
        json_files = [f for f in annotation_files if f.suffix == ".json"]
        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "shapes" in data:
            return "labelme"
        if isinstance(data, dict) and "images" in data:
            raise ValueError(
                f"{root} appears to contain a COCO JSON file, "
                f"but COCO is a single file.  Point to the file directly."
            )
        raise ValueError(
            f"Cannot determine format from JSON files in: {root}."
        )

    raise ValueError(
        f"Cannot determine annotation format from: {root}."
    )


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
