"""
Log formatting templates for the Visualize module.

These are **pure functions** that return strings — they never call
``logger.info()`` or any other logging method.  The caller decides the
log level and passes the result to the logger.
"""

from typing import Any, Dict, List, Optional

from dataflow.util.logging import format_divider, format_section, format_kv


def format_viz_header(
    format_name: str,
    label_dir: str,
    image_dir: str,
    is_show: bool = True,
    is_save: bool = False,
    output_dir: Optional[str] = None,
) -> str:
    """Return a header block for visualization start.

    Args:
        format_name: Format name — ``"YOLO"``, ``"COCO"``, ``"LabelMe"``.
        label_dir: Annotation directory path.
        image_dir: Image directory path.
        is_show: Whether display window is enabled.
        is_save: Whether saving is enabled.
        output_dir: Save directory path (if ``is_save`` is True).

    Returns:
        Formatted header string.
    """
    lines: List[str] = []
    lines.append(format_divider())
    lines.append(f"Visualize: {format_name}")
    lines.append(format_kv("Labels", label_dir))
    lines.append(format_kv("Images", image_dir))
    lines.append(format_kv("Display", "yes" if is_show else "no"))
    if is_save and output_dir:
        lines.append(format_kv("Save", f"yes → {output_dir}"))
    else:
        lines.append(format_kv("Save", "no"))
    lines.append("")
    return "\n".join(lines)


def format_viz_progress(
    index: int,
    image_name: str,
    n_objects: int,
    status: str,
) -> str:
    """Return a single-line progress entry.

    Args:
        index: Image counter (1-based).
        image_name: Image file name.
        n_objects: Number of objects in this image.
        status: Status marker — ``"✓"`` for success, ``"✗"`` for failure.

    Returns:
        Single-line progress string.
    """
    return f"  {index:03d}  {image_name}  ({n_objects} objects)  {status}"


def format_viz_result(stats: Dict[str, Any]) -> str:
    """Return a final result block for visualization.

    Args:
        stats: Statistics dict with keys:
            ``"total"``, ``"success"``, ``"failed"``, ``"objects"``,
            ``"duration"``, ``"log_path"``.

    Returns:
        Formatted result block.
    """
    from dataflow.util.logging import format_result_block

    total = stats.get("total", 0)
    success = stats.get("success", 0)
    failed = stats.get("failed", 0)

    items: Dict[str, Any] = {
        "Status": "✓ Success",
        "Images": f"{success} / {total} ({failed} failed)",
        "Objects": stats.get("objects", 0),
    }

    duration = stats.get("duration")
    if duration:
        items["Duration"] = duration

    return format_result_block(
        "✓ Success",
        items,
        log_path=stats.get("log_path"),
    )
