"""
Log formatting templates for the Convert module.

These are **pure functions** that return strings — they never call
``logger.info()`` or any other logging method.  The caller decides the
log level and passes the result to the logger.
"""

from typing import Any, Dict, List, Optional

from dataflow.util.logging import format_divider, format_section, format_kv


def format_convert_header(
    source_format: str,
    target_format: str,
    source_path: str,
    target_path: str,
    mode: str = "label",
    strict: bool = True,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a header block for conversion start.

    Args:
        source_format: Source format name (``"yolo"``, ``"coco"``, ``"labelme"``).
        target_format: Target format name.
        source_path: Source annotation path.
        target_path: Target output path.
        mode: Conversion mode — ``"label"`` or ``"prediction"``.
        strict: Whether strict mode is enabled.
        options: Optional flags, e.g. ``{"do_rle": True}``.

    Returns:
        Formatted header string.
    """
    lines: List[str] = []
    lines.append(format_divider())
    lines.append(f"Convert: {source_format.upper()} → {target_format.upper()}")
    lines.append(format_kv("Source", source_path))
    lines.append(format_kv("Target", target_path))
    lines.append(format_kv("Mode", f"{mode}, {'strict' if strict else 'non-strict'}"))
    if options:
        for key, value in options.items():
            lines.append(format_kv(key.capitalize(), value))
    lines.append("")
    return "\n".join(lines)


def format_convert_phase(phase: str, stats: Dict[str, Any]) -> str:
    """Return a phase marker with statistics.

    Args:
        phase: Phase name — ``"Read"``, ``"Convert"``, ``"Write"``.
        stats: Statistics for this phase, e.g.
            ``{"images": 500, "categories": 3, "objects": 3240}``.

    Returns:
        Formatted phase string.
    """
    lines: List[str] = [format_section(f"Phase: {phase}")]
    for key, value in stats.items():
        lines.append(format_kv(key.capitalize(), value))
    lines.append("")
    return "\n".join(lines)


def format_convert_result(result: Any) -> str:
    """Return a final result block for a completed conversion.

    Args:
        result: A ``ConversionResult`` instance.

    Returns:
        Formatted result block.
    """
    from dataflow.util.logging import format_result_block

    status = "✓ Success" if result.success else "✗ Failed"
    items: Dict[str, Any] = {
        "Images": result.num_images_converted,
        "Objects": result.num_objects_converted,
    }

    if getattr(result, "metadata", None):
        duration = result.metadata.get("duration_seconds")
        if duration:
            items["Duration"] = f"{duration}s"

    if result.warnings:
        items["Warnings"] = len(result.warnings)

    return format_result_block(status, items, log_path=result.log_path)
