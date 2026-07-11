"""
Log formatting templates for the Analyse module.

Pure functions that return formatted strings — they never call the
logger directly.  The caller decides the log level and passes the
result to ``self.logger.info()`` / ``.debug()`` / etc.
"""

from pathlib import Path
from typing import Dict

from ..util.logging import (
    format_divider,
    format_kv,
    format_result_block,
    format_section,
    format_table,
)


def format_analyse_header(
    operation: str,
    label_path: Path,
    format_name: str,
) -> str:
    """Header block shown at the start of an analysis operation.

    Args:
        operation: Operation name, e.g. ``"Dataset Statistics"``.
        label_path: Path to the label source.
        format_name: Detected format, e.g. ``"yolo (auto-detected)"``.

    Returns:
        Formatted header string.
    """
    lines = [
        format_divider("═"),
        f"Analyse: {operation}",
        format_kv("Source", str(label_path)),
        format_kv("Format", format_name),
        "",
    ]
    return "\n".join(lines)


def format_stats_result(
    total_files: int,
    total_annotations: int,
    per_class: Dict[str, int],
    categories: Dict[int, str] = None,
) -> str:
    """Per-class statistics table with summary.

    Args:
        total_files: Number of label files.
        total_annotations: Total annotation objects.
        per_class: ``{class_name: count}`` dict, pre-ordered.
        categories: ``{class_id: class_name}`` dict for ID lookup.
            When provided, an ``ID`` column is added (0-indexed).

    Returns:
        Formatted statistics output.
    """
    lines = [
        format_section("Summary"),
        format_kv("Total files", str(total_files)),
        format_kv("Total annotations", str(total_annotations)),
        format_kv("Categories", str(len(per_class))),
        "",
    ]

    if per_class:
        # Build name→id reverse mapping
        name_to_id: Dict[str, int] = {}
        if categories:
            for cid, cname in categories.items():
                if cname not in name_to_id:
                    name_to_id[cname] = cid

        if categories:
            headers = ["Class", "ID", "Count"]
            rows = [
                [name, str(name_to_id.get(name, "")), str(count)]
                for name, count in per_class.items()
            ]
            rows.append(["─" * 15, "─" * 4, "─" * 7])
            rows.append([
                f"Total ({len(per_class)})",
                "",
                str(sum(per_class.values())),
            ])
        else:
            headers = ["Class", "Count"]
            rows = [[name, str(count)] for name, count in per_class.items()]
            rows.append(["─" * 15, "─" * 7])
            rows.append([
                f"Total ({len(per_class)})",
                str(sum(per_class.values())),
            ])
        lines.append(format_section("Per-Class"))
        lines.append(format_table(headers, rows))
    else:
        lines.append(format_section("Per-Class"))
        lines.append("  (no annotations)")

    return "\n".join(lines)


def format_split_result(
    train_count: int,
    val_count: int,
    train_dir: Path,
    val_dir: Path,
    ratio: float,
    seed: int,
) -> str:
    """Split summary block.

    Args:
        train_count: Number of images in training set.
        val_count: Number of images in validation set.
        train_dir: Output directory for training data.
        val_dir: Output directory for validation data.
        ratio: Train ratio used.
        seed: Random seed used.

    Returns:
        Formatted split summary.
    """
    total = train_count + val_count
    lines = [
        format_kv("Ratio", str(ratio)),
        format_kv("Seed", str(seed)),
        "",
        format_section("Split"),
        format_kv("Train", f"{train_count} images → {train_dir}"),
        format_kv("Val", f"{val_count} images → {val_dir}"),
        format_kv("Total", str(total)),
    ]
    return "\n".join(lines)


def format_analyse_result(
    status: str,
    log_path: str,
) -> str:
    """Final result block.

    Args:
        status: Status string, e.g. ``"✓ Success"``.
        log_path: Path to the log file (or empty string).

    Returns:
        Formatted result block.
    """
    return format_result_block(status, {}, log_path or None)
