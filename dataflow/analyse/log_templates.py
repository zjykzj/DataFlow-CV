"""
Log formatting templates for the Analyse module.

Pure functions that return formatted strings — they never call the
logger directly.  The caller decides the log level and passes the
result to ``self.logger.info()`` / ``.debug()`` / etc.
"""

from pathlib import Path
from typing import Dict, List, Optional

from ..util.logging import (
    format_divider,
    format_kv,
    format_result_block,
    format_section,
    format_table,
)


def format_analyse_header(
    operation: str,
    label_paths,
    format_name: str,
    class_file: Optional[Path] = None,
    recursive: bool = False,
) -> str:
    """Header block shown at the start of an analysis operation.

    Args:
        operation: Operation name, e.g. ``"Dataset Statistics"``.
        label_paths: Single ``Path`` or list of ``Path`` objects.
        format_name: Detected format, e.g. ``"yolo (auto-detected)"``.
        class_file: Optional class file path for display.
        recursive: Whether recursive traversal was used.

    Returns:
        Formatted header string.
    """
    lines = [
        format_divider("═"),
        f"Analyse: {operation}",
    ]

    # Normalise to list
    if isinstance(label_paths, (str, Path)):
        path_list = [Path(label_paths)]
    else:
        path_list = list(label_paths)

    if len(path_list) == 1:
        label = f"{path_list[0]}"
        if recursive:
            label += " (recursive)"
        lines.append(format_kv("Source", label))
    else:
        lines.append(
            format_kv("Sources", f"{', '.join(str(p) for p in path_list)}"
                      f"    ({len(path_list)} paths)")
        )

    if class_file is not None:
        lines.append(format_kv("Class file", str(class_file)))
    lines.append(format_kv("Format", format_name))
    lines.append("")
    return "\n".join(lines)


def format_stats_path_breakdown(path_stats) -> str:
    """Per-path file and annotation counts (multi-path only).

    Args:
        path_stats: List of dicts with keys ``path`` (Path), ``files``
            (int), ``annotations`` (int), ``recursive`` (bool).

    Returns:
        Formatted path breakdown section.
    """
    lines = [format_section("Path Breakdown")]
    for ps in path_stats:
        label = f"{ps['path']}"
        if ps.get("recursive"):
            label += " (recursive)"
        lines.append(
            f"  {label:<40} "
            f"{ps['files']:>5} files, "
            f"{ps['annotations']:>5} annotations"
        )
    lines.append("")
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


def format_filter_result(
    total_files: int,
    total_files_with_annotations: int,
    annotations_before: int,
    annotations_after: int,
    kept_categories: List,
    removed_categories: List,
    missing_categories: List[str],
    output_dir: Path,
) -> str:
    """Filter comparison + summary block.

    Args:
        total_files: Total label files processed.
        total_files_with_annotations: Files that still have annotations.
        annotations_before: Annotation count before filtering.
        annotations_after: Annotation count after filtering.
        kept_categories: ``[CategoryMapping, ...]`` — kept categories
            with {new_id, old_id, name}.
        removed_categories: ``[RemovedCategory, ...]`` — discarded
            categories with {old_id, name}.
        missing_categories: ``[str, ...]`` — class names in the new
            class file that were not found in the source data.
        output_dir: Output directory.

    Returns:
        Formatted filter comparison + summary.
    """
    lines: List[str] = []

    # ── Category Comparison ──
    lines.append(format_section("Category Comparison"))

    # Kept (remapped)
    kept_count = len(kept_categories)
    lines.append(f"  Kept (remapped):     {kept_count} categories")
    for km in kept_categories:
        lines.append(f"    [{km.new_id}] {km.name:<12} (was: class_id={km.old_id})")
    lines.append("")

    # Removed
    removed_count = len(removed_categories)
    lines.append(f"  Removed:             {removed_count} categories")
    for rc in removed_categories:
        lines.append(f'    class_id={rc.old_id:<3} "{rc.name}"')
    lines.append("")

    # Not found in source
    missing_count = len(missing_categories)
    if missing_count > 0:
        lines.append(f"  Not found in source: {missing_count} categories")
        for name in missing_categories:
            lines.append(f'    "{name}"')
    else:
        lines.append(f"  Not found in source: 0 categories")
    lines.append("")

    # ── Filter Summary ──
    lines.append(format_section("Filter Summary"))
    lines.append(format_kv("Total files", str(total_files)))
    lines.append(format_kv("Files with annotations", str(total_files_with_annotations)))
    lines.append(format_kv("Annotations before", str(annotations_before)))
    lines.append(format_kv("Annotations after", str(annotations_after)))
    lines.append(format_kv(
        "Output",
        f"{total_files_with_annotations} files → {output_dir}",
    ))

    return "\n".join(lines)


def format_partition_result(
    num_partitions: int,
    partition_sizes: List[int],
    partition_dirs: List[Path],
    total_files: int,
    seed: int,
    shuffle: bool,
    mode: str,
    move: bool,
) -> str:
    """Partition summary with per-partition breakdown.

    Args:
        num_partitions: Number of partitions (N).
        partition_sizes: File count per partition.
        partition_dirs: Output directory per partition.
        total_files: Total files processed.
        seed: Random seed used (meaningful when ``shuffle=True``).
        shuffle: Whether shuffle was applied.
        mode: ``"images"`` | ``"labels"`` | ``"both"``.
        move: Whether move mode was used.

    Returns:
        Formatted partition summary.
    """
    mode_label = {"images": "Images only", "labels": "Labels only",
                  "both": "Labels + Images"}[mode]

    lines = [
        format_kv("Mode", mode_label),
        format_kv("Partitions", str(num_partitions)),
        format_kv("Shuffle", f"{'Yes' if shuffle else 'No'}"
                  f"{f' (seed={seed})' if shuffle else ''}"),
        format_kv("Move", "Yes" if move else "No"),
        format_kv("Total files", str(total_files)),
        "",
        format_section("Partition"),
    ]

    for i in range(num_partitions):
        lines.append(
            f"  Part {i + 1}:  {partition_sizes[i]:>6} files → "
            f"{partition_dirs[i]}"
        )

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
