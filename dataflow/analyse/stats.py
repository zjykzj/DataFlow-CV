"""
Dataset statistics computation.

``StatsAnalyser`` computes per-class annotation counts for any supported
annotation format (YOLO, LabelMe, COCO).  The format is auto-detected
from the label path.

Supports **multiple label paths** (merged into a single result) and
**recursive subdirectory traversal** (``--recursive`` / ``-R``) for
YOLO and LabelMe formats.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

from .base import AnalysisResult, BaseAnalyser, StatsResult
from .log_templates import (
    format_analyse_header,
    format_analyse_result,
    format_stats_path_breakdown,
    format_stats_result,
)
from .utils import (
    _collect_files_recursive,
    _detect_format_recursive,
    _scan_yolo_class_ids,
    create_handler,
    detect_format,
    load_class_names,
)


class StatsAnalyser(BaseAnalyser):
    """Compute dataset statistics for any supported annotation format.

    Constructor: ``StatsAnalyser(log_config=None)``

    Example::

        from dataflow.analyse import StatsAnalyser

        analyser = StatsAnalyser()
        result = analyser.analyse(
            label_paths=[Path("yolo_labels/")],
            class_file=Path("classes.txt"),
        )
        if result.success:
            stats = result.data
            print(f"Total annotations: {stats.total_annotations}")

        # Multi-path with recursive traversal
        result = analyser.analyse(
            label_paths=[Path("proj_a/"), Path("proj_b/")],
            class_file=Path("classes.txt"),
            recursive=True,
        )
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_annotations(dataset) -> tuple:
        """Count files, total annotations, and per-class tally from a dataset.

        Returns:
            ``(total_files, total_annotations, per_class)`` where
            *per_class* is ``{class_name: count}``.
        """
        total_files = dataset.num_images
        total_annotations = dataset.num_objects

        per_class: Dict[str, int] = {}
        for image_ann in dataset.images:
            for obj in image_ann.objects:
                name = obj.class_name or f"class_{obj.class_id}"
                per_class[name] = per_class.get(name, 0) + 1

        return total_files, total_annotations, per_class

    @staticmethod
    def _merge_per_class(all_per_class: List[Dict[str, int]]) -> Dict[str, int]:
        """Merge per-class tally dicts by summing counts for matching names.

        Args:
            all_per_class: List of ``{class_name: count}`` dicts, one
                per source path.

        Returns:
            Merged ``{class_name: total_count}`` dict.
        """
        merged: Dict[str, int] = {}
        for pc in all_per_class:
            for name, count in pc.items():
                merged[name] = merged.get(name, 0) + count
        return merged

    @staticmethod
    def _validate_class_consistency(
        per_class: Dict[str, int],
        class_names_from_file: List[str],
        source_label: str,
    ) -> Optional[str]:
        """Check that all observed classes exist in the class file.

        Args:
            per_class: ``{class_name: count}`` from data.
            class_names_from_file: Class names from the class file
                (values of ``load_class_names()``).
            source_label: Human-readable label for error messages
                (e.g. the path).

        Returns:
            An error message string if unknown classes are found,
            or ``None`` if everything is consistent.
        """
        file_names = set(class_names_from_file)
        data_names = set(per_class.keys())
        unknown = data_names - file_names
        if unknown:
            return (
                f"Categories in data not found in class file: "
                f"{sorted(unknown)}. Source: {source_label}"
            )
        return None

    @staticmethod
    def _order_per_class(
        per_class: Dict[str, int],
        class_file: Optional[Path],
        categories: Dict[int, str],
        sort_by: str,
        descending: bool,
    ) -> Dict[str, int]:
        """Order the per-class tally.

        When *class_file* is provided, order follows the file's line
        order.  Otherwise, order by *sort_by* / *descending*.
        """
        if class_file is not None and class_file.exists():
            # Order by class_file lines
            ordered: Dict[str, int] = {}
            try:
                names_in_order = list(load_class_names(class_file).values())
            except (FileNotFoundError, ValueError):
                names_in_order = []
            for name in names_in_order:
                if name in per_class:
                    ordered[name] = per_class.pop(name)
            # Remaining classes — sort alphabetically
            for name in sorted(per_class):
                ordered[name] = per_class[name]
            return ordered

        # No class_file — sort by id or count
        name_to_id: Dict[str, int] = {}
        for cid, cname in categories.items():
            if cname not in name_to_id:
                name_to_id[cname] = cid

        if sort_by == "id":
            key_fn = lambda item: (name_to_id.get(item[0], 999999), item[0])
        elif descending:
            key_fn = lambda item: (-item[1], item[0])
        else:
            key_fn = lambda item: (item[1], item[0])

        return dict(
            sorted(
                per_class.items(),
                key=key_fn,
                reverse=(sort_by == "id" and descending),
            )
        )

    @staticmethod
    def _normalize_paths(label_paths: Union[Path, List[Path]]) -> List[Path]:
        """Normalize input to a list of Paths.  Single Path → [Path]."""
        if isinstance(label_paths, (str, Path)):
            return [Path(label_paths)]
        return [Path(p) for p in label_paths]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyse(
        self,
        label_paths: Union[Path, List[Path]],
        class_file: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        sort_by: str = "id",
        descending: bool = False,
        recursive: bool = False,
    ) -> AnalysisResult:
        """Compute statistics for one or more datasets.

        Args:
            label_paths: One or more paths to labels — directory
                (YOLO/LabelMe) or JSON file (COCO).  All paths must
                be the same format.
            class_file: Optional classes.txt for name mapping and
                output ordering.  When provided, **strict validation**
                is enforced — any class in the data not present in
                this file causes an ERROR.
            image_dir: Optional image directory for YOLO format
                (auto-detected if omitted).
            sort_by: Sort key when no class_file is given —
                ``"id"`` (default, class_id ascending) or
                ``"count"`` (annotation count).
            descending: When True, reverse the sort direction.
            recursive: When True, recursively find label files in
                subdirectories (YOLO/LabelMe only).  Ignored for COCO.

        Returns:
            ``AnalysisResult`` with ``StatsResult`` in ``.data``.
        """
        result = self._create_result()
        paths = self._normalize_paths(label_paths)
        temp_dirs: List[Path] = []  # for cleanup

        # ---- 0. Validate class_file early -------------------------------
        class_names_from_file: Optional[List[str]] = None
        if class_file is not None:
            if not class_file.exists():
                result.add_error(f"Class file not found: {class_file}")
                return result
            try:
                class_names_from_file = list(
                    load_class_names(class_file).values()
                )
            except (FileNotFoundError, ValueError) as e:
                result.add_error(f"Failed to load class file: {e}")
                return result

        # ---- Per-path state ---------------------------------------------
        fmt: Optional[str] = None
        all_categories: Dict[int, str] = {}
        total_files = 0
        total_annotations = 0
        all_per_class: List[Dict[str, int]] = []
        path_stats: List[Dict[str, object]] = []  # for breakdown display

        for path in paths:
            # ---- 1. Validate path ---------------------------------------
            if not path.exists():
                result.add_error(f"Label path does not exist: {path}")
                return result

            # ---- 2. Detect format ---------------------------------------
            if recursive:
                # For recursive mode, look at all files under the root
                # to determine format, since the root itself may only
                # contain subdirectories.
                try:
                    path_fmt = _detect_format_recursive(path)
                except ValueError as e:
                    result.add_error(str(e))
                    return result
            else:
                try:
                    path_fmt = detect_format(path)
                except ValueError as e:
                    result.add_error(str(e))
                    return result

            if fmt is None:
                fmt = path_fmt
            elif path_fmt != fmt:
                result.add_error(
                    f"All paths must be the same format. "
                    f"Got {fmt} for {paths[0]} but {path} is {path_fmt}"
                )
                return result

            # ---- 3. Recursive file collection (YOLO/LabelMe only) ------
            effective_path = path
            if recursive and fmt in ("yolo", "labelme"):
                try:
                    tmp_dir = _collect_files_recursive(path, fmt)
                    temp_dirs.append(tmp_dir)
                    effective_path = tmp_dir
                except ValueError as e:
                    result.add_error(str(e))
                    return result
            elif recursive and fmt == "coco":
                # COCO is single-file; recursive is a no-op for it
                pass

            # ---- 3a. YOLO pre-scan for strict class validation --------
            if class_names_from_file is not None and fmt == "yolo":
                raw_ids = _scan_yolo_class_ids(effective_path)
                valid_max = len(class_names_from_file) - 1
                invalid_ids = {
                    i for i in raw_ids if i > valid_max or i < 0
                }
                if invalid_ids:
                    result.add_error(
                        f"Class IDs in data not found in class file "
                        f"{class_file.name}: {sorted(invalid_ids)}. "
                        f"Max valid ID is {valid_max}. "
                        f"Source: {path}"
                    )
                    return result

            # ---- 4. Create handler --------------------------------------
            try:
                handler = create_handler(
                    effective_path,
                    fmt,
                    class_file=class_file,
                    image_dir=image_dir,
                    logger=self.logger,
                    skip_image_loading=True,
                )
            except (ValueError, FileNotFoundError) as e:
                result.add_error(str(e))
                return result

            # ---- 5. Read annotations ------------------------------------
            try:
                read_result = handler.read()
            except Exception as e:
                result.add_error(f"Failed to read annotations: {e}")
                return result

            if not read_result.success:
                for err in read_result.errors:
                    result.add_error(err)
                return result

            dataset = read_result.data
            if dataset is None:
                result.add_error(f"Handler returned no data for: {path}")
                return result

            # ---- 6. Count -----------------------------------------------
            p_files, p_anns, p_per_class = self._count_annotations(dataset)
            total_files += p_files
            total_annotations += p_anns
            all_per_class.append(p_per_class)
            all_categories.update(dict(dataset.categories))

            path_stats.append({
                "path": path,
                "files": p_files,
                "annotations": p_anns,
                "recursive": recursive and fmt in ("yolo", "labelme"),
            })

        # ---- 7. Merge per-class -----------------------------------------
        per_class = self._merge_per_class(all_per_class)

        # ---- 8. Strict class validation (when class_file is given) ------
        if class_names_from_file is not None:
            # Check against merged per_class
            file_names = set(class_names_from_file)
            data_names = set(per_class.keys())
            unknown = data_names - file_names
            if unknown:
                result.add_error(
                    f"Categories in data not found in class file "
                    f"{class_file.name}: {sorted(unknown)}. "
                    f"Sources: {', '.join(str(p) for p in paths)}"
                )
                return result

        # ---- 9. Order per_class -----------------------------------------
        per_class = self._order_per_class(
            per_class, class_file, all_categories, sort_by, descending
        )

        # ---- 10. Build result -------------------------------------------
        stats_result = StatsResult(
            total_files=total_files,
            total_annotations=total_annotations,
            per_class=per_class,
            format=fmt,
            categories=all_categories,
            source_paths=list(paths),
        )
        result.data = stats_result
        result.log_path = self._log_manager.log_path

        # ---- 11. Log output ---------------------------------------------
        self._log_info(
            format_analyse_header(
                "Dataset Statistics",
                paths,
                f"{fmt} (auto-detected)",
                class_file=class_file,
                recursive=recursive and fmt in ("yolo", "labelme"),
            )
        )
        if len(paths) > 1:
            self._log_info(format_stats_path_breakdown(path_stats))
        self._log_info(
            format_stats_result(
                total_files, total_annotations, per_class, all_categories
            )
        )
        if result.log_path:
            self._log_info(
                format_analyse_result("✓ Success", result.log_path)
            )

        # ---- 12. Cleanup temp dirs --------------------------------------
        for td in temp_dirs:
            try:
                shutil.rmtree(td, ignore_errors=True)
            except OSError:
                pass

        return result
