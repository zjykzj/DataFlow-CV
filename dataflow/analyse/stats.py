"""
Dataset statistics computation.

``StatsAnalyser`` computes per-class annotation counts for any supported
annotation format (YOLO, LabelMe, COCO).  The format is auto-detected
from the label path.
"""

from pathlib import Path
from typing import Optional

from .base import AnalysisResult, BaseAnalyser, StatsResult
from .log_templates import (
    format_analyse_header,
    format_analyse_result,
    format_stats_result,
)
from .utils import create_handler, detect_format, load_class_names


class StatsAnalyser(BaseAnalyser):
    """Compute dataset statistics for any supported annotation format.

    Constructor: ``StatsAnalyser(log_config=None)``

    Example::

        from dataflow.analyse import StatsAnalyser

        analyser = StatsAnalyser()
        result = analyser.analyse(
            label_path=Path("yolo_labels/"),
            class_file=Path("classes.txt"),
        )
        if result.success:
            stats = result.data
            print(f"Total annotations: {stats.total_annotations}")
    """

    def analyse(
        self,
        label_path: Path,
        class_file: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        sort_by: str = "id",
        descending: bool = False,
    ) -> AnalysisResult:
        """Compute statistics for the dataset at ``label_path``.

        Args:
            label_path: Path to labels — directory (YOLO/LabelMe) or
                JSON file (COCO).
            class_file: Optional classes.txt for name mapping and
                output ordering.  When provided, per-class output
                follows the file's line order regardless of other
                sort settings.
            image_dir: Optional image directory for YOLO format
                (auto-detected if omitted).
            sort_by: Sort key when no class_file is given —
                ``"id"`` (default, class_id ascending) or
                ``"count"`` (annotation count).
            descending: When True, reverse the sort direction.

        Returns:
            ``AnalysisResult`` with ``StatsResult`` in ``.data``.
        """
        result = self._create_result()

        # 1. Detect format
        try:
            fmt = detect_format(label_path)
        except ValueError as e:
            result.add_error(str(e))
            return result

        # 2. Create handler
        try:
            handler = create_handler(
                label_path,
                fmt,
                class_file=class_file,
                image_dir=image_dir,
                logger=self.logger,
            )
        except (ValueError, FileNotFoundError) as e:
            result.add_error(str(e))
            return result

        # 3. Read annotations
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
            result.add_error("Handler returned no data")
            return result

        # 4. Count
        total_files = dataset.num_images
        total_annotations = dataset.num_objects

        # Per-class tally — count by class_name
        per_class_raw: dict[str, int] = {}
        categories = dict(dataset.categories)  # {id: name}

        for image_ann in dataset.images:
            for obj in image_ann.objects:
                name = obj.class_name or f"class_{obj.class_id}"
                if name not in per_class_raw:
                    per_class_raw[name] = 1
                else:
                    per_class_raw[name] += 1

        # 5. Order per_class
        if class_file is not None and class_file.exists():
            # Order by class_file lines, append unknowns
            ordered: dict[str, int] = {}
            try:
                names_in_order = list(load_class_names(class_file).values())
            except (FileNotFoundError, ValueError):
                names_in_order = []
            for name in names_in_order:
                if name in per_class_raw:
                    ordered[name] = per_class_raw.pop(name)
            # Remaining classes — sort alphabetically
            for name in sorted(per_class_raw):
                ordered[name] = per_class_raw[name]
            per_class = ordered
        else:
            # Build name→id mapping for sort_by='id'
            name_to_id: dict[str, int] = {}
            for cid, cname in categories.items():
                if cname not in name_to_id:
                    name_to_id[cname] = cid

            if sort_by == "id":
                key_fn = lambda item: (name_to_id.get(item[0], 999999), item[0])
            elif descending:
                key_fn = lambda item: (-item[1], item[0])
            else:
                key_fn = lambda item: (item[1], item[0])

            per_class = dict(
                sorted(
                    per_class_raw.items(),
                    key=key_fn,
                    reverse=(sort_by == "id" and descending),
                )
            )

        # 6. Build result
        stats_result = StatsResult(
            total_files=total_files,
            total_annotations=total_annotations,
            per_class=per_class,
            format=fmt,
            categories=categories,
        )
        result.data = stats_result
        result.log_path = self._log_manager.log_path

        # 7. Log output
        self._log_info(
            format_analyse_header("Dataset Statistics", label_path, f"{fmt} (auto-detected)")
        )
        self._log_info(
            format_stats_result(total_files, total_annotations, per_class, categories)
        )
        if result.log_path:
            self._log_info(
                format_analyse_result("✓ Success", result.log_path)
            )

        return result
