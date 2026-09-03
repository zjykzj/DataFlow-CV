"""
Category-based annotation filtering.

``FilterAnalyser`` filters dataset annotations by category, retaining
only specified classes and re-mapping class IDs to match the new class
file order.  Supports all annotation formats (YOLO, LabelMe, COCO).
The format is auto-detected from the label path.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional

from .base import (
    AnalysisResult,
    BaseAnalyser,
    CategoryMapping,
    FilterResult,
    RemovedCategory,
)
from .log_templates import (
    format_analyse_header,
    format_analyse_result,
    format_filter_result,
)
from .utils import create_handler, detect_format, load_class_names
from dataflow.label.models import DatasetAnnotations, ImageAnnotation, ObjectAnnotation


class FilterAnalyser(BaseAnalyser):
    """Filter dataset annotations by category.

    Keeps only the categories listed in ``new_class_file``, remapping
    their IDs to match the new file's line order.  Both original and
    new class files are **required** — unlike stats/split, there is
    no auto-detection fallback.

    Constructor: ``FilterAnalyser(log_config=None)``

    Example::

        from dataflow.analyse import FilterAnalyser

        analyser = FilterAnalyser()
        result = analyser.analyse(
            label_path=Path("yolo_labels/"),
            original_class_file=Path("classes.txt"),
            new_class_file=Path("new_classes.txt"),
            output_dir=Path("filtered_output/"),
        )
        if result.success:
            fr = result.data
            print(f"Kept {fr.total_annotations_after} annotations "
                  f"across {fr.total_files_with_annotations} files")
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter_mapping(
        new_classes: Dict[int, str],
        data_categories: Dict[int, str],
        logger,
    ):
        """Build the filter mapping from actual data categories.

        Uses *data_categories* (from handler or dataset) as the
        source-of-truth for old_id values, because IDs can differ
        across formats (e.g., COCO uses 1-indexed IDs while YOLO
        uses 0-indexed).

        Args:
            new_classes: ``{new_id: name}`` from the target class file.
            data_categories: ``{actual_id: name}`` from the handler
                or dataset (source-of-truth for old IDs).
            logger: Logger for warnings.

        Returns:
            ``(old_to_new, kept, removed, missing)`` tuple:
            - *old_to_new*: ``{old_id: CategoryMapping}``
            - *kept*: ``[CategoryMapping, ...]`` in new_id order
            - *removed*: ``[RemovedCategory, ...]``
            - *missing*: ``[str, ...]`` — names in new_classes not in
              data_categories
        """
        # name → actual old_id
        name_to_old_id: Dict[str, int] = {}
        for old_id, name in data_categories.items():
            if name not in name_to_old_id:
                name_to_old_id[name] = old_id

        old_to_new: Dict[int, CategoryMapping] = {}
        kept: List[CategoryMapping] = []
        missing: List[str] = []

        for new_id, name in new_classes.items():
            if name in name_to_old_id:
                old_id = name_to_old_id[name]
                mapping = CategoryMapping(new_id=new_id, old_id=old_id, name=name)
                old_to_new[old_id] = mapping
                kept.append(mapping)
            else:
                missing.append(name)
                if logger:
                    logger.warning(
                        f'Category "{name}" in new class file not found in source — skipping'
                    )

        # Build removed list
        kept_old_ids = set(old_to_new.keys())
        removed: List[RemovedCategory] = []
        for old_id, name in data_categories.items():
            if old_id not in kept_old_ids:
                removed.append(RemovedCategory(old_id=old_id, name=name))

        return old_to_new, kept, removed, missing

    @staticmethod
    def _filter_dataset_images(
        dataset: "DatasetAnnotations",
        old_to_new: Dict[int, CategoryMapping],
    ):
        """Filter and remap objects in-place across all images.

        Returns (total_files_with_annotations, total_after).
        """
        total_files_with_annotations = 0
        total_after = 0

        for image_ann in dataset.images:
            filtered = [obj for obj in image_ann.objects if obj.class_id in old_to_new]
            for obj in filtered:
                mapping = old_to_new[obj.class_id]
                obj.class_id = mapping.new_id
                obj.class_name = mapping.name
            image_ann.objects = filtered
            if filtered:
                total_files_with_annotations += 1
            total_after += len(filtered)

        return total_files_with_annotations, total_after

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyse(
        self,
        label_path: Path,
        original_class_file: Path,
        new_class_file: Path,
        output_dir: Path,
        image_dir: Optional[Path] = None,
    ) -> AnalysisResult:
        """Filter annotations, keeping only categories in ``new_class_file``.

        Args:
            label_path: Path to labels — directory (YOLO/LabelMe) or
                JSON file (COCO).
            original_class_file: Source class file defining all
                categories in the source dataset.
            new_class_file: Target class file defining which categories
                to keep and their new order (line index = new class ID).
            output_dir: Output root directory.
            image_dir: Optional image directory for YOLO format
                (auto-detected if omitted).

        Returns:
            ``AnalysisResult`` with ``FilterResult`` in ``.data``.
        """
        result = self._create_result()

        # ---- 1. Validate both class files exist -----------------------
        for label, path in [
            ("Original class", original_class_file),
            ("New class", new_class_file),
        ]:
            if not path.exists():
                result.add_error(f"{label} file not found: {path}")
                return result

        # ---- 2. Load class file contents ------------------------------
        try:
            original_classes = load_class_names(original_class_file)
        except (FileNotFoundError, ValueError) as e:
            result.add_error(f"Failed to load original class file: {e}")
            return result

        try:
            new_classes = load_class_names(new_class_file)
        except (FileNotFoundError, ValueError) as e:
            result.add_error(f"Failed to load new class file: {e}")
            return result

        if not new_classes:
            result.add_error(f"No valid class names in new class file: {new_class_file}")
            return result

        # ---- 3. Detect format + create handler ------------------------
        try:
            fmt = detect_format(label_path)
        except ValueError as e:
            result.add_error(str(e))
            return result

        try:
            handler = create_handler(
                label_path,
                fmt,
                class_file=original_class_file,
                image_dir=image_dir,
                logger=self.logger,
            )
        except (ValueError, FileNotFoundError) as e:
            result.add_error(str(e))
            return result

        # ---- 4. Build filter mapping from ACTUAL data categories ------
        # We use the handler's / dataset's categories as source-of-truth
        # for old_id values because IDs differ across formats:
        #   YOLO:   0-indexed from classes.txt
        #   COCO:   COCO's own IDs (usually 1-indexed)
        #   LabelMe: 0-indexed discovery order
        if fmt == "yolo":
            # YOLO handler populates categories from class_file in __init__
            data_categories = dict(handler.categories)
        else:
            # LabelMe / COCO — need to read first to get actual categories
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
            data_categories = dict(dataset.categories)

        old_to_new, kept_categories, removed_categories, missing_categories = (
            self._build_filter_mapping(new_classes, data_categories, self.logger)
        )

        if not old_to_new:
            result.add_error("No matching categories between source and new class file")
            return result

        # ---- 5. Ensure output directory --------------------------------
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            result.add_error(f"Cannot create output directory: {e}")
            return result

        # ---- 6. Filter and write ---------------------------------------
        total_files = 0
        total_files_with_annotations = 0
        total_before = 0
        total_after = 0

        if fmt == "yolo":
            # Streaming path: iterate, filter, write_one per image.
            # Create a separate write handler with the new class file so
            # that _object_to_yolo_line() uses the correct category IDs.
            from dataflow.label.yolo_handler import YoloAnnotationHandler

            write_handler = YoloAnnotationHandler(
                label_dir=str(output_dir),
                class_file=str(new_class_file),
                image_dir=str(output_dir),
                strict_mode=False,
                logger=self.logger,
            )

            try:
                for image_ann in handler.iter_images():
                    total_files += 1
                    total_before += len(image_ann.objects)

                    filtered_objects = []
                    for obj in image_ann.objects:
                        mapping = old_to_new.get(obj.class_id)
                        if mapping is not None:
                            # Create a new object with remapped ID/name
                            # rather than mutating the original (avoid
                            # aliasing — the original may be reused by the
                            # iterator or shared across images).
                            filtered_objects.append(
                                ObjectAnnotation(
                                    class_id=mapping.new_id,
                                    class_name=mapping.name,
                                    bbox=obj.bbox,
                                    segmentation=obj.segmentation,
                                    confidence=obj.confidence,
                                    is_crowd=obj.is_crowd,
                                )
                            )
                    total_after += len(filtered_objects)

                    if filtered_objects:
                        total_files_with_annotations += 1

                    filtered_img = ImageAnnotation(
                        image_path=image_ann.image_path,
                        image_id=image_ann.image_id,
                        width=image_ann.width,
                        height=image_ann.height,
                        objects=filtered_objects,
                    )

                    wr = write_handler.write_one(filtered_img, output_dir)
                    if not wr.success:
                        for err in wr.errors:
                            result.add_error(f"Write {image_ann.image_id}: {err}")
                        return result
            except Exception as e:
                result.add_error(f"Failed during streaming filter: {e}")
                return result

        elif fmt == "labelme":
            # Semi-streaming: read all for categories, filter in-memory,
            # write_one per image.
            total_files = dataset.num_images
            total_before = dataset.num_objects

            total_files_with_annotations, total_after = self._filter_dataset_images(
                dataset, old_to_new
            )

            # Update categories
            dataset.categories = {km.new_id: km.name for km in kept_categories}

            try:
                for image_ann in dataset.images:
                    wr = handler.write_one(image_ann, output_dir)
                    if not wr.success:
                        for err in wr.errors:
                            result.add_error(f"Write {image_ann.image_id}: {err}")
                        return result
            except Exception as e:
                result.add_error(f"Failed to write filtered output: {e}")
                return result

        else:  # coco
            # Batch path: read, filter in-memory, write single JSON.
            total_files = dataset.num_images
            total_before = dataset.num_objects

            total_files_with_annotations, total_after = self._filter_dataset_images(
                dataset, old_to_new
            )

            # Update categories to match new class file
            dataset.categories = {km.new_id: km.name for km in kept_categories}

            output_path = output_dir / label_path.name
            try:
                write_result = handler.write(dataset, str(output_path))
                if not write_result.success:
                    for err in write_result.errors:
                        result.add_error(f"Write: {err}")
                    return result
            except Exception as e:
                result.add_error(f"Failed to write filtered output: {e}")
                return result

        # ---- 7. Copy new class file to output_dir ----------------------
        try:
            shutil.copy2(str(new_class_file), str(output_dir / "classes.txt"))
        except OSError as e:
            result.add_warning(f"Could not copy class file: {e}")

        # ---- 8. Build result -------------------------------------------
        filter_result = FilterResult(
            total_files=total_files,
            total_files_with_annotations=total_files_with_annotations,
            total_annotations_before=total_before,
            total_annotations_after=total_after,
            kept_categories=kept_categories,
            removed_categories=removed_categories,
            missing_categories=missing_categories,
            output_dir=output_dir,
            format=fmt,
        )
        result.data = filter_result
        result.log_path = self._log_manager.log_path

        if missing_categories:
            s = "y" if len(missing_categories) == 1 else "ies"
            result.add_warning(
                f"{len(missing_categories)} categor{s} in new class file not found in source"
            )

        if total_after == 0 and total_before > 0:
            result.add_warning("All annotations were filtered out — output files are empty")

        # ---- 9. Log output ---------------------------------------------
        self._log_info(
            format_analyse_header("Category Filter", label_path, f"{fmt} (auto-detected)")
        )
        self._log_info(
            f"  Original class: {original_class_file.name} ({len(original_classes)} categories)"
        )
        self._log_info(f"  New class:      {new_class_file.name} ({len(new_classes)} categories)\n")
        self._log_info(
            format_filter_result(
                total_files=total_files,
                total_files_with_annotations=total_files_with_annotations,
                annotations_before=total_before,
                annotations_after=total_after,
                kept_categories=kept_categories,
                removed_categories=removed_categories,
                missing_categories=missing_categories,
                output_dir=output_dir,
            )
        )
        if result.log_path:
            self._log_info(format_analyse_result("✓ Success", result.log_path))

        return result
