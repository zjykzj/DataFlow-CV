"""
Train/test dataset splitting.

``SplitAnalyser`` splits a dataset into training and validation subsets
with deterministic shuffling.  Supports all annotation formats (YOLO,
LabelMe, COCO).  The format is auto-detected from the label path.
"""

import random
import shutil
from pathlib import Path
from typing import Optional

from .base import AnalysisResult, BaseAnalyser, SplitResult
from .log_templates import (
    format_analyse_header,
    format_analyse_result,
    format_split_result,
)
from .utils import create_handler, detect_format, load_class_names


class SplitAnalyser(BaseAnalyser):
    """Split dataset into train/val subsets.

    Constructor: ``SplitAnalyser(log_config=None)``

    Example::

        from dataflow.analyse import SplitAnalyser

        analyser = SplitAnalyser()
        result = analyser.analyse(
            label_path=Path("yolo_labels/"),
            output_dir=Path("split_output/"),
            ratio=0.8,
            seed=42,
            class_file=Path("classes.txt"),
        )
        if result.success:
            split = result.data
            print(f"Train: {split.train_count}, Val: {split.val_count}")
    """

    def analyse(
        self,
        label_path: Path,
        output_dir: Path,
        ratio: float = 0.8,
        seed: int = 42,
        class_file: Optional[Path] = None,
    ) -> AnalysisResult:
        """Split the dataset at ``label_path`` into train and val.

        Args:
            label_path: Path to labels — directory (YOLO/LabelMe) or
                JSON file (COCO).
            output_dir: Output root directory (``train/`` and ``val/``
                subdirectories are created inside).
            ratio: Proportion of data for training (default 0.8).
            seed: Random seed for reproducible shuffling.
            class_file: Optional classes.txt.  Required for YOLO format
                (copied to both output directories).

        Returns:
            ``AnalysisResult`` with ``SplitResult`` in ``.data``.
        """
        result = self._create_result()

        # Validate ratio
        if not 0.0 < ratio < 1.0:
            result.add_error(
                f"Ratio must be between 0 and 1 (exclusive), got: {ratio}"
            )
            return result

        # 1. Detect format
        try:
            fmt = detect_format(label_path)
        except ValueError as e:
            result.add_error(str(e))
            return result

        # 2. Create handler for reading
        try:
            handler = create_handler(
                label_path,
                fmt,
                class_file=class_file,
                logger=self.logger,
            )
        except (ValueError, FileNotFoundError) as e:
            result.add_error(str(e))
            return result

        # 3. Read all annotations
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

        total = dataset.num_images
        if total == 0:
            result.add_warning("Dataset is empty — nothing to split")
            split_result = SplitResult(
                train_count=0,
                val_count=0,
                train_dir=output_dir / "train",
                val_dir=output_dir / "val",
                ratio=ratio,
                seed=seed,
                format=fmt,
            )
            result.data = split_result
            return result

        # 4. Shuffle
        images = list(dataset.images)
        rng = random.Random(seed)
        rng.shuffle(images)

        # 5. Split
        split_idx = int(len(images) * ratio)
        # Ensure at least 1 image in each split when dataset has ≥2 images
        if split_idx == 0 and len(images) >= 2:
            split_idx = 1
        elif split_idx == len(images) and len(images) >= 2:
            split_idx = len(images) - 1

        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # 6. Create output DatasetAnnotations
        from dataflow.label.models import AnnotationFormat, DatasetAnnotations

        format_enum = {
            "yolo": AnnotationFormat.YOLO,
            "labelme": AnnotationFormat.LABELME,
            "coco": AnnotationFormat.COCO,
        }[fmt]

        train_ds = DatasetAnnotations(
            images=list(train_images),
            categories=dict(dataset.categories),
            format=format_enum,
            dataset_info=dict(dataset.dataset_info),
        )
        val_ds = DatasetAnnotations(
            images=list(val_images),
            categories=dict(dataset.categories),
            format=format_enum,
            dataset_info=dict(dataset.dataset_info),
        )

        # 7. Write outputs
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"

        try:
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            result.add_error(f"Cannot create output directory: {e}")
            return result

        if fmt == "coco":
            # Batch write — one JSON per split
            train_path = output_dir / "train.json"
            val_path = output_dir / "val.json"
            try:
                handler.write(train_ds, str(train_path))
                handler.write(val_ds, str(val_path))
            except Exception as e:
                result.add_error(f"Failed to write split output: {e}")
                return result
        else:
            # Streaming write — per-file for YOLO and LabelMe
            try:
                for img in train_ds.images:
                    handler.write_one(img, train_dir)
                for img in val_ds.images:
                    handler.write_one(img, val_dir)
            except Exception as e:
                result.add_error(f"Failed to write split output: {e}")
                return result

        # 8. Copy class_file to both output dirs if provided
        if class_file is not None and class_file.exists():
            try:
                shutil.copy2(str(class_file), str(train_dir / class_file.name))
                shutil.copy2(str(class_file), str(val_dir / class_file.name))
            except OSError as e:
                result.add_warning(f"Could not copy class file: {e}")

        # 9. Build result
        split_result = SplitResult(
            train_count=len(train_images),
            val_count=len(val_images),
            train_dir=train_dir,
            val_dir=val_dir,
            ratio=ratio,
            seed=seed,
            format=fmt,
        )
        result.data = split_result
        result.log_path = self._log_manager.log_path

        # 10. Log output
        self._log_info(
            format_analyse_header("Train/Test Split", label_path, fmt)
        )
        self._log_info(
            format_split_result(
                len(train_images),
                len(val_images),
                train_dir,
                val_dir,
                ratio,
                seed,
            )
        )
        if result.log_path:
            self._log_info(
                format_analyse_result("✓ Success", result.log_path)
            )

        return result
