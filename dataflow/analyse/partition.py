"""
N-way dataset partitioning.

``PartitionAnalyser`` splits a dataset into N roughly-equal subsets.
Supports three modes: labels-only, images-only, or labels+images together
(where labels drive the partition and images follow by stem matching).

Supports YOLO and LabelMe formats only (not COCO — single-file JSONs
should use ``SplitAnalyser``).
"""

import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import AnalysisResult, BaseAnalyser, PartitionResult
from .log_templates import (
    format_analyse_header,
    format_analyse_result,
    format_partition_result,
)
from .utils import (
    _collect_image_files,
    _copy_or_move_file,
    _IMAGE_EXTENSIONS,
    create_handler,
    detect_format,
    load_class_names,
)


class PartitionAnalyser(BaseAnalyser):
    """Partition dataset into N roughly-equal subsets.

    Constructor: ``PartitionAnalyser(log_config=None)``

    Supports three modes:
    - **Labels-only**: ``--label-dir`` only — partition label files.
    - **Images-only**: ``--image-dir`` only — partition image files.
    - **Both**: labels drive partition; images matched by stem.

    Example::

        from dataflow.analyse import PartitionAnalyser

        analyser = PartitionAnalyser()
        result = analyser.analyse(
            output_dir=Path("output/"),
            num=3,
            label_dir=Path("yolo_labels/"),
            image_dir=Path("images/"),
            shuffle=True,
            seed=42,
            class_file=Path("classes.txt"),
        )
        if result.success:
            pr = result.data
            print(f"Partitions: {pr.partition_sizes}")
    """

    def analyse(
        self,
        output_dir: Path,
        num: int,
        label_dir: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        shuffle: bool = False,
        seed: int = 42,
        class_file: Optional[Path] = None,
        move: bool = False,
    ) -> AnalysisResult:
        """Partition the dataset into ``num`` subsets.

        At least one of ``label_dir`` or ``image_dir`` must be provided.

        Args:
            output_dir: Output root directory.
            num: Number of partitions (>= 2).
            label_dir: Optional label directory (YOLO or LabelMe).
            image_dir: Optional image directory.
            shuffle: When True, randomly shuffle before partitioning.
            seed: Random seed for shuffle reproducibility.
            class_file: Optional classes.txt for label mode.
            move: When True, move source files instead of copying.

        Returns:
            ``AnalysisResult`` with ``PartitionResult`` in ``.data``.
        """
        result = self._create_result()

        # ------------------------------------------------------------------
        # 1. Validate inputs
        # ------------------------------------------------------------------
        if num < 2:
            result.add_error(
                f"Number of partitions must be at least 2, got: {num}"
            )
            return result

        if label_dir is None and image_dir is None:
            result.add_error(
                "At least one of label_dir or image_dir must be provided"
            )
            return result

        # ------------------------------------------------------------------
        # 2. Determine mode
        # ------------------------------------------------------------------
        fmt = ""
        handler = None
        items: list  # List of ImageAnnotation or (Path, stem) tuples

        if label_dir is not None:
            # Detect format and validate
            try:
                fmt = detect_format(label_dir)
            except ValueError as e:
                result.add_error(str(e))
                return result

            if fmt == "coco":
                result.add_error(
                    "partition does not support COCO format. "
                    "COCO is a single JSON file — "
                    "use 'analyse split' for train/val split."
                )
                return result

            # Create handler and read all annotations
            try:
                handler = create_handler(
                    label_dir,
                    fmt,
                    class_file=class_file,
                    image_dir=image_dir,
                    logger=self.logger,
                )
            except (ValueError, FileNotFoundError) as e:
                result.add_error(str(e))
                return result

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

            # Sort images by image_id for stable ordering
            items = sorted(dataset.images, key=lambda img: img.image_id)

            if image_dir is not None:
                mode = "both"
            else:
                mode = "labels"
        else:
            # Images-only mode
            mode = "images"
            items = _collect_image_files(image_dir)

        total = len(items)

        if total == 0:
            result.add_warning("Dataset is empty — nothing to partition")
            pr = PartitionResult(
                num_partitions=num,
                partition_sizes=[0] * num,
                partition_dirs=[],
                total_files=0,
                seed=seed,
                shuffle=shuffle,
                mode=mode,
                format=fmt,
                move=move,
            )
            result.data = pr
            return result

        # ------------------------------------------------------------------
        # 3. Shuffle (optional)
        # ------------------------------------------------------------------
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(items)

        # ------------------------------------------------------------------
        # 4. Partition: divide into N parts
        # ------------------------------------------------------------------
        base_size = total // num
        remainder = total % num

        partition_sizes: List[int] = []
        partition_ranges: List[Tuple[int, int]] = []  # (start, end)

        idx = 0
        for i in range(num):
            size = base_size + (1 if i >= num - remainder else 0)
            partition_sizes.append(size)
            partition_ranges.append((idx, idx + size))
            idx += size

        # ------------------------------------------------------------------
        # 5. Create output directory and write per partition
        # ------------------------------------------------------------------
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            result.add_error(f"Cannot create output directory: {e}")
            return result

        partition_dirs: List[Path] = []
        image_stems: Dict[str, Path] = {}  # stem → source path, for both mode

        if mode == "both":
            # Pre-index image files by stem for fast lookup
            for p in sorted(image_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
                    stem = p.stem
                    if stem not in image_stems:
                        image_stems[stem] = p

        for i, (start, end) in enumerate(partition_ranges):
            part_dir = output_dir / f"part_{i + 1}"
            part_dir.mkdir(parents=True, exist_ok=True)
            partition_dirs.append(part_dir)

            if mode == "images":
                # Images-only: copy/move image files directly
                for img_path, _stem in items[start:end]:
                    _copy_or_move_file(img_path, part_dir, move, self.logger)

            elif mode == "labels":
                # Labels-only: write labels via handler
                for img_ann in items[start:end]:
                    try:
                        wr = handler.write_one(img_ann, part_dir)
                        if not wr.success:
                            for err in wr.errors:
                                result.add_error(
                                    f"Write {part_dir.name}/"
                                    f"{img_ann.image_id}: {err}"
                                )
                    except Exception as e:
                        result.add_error(
                            f"Write {part_dir.name}/"
                            f"{img_ann.image_id}: {e}"
                        )

                # Move label files if in move mode
                if move:
                    for img_ann in items[start:end]:
                        # Construct the source label path
                        if fmt == "yolo":
                            src_label = label_dir / f"{img_ann.image_id}.txt"
                        else:  # labelme
                            src_label = label_dir / f"{img_ann.image_id}.json"
                        if src_label.exists():
                            _copy_or_move_file(
                                src_label, part_dir, move=True, logger=self.logger
                            )

            elif mode == "both":
                labels_subdir = part_dir / "labels"
                images_subdir = part_dir / "images"
                labels_subdir.mkdir(parents=True, exist_ok=True)
                images_subdir.mkdir(parents=True, exist_ok=True)

                for img_ann in items[start:end]:
                    # Write label
                    try:
                        wr = handler.write_one(img_ann, labels_subdir)
                        if not wr.success:
                            for err in wr.errors:
                                result.add_error(
                                    f"Write {part_dir.name}/labels/"
                                    f"{img_ann.image_id}: {err}"
                                )
                    except Exception as e:
                        result.add_error(
                            f"Write {part_dir.name}/labels/"
                            f"{img_ann.image_id}: {e}"
                        )

                    # Match and copy/move image
                    stem = img_ann.image_id
                    if stem in image_stems:
                        _copy_or_move_file(
                            image_stems[stem], images_subdir, move, self.logger
                        )
                    else:
                        self._log_warning(
                            f"No matching image found for label "
                            f"'{img_ann.image_id}' in {image_dir}"
                        )

                # Move label files if in move mode
                if move:
                    for img_ann in items[start:end]:
                        if fmt == "yolo":
                            src_label = label_dir / f"{img_ann.image_id}.txt"
                        else:
                            src_label = label_dir / f"{img_ann.image_id}.json"
                        if src_label.exists():
                            _copy_or_move_file(
                                src_label, labels_subdir,
                                move=True, logger=self.logger
                            )

                # Report unmatched images (in image_dir but not in labels)
                if not move:  # Only warn for copy mode; move mode self-resolves
                    label_stems = {
                        img_ann.image_id
                        for img_ann in items[start:end]
                    }
                    for stem, img_path in image_stems.items():
                        if stem not in label_stems:
                            self._log_warning(
                                f"Image '{img_path.name}' has no matching "
                                f"label — skipped"
                            )

            # Copy class_file to partition directory
            if class_file is not None and class_file.exists():
                try:
                    target_cf = part_dir / class_file.name
                    if not target_cf.exists():
                        shutil.copy2(str(class_file), str(target_cf))
                except OSError as e:
                    result.add_warning(
                        f"Could not copy class file to "
                        f"{part_dir.name}: {e}"
                    )

        # ------------------------------------------------------------------
        # 6. Build result
        # ------------------------------------------------------------------
        pr = PartitionResult(
            num_partitions=num,
            partition_sizes=partition_sizes,
            partition_dirs=partition_dirs,
            total_files=total,
            seed=seed,
            shuffle=shuffle,
            mode=mode,
            format=fmt,
            move=move,
        )
        result.data = pr
        result.log_path = self._log_manager.log_path

        # ------------------------------------------------------------------
        # 7. Log output
        # ------------------------------------------------------------------
        mode_label = {"images": "Images Only", "labels": "Labels Only",
                      "both": "Labels + Images"}[mode]
        label_paths = label_dir if label_dir else image_dir
        self._log_info(
            format_analyse_header(
                f"Dataset Partition ({mode_label})",
                label_paths,
                f"{fmt} (auto-detected)" if fmt else "images only",
                class_file=class_file,
            )
        )
        self._log_info(
            format_partition_result(
                num_partitions=num,
                partition_sizes=partition_sizes,
                partition_dirs=partition_dirs,
                total_files=total,
                seed=seed,
                shuffle=shuffle,
                mode=mode,
                move=move,
            )
        )
        if result.log_path:
            self._log_info(
                format_analyse_result("✓ Success", result.log_path)
            )

        return result
