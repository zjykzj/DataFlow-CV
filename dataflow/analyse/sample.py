"""
Dataset sampling — collect N files from a dataset.

``SampleAnalyser`` collects a fixed number of files from a dataset,
supporting random or sequential selection.  Three modes are supported:
labels-only, images-only, or both (labels drive sampling, images
matched by file stem).

The operation is a pure file-level operation — annotation content is
never parsed, making it fast and dependency-free.
"""

import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import AnalysisResult, BaseAnalyser, SampleResult
from .log_templates import (
    format_analyse_header,
    format_analyse_result,
    format_sample_result,
)
from .utils import (
    _IMAGE_EXTENSIONS,
    _collect_image_files,
    _collect_label_files,
    _copy_or_move_file,
    detect_format,
)


class SampleAnalyser(BaseAnalyser):
    """Collect N files from a dataset.

    Constructor: ``SampleAnalyser(log_config=None)``

    Supports three modes (auto-detected from provided inputs):

    - **Labels-only**: ``-l/--label-dir`` only — sample label files.
    - **Images-only**: ``-i/--image-dir`` only — sample image files.
    - **Both**: labels drive sampling; images matched by stem.

    Only YOLO and LabelMe formats are supported (not COCO).

    Example::

        from dataflow.analyse import SampleAnalyser

        analyser = SampleAnalyser()
        result = analyser.analyse(
            output_dir=Path("sampled/"),
            count=10,
            label_dir=Path("yolo_labels/"),
            shuffle=True,
            seed=42,
        )
        if result.success:
            sr = result.data
            print(f"Collected {sr.sampled_count}/{sr.total_count} files")
    """

    def analyse(
        self,
        output_dir: Path,
        count: int,
        label_dir: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        shuffle: bool = True,
        seed: int = 42,
        class_file: Optional[Path] = None,
        move: bool = False,
    ) -> AnalysisResult:
        """Collect *count* files from the dataset.

        At least one of ``label_dir`` or ``image_dir`` must be provided.

        Args:
            output_dir: Output directory.  Sampled files are placed here
                (flat layout for single mode, ``labels/`` + ``images/``
                subdirectories for both mode).
            count: Number of files to collect (>= 1).
            label_dir: Optional label directory (YOLO or LabelMe).
            image_dir: Optional image directory.
            shuffle: When True, randomly sample.  When False, take the
                first *count* files in sort order.
            seed: Random seed for reproducible shuffling.
            class_file: Optional classes.txt (copied to output dir).
            move: When True, move source files instead of copying.

        Returns:
            ``AnalysisResult`` with ``SampleResult`` in ``.data``.

        Output layout by mode:

        - **labels**: ``output_dir/*.txt`` (or ``*.json`` for LabelMe)
        - **images**: ``output_dir/*.jpg``
        - **both**: ``output_dir/labels/`` + ``output_dir/images/``
        """
        result = self._create_result()

        # ------------------------------------------------------------------
        # 1. Validate inputs
        # ------------------------------------------------------------------
        if label_dir is None and image_dir is None:
            result.add_error("At least one of label_dir or image_dir must be provided")
            return result

        if count < 1:
            result.add_error(f"Count must be at least 1, got: {count}")
            return result

        # ------------------------------------------------------------------
        # 2. Determine mode and collect items
        # ------------------------------------------------------------------
        fmt = ""
        items: List[Tuple[Path, str]]  # List of (path, stem) tuples
        image_stems: Dict[str, Path] = {}  # stem → source path (both mode)

        if label_dir is not None:
            # Detect format (YOLO or LabelMe only)
            try:
                fmt = detect_format(label_dir)
            except ValueError as e:
                result.add_error(str(e))
                return result

            if fmt == "coco":
                result.add_error(
                    "sample does not support COCO format. "
                    "COCO is a single JSON file — "
                    "use 'analyse filter' to select a subset of "
                    "annotations from a COCO dataset."
                )
                return result

            if image_dir is not None:
                mode = "both"
            else:
                mode = "labels"

            # Collect label files (file-level — no handler needed)
            ext = ".txt" if fmt == "yolo" else ".json"
            items = _collect_label_files(label_dir, ext)
        else:
            # Images-only mode
            mode = "images"
            fmt = ""
            items = _collect_image_files(image_dir)

        total = len(items)

        if total == 0:
            result.add_warning("Dataset is empty — nothing to sample")
            sample_result = SampleResult(
                sampled_count=0,
                total_count=0,
                output_dir=output_dir,
                count=count,
                shuffle=shuffle,
                seed=seed,
                mode=mode,
                format=fmt,
                move=move,
            )
            result.data = sample_result
            return result

        # ------------------------------------------------------------------
        # 3. Pre-index image files for both mode
        # ------------------------------------------------------------------
        if mode == "both":
            for p in sorted(image_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
                    stem = p.stem
                    if stem not in image_stems:
                        image_stems[stem] = p

        # ------------------------------------------------------------------
        # 4. Apply sampling strategy
        # ------------------------------------------------------------------
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(items)

        actual_count = min(count, total)
        if count > total:
            result.add_warning(
                f"Requested {count} files but only {total} available — collecting all"
            )

        sampled = items[:actual_count]

        # ------------------------------------------------------------------
        # 5. Create output directory
        # ------------------------------------------------------------------
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            result.add_error(f"Cannot create output directory: {e}")
            return result

        if mode == "both":
            label_subdir = output_dir / "labels"
            image_subdir = output_dir / "images"
            label_subdir.mkdir(parents=True, exist_ok=True)
            image_subdir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # 6. Copy/move files
        # ------------------------------------------------------------------
        unmatched_image_warnings = 0
        for src_path, stem in sampled:
            if mode == "both":
                # Copy label
                _copy_or_move_file(src_path, label_subdir, move, self.logger)
                # Match and copy image
                if stem in image_stems:
                    _copy_or_move_file(image_stems[stem], image_subdir, move, self.logger)
                else:
                    self._log_warning(
                        f"No matching image found for label '{stem}' in image directory"
                    )
                    unmatched_image_warnings += 1
            else:
                _copy_or_move_file(src_path, output_dir, move, self.logger)

        # Report unmatched images in both mode (copy only — move self-resolves)
        if mode == "both" and not move:
            sampled_stems = {stem for _, stem in sampled}
            for stem, img_path in image_stems.items():
                if stem not in sampled_stems:
                    self._log_warning(f"Image '{img_path.name}' has no matching label — skipped")

        # ------------------------------------------------------------------
        # 7. Copy class_file to output_dir if provided
        # ------------------------------------------------------------------
        if class_file is not None and class_file.exists():
            try:
                target_cf = output_dir / "classes.txt"
                if not target_cf.exists():
                    if move:
                        shutil.move(str(class_file), str(target_cf))
                    else:
                        shutil.copy2(str(class_file), str(target_cf))
            except OSError as e:
                result.add_warning(f"Could not copy class file: {e}")

        # ------------------------------------------------------------------
        # 8. Build result
        # ------------------------------------------------------------------
        sample_result = SampleResult(
            sampled_count=actual_count,
            total_count=total,
            output_dir=output_dir,
            count=count,
            shuffle=shuffle,
            seed=seed,
            mode=mode,
            format=fmt,
            move=move,
        )
        result.data = sample_result
        result.log_path = self._log_manager.log_path

        # ------------------------------------------------------------------
        # 9. Log output
        # ------------------------------------------------------------------
        mode_label = {
            "images": "Images Only",
            "labels": "Labels Only",
            "both": "Labels + Images",
        }[mode]
        source_path = label_dir if label_dir else image_dir
        self._log_info(
            format_analyse_header(
                f"Dataset Sampling ({mode_label})",
                source_path,
                f"{fmt} (auto-detected)" if fmt else "images only",
                class_file=class_file,
            )
        )
        self._log_info(
            format_sample_result(
                sampled_count=actual_count,
                total_count=total,
                requested_count=count,
                output_dir=output_dir,
                shuffle=shuffle,
                seed=seed,
                mode=mode,
                move=move,
            )
        )
        if result.log_path:
            self._log_info(format_analyse_result("✓ Success", result.log_path))

        return result
