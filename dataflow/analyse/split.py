"""
Train/test dataset splitting.

``SplitAnalyser`` splits a dataset into training and validation subsets
with deterministic shuffling.  Supports three modes (labels-only,
images-only, or both) for YOLO and LabelMe formats.

For labels-only mode, the operation is a pure file-level split —
annotation content is never parsed, making it fast and dependency-free.
"""

import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import AnalysisResult, BaseAnalyser, SplitResult
from .log_templates import (
    format_analyse_header,
    format_analyse_result,
    format_split_result,
)
from .utils import (
    _IMAGE_EXTENSIONS,
    _collect_image_files,
    _collect_label_files,
    _copy_or_move_file,
    detect_format,
)


class SplitAnalyser(BaseAnalyser):
    """Split dataset into train/val subsets.

    Constructor: ``SplitAnalyser(log_config=None)``

    Supports three modes (auto-detected from provided inputs):

    - **Labels-only**: ``-l/--label-dir`` only — split label files.
    - **Images-only**: ``-i/--image-dir`` only — split image files.
    - **Both**: labels drive split; images matched by stem.

    Only YOLO and LabelMe formats are supported (not COCO).

    Example::

        from dataflow.analyse import SplitAnalyser

        analyser = SplitAnalyser()
        result = analyser.analyse(
            output_dir=Path("split_output/"),
            ratio=0.8,
            seed=42,
            label_dir=Path("yolo_labels/"),
            image_dir=Path("images/"),
            class_file=Path("classes.txt"),
        )
        if result.success:
            split = result.data
            print(f"Train: {split.train_count}, Val: {split.val_count}")
    """

    def analyse(
        self,
        output_dir: Path,
        ratio: float = 0.8,
        seed: int = 42,
        label_dir: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        class_file: Optional[Path] = None,
        move: bool = False,
    ) -> AnalysisResult:
        """Split the dataset into train and val subsets.

        At least one of ``label_dir`` or ``image_dir`` must be provided.

        Args:
            output_dir: Output root directory.  Train and val outputs
                are placed here (see mode-specific layouts below).
            ratio: Proportion of data for training (default 0.8).
            seed: Random seed for reproducible shuffling.
            label_dir: Optional label directory (YOLO or LabelMe).
            image_dir: Optional image directory.
            class_file: Optional classes.txt (copied to output dirs).
            move: When True, move source files instead of copying.

        Returns:
            ``AnalysisResult`` with ``SplitResult`` in ``.data``.

        Output layout by mode:

        - **labels**: ``output_dir/train/*.txt``, ``output_dir/val/*.txt``
          (or ``*.json`` for LabelMe)
        - **images**: ``output_dir/train/*.jpg``, ``output_dir/val/*.jpg``
        - **both**: ``output_dir/train/labels/`` + ``output_dir/train/images/``,
          same for ``val/``
        """
        result = self._create_result()

        # ------------------------------------------------------------------
        # 1. Validate inputs
        # ------------------------------------------------------------------
        if label_dir is None and image_dir is None:
            result.add_error("At least one of label_dir or image_dir must be provided")
            return result

        if not 0.0 < ratio < 1.0:
            result.add_error(f"Ratio must be between 0 and 1 (exclusive), got: {ratio}")
            return result

        # ------------------------------------------------------------------
        # 2. Determine mode and collect items
        # ------------------------------------------------------------------
        fmt = ""
        items: list  # List of (Path, stem) tuples for all modes
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
                    "split does not support COCO format. "
                    "COCO is a single JSON file — "
                    "use 'analyse partition' for N-way split or "
                    "convert to YOLO/LabelMe first."
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
            result.add_warning("Dataset is empty — nothing to split")
            train_dir = output_dir / "train"
            val_dir = output_dir / "val"
            split_result = SplitResult(
                train_count=0,
                val_count=0,
                train_dir=train_dir,
                val_dir=val_dir,
                ratio=ratio,
                seed=seed,
                format=fmt,
                mode=mode,
                move=move,
            )
            result.data = split_result
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
        # 4. Shuffle and split
        # ------------------------------------------------------------------
        rng = random.Random(seed)
        rng.shuffle(items)

        split_idx = int(len(items) * ratio)
        # Ensure at least 1 item in each split when dataset has ≥2 items
        if split_idx == 0 and len(items) >= 2:
            split_idx = 1
        elif split_idx == len(items) and len(items) >= 2:
            split_idx = len(items) - 1

        train_items = items[:split_idx]
        val_items = items[split_idx:]

        # ------------------------------------------------------------------
        # 5. Create output directories
        # ------------------------------------------------------------------
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            result.add_error(f"Cannot create output directory: {e}")
            return result

        if mode == "both":
            train_label_dir = output_dir / "train" / "labels"
            train_image_dir = output_dir / "train" / "images"
            val_label_dir = output_dir / "val" / "labels"
            val_image_dir = output_dir / "val" / "images"
            train_label_dir.mkdir(parents=True, exist_ok=True)
            train_image_dir.mkdir(parents=True, exist_ok=True)
            val_label_dir.mkdir(parents=True, exist_ok=True)
            val_image_dir.mkdir(parents=True, exist_ok=True)
        else:
            train_dir = output_dir / "train"
            val_dir = output_dir / "val"
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # 6. Copy/move files
        # ------------------------------------------------------------------
        _split_files(
            train_items,
            val_items,
            output_dir,
            mode,
            image_stems,
            move,
            self.logger,
        )

        # Report unmatched images in both mode (copy only — move self-resolves)
        if mode == "both" and not move:
            label_stems = {stem for _, stem in train_items + val_items}
            for stem, img_path in image_stems.items():
                if stem not in label_stems:
                    self._log_warning(f"Image '{img_path.name}' has no matching label — skipped")

        # ------------------------------------------------------------------
        # 7. Copy class_file to both output dirs if provided
        # ------------------------------------------------------------------
        _copy_class_file(class_file, output_dir, mode, move, self.logger)

        # ------------------------------------------------------------------
        # 8. Build result
        # ------------------------------------------------------------------
        if mode == "both":
            train_path = output_dir / "train"
            val_path = output_dir / "val"
        else:
            train_path = output_dir / "train"
            val_path = output_dir / "val"

        split_result = SplitResult(
            train_count=len(train_items),
            val_count=len(val_items),
            train_dir=train_path,
            val_dir=val_path,
            ratio=ratio,
            seed=seed,
            format=fmt,
            mode=mode,
            move=move,
        )
        result.data = split_result
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
                f"Train/Test Split ({mode_label})",
                source_path,
                f"{fmt} (auto-detected)" if fmt else "images only",
                class_file=class_file,
            )
        )
        self._log_info(
            format_split_result(
                train_count=len(train_items),
                val_count=len(val_items),
                train_dir=train_path,
                val_dir=val_path,
                ratio=ratio,
                seed=seed,
                mode=mode,
                move=move,
            )
        )
        if result.log_path:
            self._log_info(format_analyse_result("✓ Success", result.log_path))

        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_files(
    train_items: List[Tuple[Path, str]],
    val_items: List[Tuple[Path, str]],
    output_dir: Path,
    mode: str,
    image_stems: Dict[str, Path],
    move: bool,
    logger,
) -> None:
    """Copy or move split items to train/val output directories.

    Args:
        train_items: ``(source_path, stem)`` tuples for train set.
        val_items: ``(source_path, stem)`` tuples for val set.
        output_dir: Root output directory.
        mode: ``"labels"`` | ``"images"`` | ``"both"``.
        image_stems: ``{stem: source_path}`` mapping (both mode only).
        move: If True, move; else copy.
        logger: Logger for warnings.
    """
    if mode == "both":
        train_label_dir = output_dir / "train" / "labels"
        train_image_dir = output_dir / "train" / "images"
        val_label_dir = output_dir / "val" / "labels"
        val_image_dir = output_dir / "val" / "images"

        for label_path, stem in train_items:
            _copy_or_move_file(label_path, train_label_dir, move, logger)
            if stem in image_stems:
                _copy_or_move_file(image_stems[stem], train_image_dir, move, logger)
            else:
                logger.warning(f"No matching image found for label '{stem}' in image directory")

        for label_path, stem in val_items:
            _copy_or_move_file(label_path, val_label_dir, move, logger)
            if stem in image_stems:
                _copy_or_move_file(image_stems[stem], val_image_dir, move, logger)
            else:
                logger.warning(f"No matching image found for label '{stem}' in image directory")
    else:
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"

        for src_path, _stem in train_items:
            _copy_or_move_file(src_path, train_dir, move, logger)

        for src_path, _stem in val_items:
            _copy_or_move_file(src_path, val_dir, move, logger)


def _copy_class_file(
    class_file: Optional[Path],
    output_dir: Path,
    mode: str,
    move: bool,
    logger,
) -> None:
    """Copy classes.txt to train/val output directories.

    Args:
        class_file: Path to classes.txt (None → skip).
        output_dir: Root output directory.
        mode: ``"labels"`` | ``"images"`` | ``"both"``.
        move: If True, move; else copy.
        logger: Logger for warnings.
    """
    if class_file is None or not class_file.exists():
        return

    targets: List[Path] = []
    if mode == "both":
        targets = [
            output_dir / "train" / class_file.name,
            output_dir / "val" / class_file.name,
        ]
    else:
        targets = [
            output_dir / "train" / class_file.name,
            output_dir / "val" / class_file.name,
        ]

    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                if move:
                    shutil.move(str(class_file), str(target))
                    # Only move once — subsequent iterations just skip
                    break
                else:
                    shutil.copy2(str(class_file), str(target))
        except OSError as e:
            logger.warning(f"Could not copy class file to {target}: {e}")
