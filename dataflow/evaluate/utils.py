"""
Evaluation utilities for DataFlow-CV.

Provides input normalization, COCO object construction, validation,
and formatting helpers used by the Evaluate module.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# pycocotools guard
# ---------------------------------------------------------------------------
try:
    from pycocotools.coco import COCO

    HAS_COCO = True
except ImportError:
    HAS_COCO = False
    COCO = None  # type: ignore


def _validate_coco_available() -> None:
    """Raise ImportError with a clear message if pycocotools is not installed."""
    if not HAS_COCO:
        raise ImportError(
            "pycocotools is required for evaluation. "
            "Install with: pip install pycocotools"
        )


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

def _load_coco(source: Union[str, Path, Dict, Any]) -> "COCO":
    """Normalize any input type to a pycocotools.COCO instance.

    Args:
        source: One of:
            - str/Path: File path to a COCO JSON file.
            - Dict: COCO dict with ``images``, ``annotations``, ``categories``.
            - DatasetAnnotations: A Label-module container with format=COCO.

    Returns:
        pycocotools.COCO instance.

    Raises:
        ImportError: If pycocotools is not installed.
        ValueError: If the source type is unsupported or the COCO dict is
            structurally invalid.
    """
    _validate_coco_available()

    # --- Path / str ---
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"COCO JSON file not found: {path}")
        return COCO(str(path))

    # --- Dict ---
    if isinstance(source, dict):
        _validate_coco_dict(source)
        # pycocotools.COCO() needs a file path, not a dict.
        # Use a temporary JSON round-trip to create the COCO object.
        return _create_coco_from_dict(source)

    # --- DatasetAnnotations ---
    # Try to detect DatasetAnnotations without importing it at module level
    if hasattr(source, "format") and hasattr(source, "images"):
        return _load_coco_from_dataset(source)

    raise ValueError(
        f"Unsupported source type: {type(source).__name__}. "
        "Expected str, Path, dict, or DatasetAnnotations."
    )


def _load_dt(
    dt_source: Union[str, Path, Dict, List, Any],
    coco_gt: "COCO",
) -> "COCO":
    """Load DT (detection/prediction) data, handling list-format JSON.

    Unlike :func:`_load_coco` which is used for GT, this function handles
    two DT-specific formats:

    * **Plain JSON list** — a top-level array of annotation dicts. Loaded
      via :meth:`coco_gt.loadRes`, which copies ``images`` and ``categories``
      from GT and indexes the annotation list.
    * **Full COCO dict** — same structure as GT plus ``score`` in each
      annotation. Loaded via the same path as :func:`_load_coco`.

    Args:
        dt_source: One of:
            - str/Path: File path to a COCO JSON dict or plain annotation list.
            - List: In-memory list of annotation dicts (with ``bbox``, ``score``).
            - Dict: In-memory COCO dict.
            - DatasetAnnotations: A Label-module container with format=COCO.
        coco_gt: The already-loaded GT :class:`pycocotools.COCO` instance.
            Required for list-format DT so that ``loadRes`` can copy images
            and categories.

    Returns:
        pycocotools.COCO instance.

    Raises:
        ImportError: If pycocotools is not installed.
        FileNotFoundError: If ``dt_source`` is a path that does not exist.
        ValueError: If the source type is unsupported or the file content
            is neither a list nor a dict.
    """
    _validate_coco_available()

    # --- List in memory ---
    if isinstance(dt_source, list):
        return coco_gt.loadRes(dt_source)

    # --- File path ---
    if isinstance(dt_source, (str, Path)):
        dt_path = Path(dt_source)
        if not dt_path.exists():
            raise FileNotFoundError(f"DT file not found: {dt_path}")
        with open(dt_path, "r", encoding="utf-8") as f:
            dt_data = json.load(f)
        if isinstance(dt_data, list):
            return coco_gt.loadRes(str(dt_path))
        if isinstance(dt_data, dict):
            _validate_coco_dict(dt_data)
            return _create_coco_from_dict(dt_data)
        raise ValueError(
            f"DT file must contain a JSON dict or list, got: {type(dt_data).__name__}"
        )

    # --- Dict ---
    if isinstance(dt_source, dict):
        _validate_coco_dict(dt_source)
        return _create_coco_from_dict(dt_source)

    # --- DatasetAnnotations ---
    if hasattr(dt_source, "format") and hasattr(dt_source, "images"):
        return _load_coco_from_dataset(dt_source)

    raise ValueError(
        f"Unsupported DT source type: {type(dt_source).__name__}. "
        "Expected str, Path, dict, list, or DatasetAnnotations."
    )


def _validate_coco_dict(data: Dict[str, Any]) -> None:
    """Validate that a dict has the required COCO top-level keys.

    Raises ValueError if required keys are missing.
    """
    required = ["images", "annotations", "categories"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"COCO dict missing required keys: {missing}"
        )


def _create_coco_from_dict(data: Dict[str, Any]) -> "COCO":
    """Create a pycocotools.COCO instance from an in-memory dict.

    pycocotools.COCO() constructor requires a file path.  We work around
    this by writing a temporary JSON string and using the ``loadRes``-style
    approach: create an empty COCO and then manually populate the datasets
    and create the index.
    """
    import tempfile
    import os

    # Write to temp file so pycocotools can load it
    tmpfd, tmppath = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(tmpfd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return COCO(tmppath)
    finally:
        os.unlink(tmppath)


def _load_coco_from_dataset(dataset: Any) -> "COCO":
    """Convert a DatasetAnnotations (format=COCO) to a pycocotools.COCO.

    Reconstructs the COCO JSON dict from DatasetAnnotations fields and
    creates a COCO instance from it.
    """
    coco_dict = _dataset_to_coco_dict(dataset)
    return _create_coco_from_dict(coco_dict)


def _dataset_to_coco_dict(dataset: Any) -> Dict[str, Any]:
    """Convert DatasetAnnotations (format=COCO) to a COCO JSON dict.

    Uses dataset_info for preserved raw COCO data (licenses, info, etc.)
    and reconstructs images/annotations/categories from the data model.
    """
    from dataflow.label.models import AnnotationFormat

    if getattr(dataset, "format", None) != AnnotationFormat.COCO:
        raise ValueError(
            f"Dataset format must be COCO, got: {dataset.format}"
        )

    # Preserve top-level keys from dataset_info
    info = dataset.dataset_info.copy()

    # Reconstruct images
    images = []
    for img in dataset.images:
        img_entry = {
            "id": int(img.image_id) if img.image_id.isdigit() else img.image_id,
            "file_name": img.image_path,
            "width": img.width,
            "height": img.height,
        }
        images.append(img_entry)

    # Reconstruct categories
    categories = [
        {"id": int(cat_id), "name": cat_name}
        for cat_id, cat_name in dataset.categories.items()
    ]

    # Reconstruct annotations
    annotations = []
    ann_id = 1
    for img in dataset.images:
        image_id = (
            int(img.image_id)
            if str(img.image_id).isdigit()
            else img.image_id
        )
        for obj in img.objects:
            ann_entry: Dict[str, Any] = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": obj.class_id,
                "iscrowd": 1 if obj.is_crowd else 0,
            }

            if obj.bbox is not None:
                ann_entry["bbox"] = [
                    obj.bbox.x,
                    obj.bbox.y,
                    obj.bbox.width,
                    obj.bbox.height,
                ]
                ann_entry["area"] = obj.bbox.width * obj.bbox.height

            if obj.segmentation is not None:
                if obj.segmentation.has_rle():
                    ann_entry["segmentation"] = obj.segmentation.rle
                else:
                    # Polygon: flatten [(x,y), ...] → [x1, y1, x2, y2, ...]
                    flat = []
                    for pt in obj.segmentation.points:
                        flat.extend([pt[0], pt[1]])
                    ann_entry["segmentation"] = [flat]

            # Preserve confidence as score (for DT)
            if hasattr(obj, "confidence") and obj.confidence is not None:
                ann_entry["score"] = obj.confidence

            annotations.append(ann_entry)
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        **{k: v for k, v in info.items() if k not in ("images", "annotations", "categories")},
    }


# ---------------------------------------------------------------------------
# Statistics and validation
# ---------------------------------------------------------------------------

def _extract_stats(
    coco_gt: "COCO", coco_dt: "COCO"
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Extract summary statistics from GT and DT COCO objects.

    Returns:
        Tuple of (gt_stats, dt_stats), each with keys:
        ``images``, ``annotations``, ``categories``.
    """
    gt_cat_ids = coco_gt.getCatIds()
    dt_cat_ids = coco_dt.getCatIds()

    gt_ann_ids = coco_gt.getAnnIds()
    dt_ann_ids = coco_dt.getAnnIds()

    return (
        {
            "images": len(coco_gt.getImgIds()),
            "annotations": len(gt_ann_ids),
            "categories": len(gt_cat_ids),
        },
        {
            "images": len(coco_dt.getImgIds()),
            "annotations": len(dt_ann_ids),
            "categories": len(dt_cat_ids),
        },
    )


def _validate_dt_scores(
    coco_dt: "COCO", strict_mode: bool = True
) -> Tuple[bool, List[str]]:
    """Verify that all DT annotations have a ``score`` field.

    Returns:
        Tuple of (valid, warnings).  If ``strict_mode`` and scores are
        missing, raises ValueError.
    """
    ann_ids = coco_dt.getAnnIds()
    missing = []
    for ann_id in ann_ids:
        anns = coco_dt.loadAnns(ann_id)
        for ann in anns:
            if "score" not in ann:
                missing.append(ann["id"])

    if missing:
        msg = (
            f"{len(missing)} DT annotation(s) missing 'score' field"
        )
        if strict_mode:
            raise ValueError(
                f"{msg}: IDs={missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        return False, [msg]

    return True, []


def _validate_common_categories(
    coco_gt: "COCO", coco_dt: "COCO"
) -> List[str]:
    """Check for DT categories unknown to GT and warn.

    Returns a list of warning messages.
    """
    warnings = []
    gt_cat_ids = set(coco_gt.getCatIds())
    dt_cat_ids = set(coco_dt.getCatIds())

    unknown = dt_cat_ids - gt_cat_ids
    if unknown:
        warnings.append(
            f"DT contains {len(unknown)} category ID(s) not in GT: "
            f"{sorted(unknown)}. These detections will be ignored."
        )

    # Check for GT categories with no DT
    no_dt = gt_cat_ids - dt_cat_ids
    if no_dt:
        names = [
            cat["name"]
            for cat in coco_gt.loadCats(sorted(no_dt))
        ]
        warnings.append(
            f"{len(no_dt)} GT categories have no detections: {names}"
        )

    return warnings


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_metric_table(metrics: Any) -> str:
    """Format the 12 COCO standard metrics as a readable table.

    Args:
        metrics: An EvaluationMetrics instance.

    Returns:
        Formatted string suitable for console output.
    """
    lines = [
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}".format(metrics.ap),
        " Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:.3f}".format(metrics.ap50),
        " Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {:.3f}".format(metrics.ap75),
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}".format(metrics.ap_small),
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}".format(metrics.ap_medium),
        " Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}".format(metrics.ap_large),
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {:.3f}".format(metrics.ar_max_1),
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {:.3f}".format(metrics.ar_max_10),
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}".format(metrics.ar_max_100),
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}".format(metrics.ar_small),
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}".format(metrics.ar_medium),
        " Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}".format(metrics.ar_large),
    ]
    return "\n".join(lines)


def format_per_class_table(
    per_class: Dict[int, Any],
) -> str:
    """Format per-class metrics as a readable table.

    Args:
        per_class: Mapping of class_id → PerClassMetrics.

    Returns:
        Formatted string suitable for console output.
    """
    if not per_class:
        return ""

    header = (
        f"{'Class':<14s} {'GT':>5s} {'DT':>5s} {'TP':>5s} {'FP':>5s} {'FN':>5s} "
        f"{'AP':>7s} {'AP50':>7s} {'AP75':>7s} {'P':>7s} {'R':>7s} {'F1':>7s}"
    )
    sep = "─" * len(header)
    lines = [
        "Per-Class Breakdown (IoU: 0.50:0.95):",
        sep,
        header,
        sep,
    ]

    for class_id in sorted(per_class.keys()):
        m = per_class[class_id]
        lines.append(
            f"{m.class_name:<14s} "
            f"{m.gt_count:>5d} {m.dt_count:>5d} "
            f"{m.tp:>5d} {m.fp:>5d} {m.fn:>5d} "
            f"{m.ap:>7.3f} {m.ap50:>7.3f} {m.ap75:>7.3f} "
            f"{m.precision:>7.3f} {m.recall:>7.3f} {m.f1_score:>7.3f}"
        )

    lines.append(sep)
    return "\n".join(lines)


def format_prf1_output(
    result: Any,
) -> str:
    """Format PRF1Result as a readable summary line.

    Args:
        result: A PRF1Result instance.

    Returns:
        Formatted string.
    """
    if not result.success or result.overall is None:
        return "P/R/F1 computation failed."

    o = result.overall
    return (
        f"Precision / Recall / F1-Score (IoU={result.iou_threshold:.2f}, "
        f"Conf={result.confidence_threshold:.2f}):\n"
        f"  Overall:  P={o.precision:.3f}  R={o.recall:.3f}  "
        f"F1={o.f1_score:.3f}  TP={o.tp}  FP={o.fp}  FN={o.fn}"
    )
