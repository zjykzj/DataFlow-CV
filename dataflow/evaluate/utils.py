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
            "pycocotools is required for evaluation. Install with: pip install pycocotools"
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
            return coco_gt.loadRes(dt_data)
        if isinstance(dt_data, dict):
            _validate_coco_dict(dt_data)
            return _create_coco_from_dict(dt_data)
        raise ValueError(f"DT file must contain a JSON dict or list, got: {type(dt_data).__name__}")

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
        raise ValueError(f"COCO dict missing required keys: {missing}")


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
        raise ValueError(f"Dataset format must be COCO, got: {dataset.format}")

    # Preserve top-level keys from dataset_info
    info = dataset.dataset_info.copy()

    # Reconstruct images — image ids resolve with the same rule as the COCO
    # handler write() (digit image_id → int, non-digit/colliding → dedicated
    # counter, unique ids guaranteed — resolve_coco_image_ids()).
    from dataflow.label.utils import resolve_coco_image_ids

    id_map = resolve_coco_image_ids(img.image_id for img in dataset.images)

    images = []
    for img in dataset.images:
        img_entry = {
            "id": id_map[img.image_id],
            "file_name": img.image_path,
            "width": img.width,
            "height": img.height,
        }
        images.append(img_entry)

    # Reconstruct categories
    categories = [
        {"id": int(cat_id), "name": cat_name} for cat_id, cat_name in dataset.categories.items()
    ]

    # Reconstruct annotations
    annotations = []
    ann_id = 1
    for img in dataset.images:
        image_id = id_map[img.image_id]
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
            elif obj.segmentation is not None and obj.segmentation.points:
                # Seg-only objects: derive bbox/area from the polygon
                # extent — pycocotools COCOeval requires area on every
                # annotation (mirrors CocoAnnotationHandler's write path,
                # spec_coco_format.md §4).
                points = obj.segmentation.points
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                min_x, min_y = min(xs), min(ys)
                w = max(xs) - min_x
                h = max(ys) - min_y
                ann_entry["bbox"] = [min_x, min_y, w, h]
                ann_entry["area"] = w * h

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


def _extract_stats(coco_gt: "COCO", coco_dt: "COCO") -> Tuple[Dict[str, int], Dict[str, int]]:
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
    coco_dt: "COCO",
) -> None:
    """Verify that all DT annotations have a ``score`` field.

    Raises ValueError if any annotation is missing the score field.
    pycocotools COCOeval requires scores for sorting — evaluation
    cannot produce correct results without them.  Returns ``None``
    (implicitly) when all annotations have valid scores.
    """
    ann_ids = coco_dt.getAnnIds()
    missing = []
    for ann_id in ann_ids:
        anns = coco_dt.loadAnns(ann_id)
        for ann in anns:
            if "score" not in ann:
                missing.append(ann["id"])

    if missing:
        raise ValueError(
            f"{len(missing)} DT annotation(s) missing 'score' field"
            f": IDs={missing[:10]}{'...' if len(missing) > 10 else ''}"
        )


def _validate_common_categories(coco_gt: "COCO", coco_dt: "COCO") -> List[str]:
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
        names = [cat["name"] for cat in coco_gt.loadCats(sorted(no_dt))]
        warnings.append(f"{len(no_dt)} GT categories have no detections: {names}")

    return warnings


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
# Moved to dataflow/evaluate/log_templates.py — re-export for backward compat.
from dataflow.evaluate.log_templates import (  # noqa: E402, F401
    format_metric_table,
    format_per_class_table,
    format_prf1_output,
)


def _validate_segm_data(coco_gt: Any, coco_dt: Any) -> Tuple[List[str], List[str]]:
    """Check segmentation data presence for mask IoU evaluation.

    Returns:
        Tuple of ``(errors, warnings)``.
        *errors* are hard failures (complete absence of segmentation
        data — evaluation cannot proceed).  *warnings* are advisory
        (mixed dataset — some annotations lack segmentation, they
        will be silently excluded by pycocotools).
    """
    errors: List[str] = []
    warnings: List[str] = []

    gt_total = 0
    gt_with_segm = 0
    for ann in coco_gt.loadAnns(coco_gt.getAnnIds()):
        gt_total += 1
        if ann.get("segmentation") and ann["segmentation"]:
            gt_with_segm += 1

    dt_total = 0
    dt_with_segm = 0
    for ann in coco_dt.loadAnns(coco_dt.getAnnIds()):
        dt_total += 1
        if ann.get("segmentation") and ann["segmentation"]:
            dt_with_segm += 1

    gt_has_segm = gt_with_segm > 0
    dt_has_segm = dt_with_segm > 0

    if not gt_has_segm and not dt_has_segm:
        errors.append(
            "Segmentation data missing — cannot evaluate with "
            "iouType='segm'. Ensure both GT and DT contain "
            "segmentation annotations."
        )
    elif not gt_has_segm:
        errors.append(
            "GT contains no segmentation data — cannot evaluate "
            "with iouType='segm'. Ensure GT contains segmentation "
            "annotations."
        )
    elif not dt_has_segm:
        errors.append(
            "DT contains no segmentation data — cannot evaluate "
            "with iouType='segm'. Ensure DT contains segmentation "
            "annotations."
        )

    # Mixed-dataset detection: some annotations have segmentation,
    # others don't — pycocotools silently excludes the ones without
    # segmentation during mask IoU computation.
    if gt_has_segm and gt_with_segm < gt_total:
        missing = gt_total - gt_with_segm
        warnings.append(
            f"GT is a mixed dataset: {missing}/{gt_total} "
            f"annotations lack segmentation and will be excluded "
            f"from mask IoU evaluation."
        )
    if dt_has_segm and dt_with_segm < dt_total:
        missing = dt_total - dt_with_segm
        warnings.append(
            f"DT is a mixed dataset: {missing}/{dt_total} "
            f"annotations lack segmentation and will be excluded "
            f"from mask IoU evaluation."
        )

    return errors, warnings


# ---------------------------------------------------------------------------
# Keypoint (pose) evaluation helpers
# ---------------------------------------------------------------------------

# COCO official per-keypoint OKS sigmas for the 17-keypoint person skeleton
# (pycocotools default `kpt_oks_sigmas`; contract: spec_evaluate.md §3.4).
COCO_KPT_SIGMAS_17 = [
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
]


def _validate_kpt_data(coco_gt: Any, coco_dt: Any) -> Tuple[List[str], List[str]]:
    """Check keypoint data presence for OKS evaluation (spec_evaluate.md §5.3 rule 8).

    Mirrors ``_validate_segm_data``: *errors* are hard failures (complete
    absence of keypoint data — OKS on bbox-only data is meaningless);
    *warnings* are advisory (mixed datasets — annotations without
    keypoints are excluded by pycocotools).
    """
    errors: List[str] = []
    warnings: List[str] = []

    gt_total = 0
    gt_with_kpt = 0
    for ann in coco_gt.loadAnns(coco_gt.getAnnIds()):
        gt_total += 1
        if ann.get("keypoints"):
            gt_with_kpt += 1

    dt_total = 0
    dt_with_kpt = 0
    for ann in coco_dt.loadAnns(coco_dt.getAnnIds()):
        dt_total += 1
        if ann.get("keypoints"):
            dt_with_kpt += 1

    if gt_total > 0 and gt_with_kpt == 0:
        errors.append(
            "GT contains no keypoint data — cannot evaluate with iouType='keypoints'. "
            "Ensure GT annotations contain keypoints."
        )
    if dt_total > 0 and dt_with_kpt == 0:
        errors.append(
            "DT contains no keypoint data — cannot evaluate with iouType='keypoints'. "
            "Ensure DT annotations contain keypoints."
        )

    if gt_total > 0 and 0 < gt_with_kpt < gt_total:
        warnings.append(
            f"GT is a mixed dataset: {gt_total - gt_with_kpt}/{gt_total} "
            "annotations lack keypoints and will be excluded from OKS evaluation."
        )
    if dt_total > 0 and 0 < dt_with_kpt < dt_total:
        warnings.append(
            f"DT is a mixed dataset: {dt_total - dt_with_kpt}/{dt_total} "
            "annotations lack keypoints and will be excluded from OKS evaluation."
        )

    return errors, warnings


def load_kpt_sigmas(config_path: Union[str, Path]) -> Optional[List[float]]:
    """Load per-keypoint OKS sigmas from a keypoint-definition YAML.

    Wraps ``dataflow.label.utils.load_kpts_config`` so callers above the
    Label layer (e.g. the CLI, spec_cli.md §8 forbids CLI imports from
    ``dataflow.label``) can obtain sigmas through the Evaluate module.
    """
    from dataflow.label.utils import load_kpts_config

    return load_kpts_config(config_path).sigmas


def resolve_kpt_sigmas(k: int, sigmas: Union[List[float], None]) -> Tuple[Any, Union[str, None]]:
    """Resolve per-keypoint OKS sigmas (contract: spec_evaluate.md §3.4).

    Priority: explicit ``sigmas`` (length must equal k) → COCO official
    17-keypoint sigmas when k == 17 → uniform 0.05 + warning otherwise.

    Args:
        k: Dataset keypoint count.
        sigmas: Explicit per-keypoint sigmas or None.

    Returns:
        ``(sigmas_array, warning_msg_or_None)``.

    Raises:
        ValueError: If explicit sigmas length mismatches k.
    """
    if sigmas is not None:
        if len(sigmas) != k:
            raise ValueError(f"sigmas must have {k} values (one per keypoint), got {len(sigmas)}")
        import numpy as np

        return np.array(sigmas, dtype=float), None

    if k == 17:
        import numpy as np

        return np.array(COCO_KPT_SIGMAS_17), None

    import numpy as np

    return np.full(k, 0.05), (
        f"Uniform OKS sigmas (0.05) assumed for {k} keypoints — "
        "pass explicit sigmas or a 17-keypoint dataset for COCO-standard values"
    )
