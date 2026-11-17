"""
Utility functions for the label module.
"""

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


def resolve_coco_image_ids(image_ids: Iterable[str]) -> Dict[str, int]:
    """Resolve format-native ``image_id`` strings to unique positive-int COCO ids.

    COCO ``images[].id`` must be an int and unique per image
    (`spec_coco_format.md`).  ``ImageAnnotation.image_id`` is a string and may
    be a file stem, so COCO writers resolve it:

    - Digit-string ``image_id`` → its ``int`` value (preserves roundtrip ids)
    - Non-digit ``image_id`` → a dedicated counter id (not the annotation
      counter); counters skip values reserved by digit-string ids
    - Uniqueness guaranteed: if a resolved id collides (e.g. ``"01"`` vs
      ``"1"``), the later image gets a fresh counter id

    Canonical implementation — used by both
    ``CocoAnnotationHandler._prepare_coco_data()`` and
    ``evaluate/utils._dataset_to_coco_dict()``.

    Args:
        image_ids: Iterable of image_id strings in output order.

    Returns:
        Mapping ``{image_id_string: resolved_int_id}``.
    """
    ids = [str(s) for s in image_ids]
    # Values reserved by digit-string ids — counters must never take them,
    # otherwise a later digit id would collide with an earlier counter.
    digit_values: Set[int] = {int(s) for s in ids if s.isdigit()}
    used: Set[int] = set()
    mapping: Dict[str, int] = {}
    counter = 1

    for s in ids:
        if s.isdigit() and int(s) not in used:
            value = int(s)
        else:
            while counter in used or counter in digit_values:
                counter += 1
            value = counter
            counter += 1
        used.add(value)
        mapping[s] = value

    return mapping


def parse_yolo_class_id(token: str) -> Optional[int]:
    """Parse a YOLO class ID token, accepting integer-valued float strings.

    YOLO format specifies ``class_id`` as an integer, but some tooling
    outputs float-formatted values like ``5.000000``.  This parser
    handles both cases gracefully and is the **canonical implementation**
    for lenient parsing of class IDs from raw YOLO text.

    For strict validation (raising on non-integer floats), use
    ``YoloAnnotationHandler._parse_class_id()`` instead.

    Args:
        token: Raw string token from a YOLO annotation line.

    Returns:
        Integer class ID, or ``None`` if *token* is not a valid
        non-negative integer or integer-valued float.

    Examples:
        >>> parse_yolo_class_id("5")
        5
        >>> parse_yolo_class_id("5.000000")
        5
        >>> parse_yolo_class_id("0.0")
        0
        >>> parse_yolo_class_id("0.5")
        None
        >>> parse_yolo_class_id("-1")
        None
        >>> parse_yolo_class_id("abc")
        None
    """
    # Fast path: try direct int conversion first
    try:
        cid = int(token)
        if cid >= 0:
            return cid
        return None
    except ValueError:
        pass
    try:
        val = float(token)
        if val.is_integer() and val >= 0:
            return int(val)
    except (ValueError, OverflowError):
        pass
    return None


def merge_polygon_segments(
    segments: List[List[Tuple[float, float]]],
) -> List[Tuple[float, float]]:
    """Merge multiple closed polygon rings into a single outline.

    COCO allows several polygon segments per instance
    (``segmentation: [[...], [...]]``) whose semantics are the **union**
    of the segments, and RLE masks decode to multiple contours (disjoint
    components plus holes).  ``Segmentation.points`` is a single flat
    outline, so the reader reduces the rings to one merged outline
    (contract: ``spec_label.md`` §4.2):

    - Rings are chained via their **nearest point pair**: repeatedly merge
      the two rings whose closest points have the smallest distance.  The
      merged ring traverses ring A fully, bridges to the nearest point of
      ring B, traverses B fully, then bridges back along the **same edge**.
      A bridge traversed out-and-back cancels under both the even-odd and
      non-zero winding fill rules, so rasterizing the merged outline
      reproduces the union of the source rings (within boundary-pixel
      tolerance).
    - Consecutive duplicate points are removed after merging (they arise
      when two rings touch at the connection point).  No intersection
      surgery is performed — doubled-back bridges self-cancel at
      rasterization.
    - Single-ring input is returned **unchanged** (identity), keeping
      single-segment annotations bit-identical to the pre-merge behavior.

    Canonical implementation — used by ``CocoAnnotationHandler`` for both
    polygon parsing and RLE contour decoding.

    Args:
        segments: Closed polygon rings, each a list of ``(x, y)`` points.

    Returns:
        One merged ring as a flat list of ``(x, y)`` points (empty for
        empty input).
    """
    rings: List[List[Tuple[float, float]]] = [list(seg) for seg in segments if len(seg) >= 3]

    if not rings:
        return []
    if len(rings) == 1:
        return rings[0]

    while len(rings) > 1:
        # Greedy nearest merge: find the ring pair whose closest points
        # have the smallest distance (scan order breaks ties deterministically).
        best: Optional[float] = None
        best_i = best_j = 0
        best_p = best_q = 0
        for i, ring_a in enumerate(rings):
            for j in range(i + 1, len(rings)):
                ring_b = rings[j]
                for pi, p in enumerate(ring_a):
                    for qj, q in enumerate(ring_b):
                        dist = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                        if best is None or dist < best:
                            best = dist
                            best_i, best_j = i, j
                            best_p, best_q = pi, qj

        ring_a = rings[best_i]
        ring_b = rings[best_j]
        p = ring_a[best_p]
        q = ring_b[best_q]

        # Rotate A so p leads, traverse A fully and close it back to p,
        # bridge to q, rotate B so q leads, traverse B fully and close it
        # back to q.  The polygon's final edge (q → first point) doubles
        # back over the p → q bridge, cancelling it at rasterization.
        merged = ring_a[best_p:] + ring_a[:best_p] + [p]
        merged += ring_b[best_q:] + ring_b[:best_q] + [q]

        rings.pop(best_j)  # remove B first (larger index)
        rings[best_i] = merged

    # Remove consecutive duplicate points (they arise when two rings touch
    # at the connection point, i.e. p == q).
    merged = rings[0]
    deduped: List[Tuple[float, float]] = [merged[0]]
    for pt in merged[1:]:
        if pt != deduped[-1]:
            deduped.append(pt)
    return deduped


def calculate_file_hash(file_path: Path, algorithm: str = "md5") -> str:
    """Calculate hash of a file."""
    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


# Category management utilities


def extract_categories_from_coco_data(coco_data: dict) -> Dict[int, str]:
    """
    Extract category mapping from COCO JSON data.

    Args:
        coco_data: COCO JSON data dictionary

    Returns:
        Dictionary mapping category IDs to names
    """
    categories = {}
    for cat in coco_data.get("categories", []):
        cat_id = cat.get("id")
        cat_name = cat.get("name", "")
        if cat_id is not None:
            categories[cat_id] = cat_name
    return categories


def generate_classes_file(categories: Dict[int, str], output_path: Path) -> bool:
    """
    Generate classes.txt file from category mapping.

    Args:
        categories: Dictionary mapping category IDs to names
        output_path: Path to output classes.txt file

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for cat_id in sorted(categories.keys()):
                f.write(f"{categories[cat_id]}\n")
        return True
    except Exception as e:
        import logging

        logging.getLogger(__name__).error(f"Failed to write classes file {output_path}: {e}")
        return False


def load_classes_file(class_file: Path) -> Dict[int, str]:
    """
    Load category mapping from classes.txt file.

    Args:
        class_file: Path to classes.txt file

    Returns:
        Dictionary mapping index (starting from 0) to category name
    """
    categories = {}
    try:
        with open(class_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    categories[i] = line
    except Exception as e:
        import logging

        logging.getLogger(__name__).error(f"Failed to load classes file {class_file}: {e}")
    return categories
