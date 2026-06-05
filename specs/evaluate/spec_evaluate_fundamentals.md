# Evaluation Fundamentals Specification

> **Version:** 1.0
> **Status:** Draft
> **Layer:** Evaluate
> **Dependencies:** None (foundation document)

## 1. Document Scope

This document defines the foundational concepts upon which all evaluation metrics are built. Every metric in `spec_evaluate_metrics.md` and every task rule in `spec_evaluate_tasks.md` derives from the definitions in this document.

## 2. IoU — Intersection over Union

IoU is the fundamental matching criterion between a predicted region and a ground truth region. Two variants exist.

### 2.1 Bounding Box IoU

For two axis-aligned rectangles `A` and `B`, each defined as `[x1, y1, x2, y2]` (top-left, bottom-right in absolute pixels):

```
intersection_width  = max(0, min(A.x2, B.x2) - max(A.x1, B.x1))
intersection_height = max(0, min(A.y2, B.y2) - max(A.y1, B.y1))
intersection_area   = intersection_width × intersection_height

union_area = area(A) + area(B) - intersection_area

IoU = intersection_area / union_area   (0 if union_area = 0)
```

**Properties:**
- Range: `[0, 1]`
- `IoU = 1` → perfect overlap
- `IoU = 0` → no overlap
- Symmetric: `IoU(A, B) = IoU(B, A)`

### 2.2 Mask IoU

For two binary masks `M_A` and `M_B` of identical dimensions `H × W`:

```
intersection = sum(M_A[i,j] ∧ M_B[i,j])    # pixel-wise AND
union        = sum(M_A[i,j] ∨ M_B[i,j])    # pixel-wise OR

IoU = intersection / union   (0 if union = 0)
```

This is equivalent to the pixel-level Jaccard index. Unlike bbox IoU which is a geometric formula, mask IoU requires binary mask rasterization and is computationally more expensive.

### 2.3 bbox IoU vs mask IoU — Key Differences

| Property | Bbox IoU | Mask IoU |
|----------|----------|----------|
| Input | Two axis-aligned rectangles | Two binary masks (H × W) |
| Computation | O(1) geometric formula | O(H×W) pixel scan |
| Precision | Exact (floating point) | Bounded by mask resolution |
| RLE requirement | No | Yes (pycocotools encodes/decodes RLE internally) |
| Typical use | Object detection evaluation | Instance segmentation evaluation |

## 3. Matching Rules

### 3.1 IoU Threshold Criterion

A detection `D` matches a ground truth `G` if and only if:

```
IoU(D, G) ≥ threshold   AND   category_id(D) = category_id(G)
```

The **IoU threshold** is configurable. COCO standard evaluation uses 10 thresholds:
```
[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
```

Per-class evaluation means matching only occurs within the same category. A "person" detection can never match a "car" ground truth, regardless of IoU.

### 3.2 Greedy One-to-One Matching

Matching follows a greedy algorithm:

```
Input:  GT list (per category, per image)
        DT list sorted by confidence DESCENDING
        IoU threshold τ

Output: TP set, FP set, FN set

Algorithm:
  1. GT_remaining = set of all GT annotations for this (image, category)
  2. TP = []
  3. FP = []
  4. For each DT in sorted order (highest confidence first):
     a. Compute IoU against all GT in GT_remaining
     b. If max(IoU) ≥ τ:
          - Match DT with the GT that gives max IoU
          - Remove that GT from GT_remaining
          - Mark DT as TP
     c. Else:
          - Mark DT as FP
  5. FN = GT_remaining   (all unmatched GT)
```

**Key properties of this algorithm:**

- **One GT → at most one DT**: Once a GT is matched, it is removed from the pool. A second DT covering the same GT becomes FP (not a second TP).
- **One DT → at most one GT**: A DT matches the single best-overlapping GT, never multiple.
- **Confidence-sorted**: Higher-confidence detections get priority in matching. A high-confidence detection can "steal" a GT from a lower-confidence one.
- **Deterministic**: Given the same input, the algorithm always produces the same matching.

### 3.3 Per-Category Isolation

Matching is performed **independently per (image, category)**. The matching in one category does not affect another. The formula for overall metrics aggregates per-category results.

## 4. TP / FP / FN Definitions

### 4.1 Formal Definitions

For a given IoU threshold `τ` and a given confidence threshold (or after greedy matching at all confidence levels):

| Term | Definition |
|------|-----------|
| **TP** (True Positive) | A detection that successfully matched a ground truth (IoU ≥ τ, same category) |
| **FP** (False Positive) | A detection that did NOT match any ground truth — either low IoU, wrong category, or duplicate detection of an already-matched GT |
| **FN** (False Negative) | A ground truth annotation that was NOT matched by any detection |
| **TN** (True Negative) | **Not applicable in object detection** — see §4.2 |

### 4.2 Why TN Is Inapplicable

In classification, TN means "correctly predicted negative class." In object detection:

- There is **no explicit negative class** — the model outputs a variable number of detections, not a per-location label.
- The "background" covers the entire image at infinitely many positions and scales. "Correctly not detecting" at every possible location is ill-defined.
- The model's task is to **find objects**, not to classify every pixel/patch.

Consequently, metrics that depend on TN (accuracy, specificity, ROC-AUC) are **not used** in object detection evaluation. Detection metrics are built entirely on TP, FP, FN.

### 4.3 The Confusion Matrix in Detection Context

The "confusion matrix" in detection is per-class and per-IoU-threshold:

```
                 | Predicted Present | Predicted Absent
─────────────────┼───────────────────┼──────────────────
Actually Present |        TP         |        FN
Actually Absent  |        FP         |       N/A (TN undefined)
```

Note the `TN` cell is empty — see §4.2.

## 5. Confidence and Score

### 5.1 Detection Confidence

Every detection in DT must carry a `score` field (float, [0, 1]) representing the model's confidence:

```json
{
  "image_id": 1,
  "category_id": 1,
  "bbox": [320.0, 135.0, 160.0, 180.0],
  "score": 0.95
}
```

Ground truth annotations do NOT have a `score` field.

### 5.2 Confidence Threshold

A **confidence threshold** `θ_conf ∈ [0, 1]` filters detections before matching:

```
DT_filtered = { d ∈ DT | d.score ≥ θ_conf }
```

- `θ_conf = 0.0`: Keep all detections (COCO evaluation default)
- `θ_conf = 0.5`: Keep only detections with score ≥ 0.5

At lower confidence thresholds, recall increases (more detections → more chances to match GT) but precision typically decreases (more FPs).

### 5.3 The Confidence Dimension

While COCO standard evaluation uses `θ_conf = 0.0` and varies the IoU threshold, the `compute_pr_f1` API supports explicit confidence thresholding. This is useful for:

- Finding the best F1 operating point
- Model deployment where a fixed confidence threshold is used
- Per-class threshold tuning

## 6. Working Example

### 6.1 Setup

```
Image: 800×600, category: "person"

GT:
  G1: bbox=[100, 200, 150, 300]  (x, y, w, h) → [100, 200, 250, 500]
  G2: bbox=[400, 100, 200, 180]  → [400, 100, 600, 280]

DT (sorted by confidence desc):
  D1: bbox=[105, 205, 145, 290], score=0.92, category="person"
  D2: bbox=[410, 110, 190, 170], score=0.85, category="person"
  D3: bbox=[500, 300, 100, 100], score=0.60, category="person"
  D4: bbox=[110, 210, 140, 280], score=0.55, category="person"
```

### 6.2 Matching at IoU Threshold = 0.50

```
Step 1: DT D1 (score=0.92)
  IoU(D1, G1) = 0.89 ≥ 0.50 ✓
  IoU(D1, G2) = 0.00
  → Match D1→G1. TP. Remove G1 from pool.

Step 2: DT D2 (score=0.85)
  G1 is already matched. Only G2 remains.
  IoU(D2, G2) = 0.78 ≥ 0.50 ✓
  → Match D2→G2. TP. Remove G2 from pool.

Step 3: DT D3 (score=0.60)
  No GT remaining.
  IoU(D3, G1) = 0.05, IoU(D3, G2) = 0.12
  → No match (max IoU < 0.50). FP.

Step 4: DT D4 (score=0.55)
  No GT remaining (both already matched).
  IoU(D4, G1) = 0.86 ≥ 0.50 BUT G1 already matched.
  → No match. FP (duplicate of already-matched GT).

Result:
  TP = 2 (D1, D2)
  FP = 2 (D3, D4)
  FN = 0 (all GT matched)

  Precision = 2 / (2+2) = 0.50
  Recall    = 2 / (2+0) = 1.00
```

**Key observation**: D4 had high IoU with G1 but became FP because D1 already claimed G1 with higher confidence. This is the intended behavior — higher confidence wins.

## 7. Edge Cases

### 7.1 No Ground Truth (Empty Image)

If an image has zero GT for a category:
- All DT for that category on that image are FP (there is nothing to match)
- FN = 0 (nothing was missed)
- TP = 0

### 7.2 No Detections (Model Outputs Nothing)

If the model outputs zero DT for a category:
- TP = 0, FP = 0
- FN = count of GT for that category
- Precision is undefined (0/0). Convention: report as 0.0 or NaN.

### 7.3 Zero-Area Regions

Bboxes with `w ≤ 0` or `h ≤ 0`, and masks with zero area:
- **GT**: Treated as invalid — excluded from evaluation with a warning
- **DT**: Skipped in matching (cannot have IoU ≥ threshold with any GT)

### 7.4 Crowd Annotations (`iscrowd=1`)

In COCO evaluation, crowd regions are handled specially:
- A detection matching a crowd region is **not counted** as TP (it is neither TP nor FP — it is ignored)
- This prevents crowd areas from inflating FP counts
- Implementations should follow pycocotools behavior for `iscrowd` handling

### 7.5 Multiple Categories per Detection

COCO format restricts each DT to exactly one `category_id`. A detection cannot claim to be both "person" and "car" simultaneously. If a model outputs per-class scores, the evaluation pipeline should either:
- Take argmax → single category (typical for single-class-per-box models), or
- Duplicate the bbox with separate category_ids (rare; not recommended)

## 8. Validation Constraints

An evaluation session MUST verify before computing:

1. GT JSON is a valid COCO annotation file (passes `CocoHandler` validation)
2. DT JSON contains `score` field in every annotation
3. DT JSON `image_id` values exist in GT JSON images
4. DT JSON `category_id` values exist in GT JSON categories
5. At least one category has both GT and DT annotations (otherwise all metrics are undefined)
6. `iou_type` is one of `"bbox"` or `"segm"`
7. If `iou_type="segm"`, segmentation data must be present in GT and DT annotations
