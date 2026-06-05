# Evaluation Metrics Specification

> **Version:** 1.0
> **Status:** Draft
> **Layer:** Evaluate
> **Dependencies:** `spec_evaluate_fundamentals.md` (IoU, matching, TP/FP/FN)

## 1. Document Scope

This document defines all evaluation metrics derived from the TP/FP/FN foundation in `spec_evaluate_fundamentals.md`. Each metric includes its formula, computation method, and configuration parameters.

## 2. Precision & Recall

### 2.1 Definitions

For a given IoU threshold `τ`, confidence threshold `θ_conf`, and category `c`:

```
P(τ, θ_conf, c) = TP / (TP + FP)
R(τ, θ_conf, c) = TP / (TP + FN)
```

Where TP, FP, FN are the results of greedy matching (see `spec_evaluate_fundamentals.md` §3.2) with IoU threshold `τ`, after filtering DT by `score ≥ θ_conf`.

### 2.2 Confidence-Ranked Recall Curve

Without a fixed confidence threshold (`θ_conf = 0.0`), as we sweep the confidence threshold from highest to lowest, we accumulate:

```
For k from 1 to N (sorted by score desc):
    TP_k = cumulative TP after processing top-k DTs
    FP_k = cumulative FP after processing top-k DTs
    FN_k = total GT - TP_k

    P(k) = TP_k / (TP_k + FP_k)
    R(k) = TP_k / (TP_k + FN_k) = TP_k / total_GT
```

This gives N `(R, P)` points for the PR curve (§4).

**Observation**: As k increases (more DTs included), recall monotonically increases (more GT matched) while precision typically decreases (more FPs).

### 2.3 Overall Precision & Recall (Multi-Class)

Overall P and R are computed by aggregating TP, FP, FN across all categories:

```
TP_overall = sum(TP_c for c in categories)
FP_overall = sum(FP_c for c in categories)
FN_overall = sum(FN_c for c in categories)

P_overall = TP_overall / (TP_overall + FP_overall)
R_overall = TP_overall / (TP_overall + FN_overall)
```

## 3. F1-Score

### 3.1 Definition

The F1-score is the harmonic mean of Precision and Recall:

```
F1 = 2 × P × R / (P + R)
```

Range: `[0, 1]`. F1 = 1 only when both P and R are 1. F1 penalizes imbalance between P and R.

### 3.2 Per-Category vs Overall

F1 is computed both:
- **Per-class**: `F1_c = 2 × P_c × R_c / (P_c + R_c)`
- **Overall**: `F1_overall = 2 × P_overall × R_overall / (P_overall + R_overall)`

### 3.3 Single-Threshold F1 (PRF1 API)

The `compute_pr_f1()` API computes F1 at a single IoU threshold and confidence threshold:

```
Input:  GT, DT, iou_threshold, confidence_threshold
Output: per-class {P, R, F1, TP, FP, FN} + overall {P, R, F1, TP, FP, FN}
```

### 3.4 Best F1 Across Confidence Thresholds

For a fixed IoU threshold, the best F1 across all confidence thresholds is:

```
best_F1(τ) = max_{k ∈ [1, N]} F1(P(k), R(k))
```

This corresponds to the optimal operating point on the PR curve.

## 4. PR Curve

### 4.1 Construction

The Precision-Recall curve plots P against R as the confidence threshold varies:

1. Sort all DT by confidence descending
2. For each unique score value, compute cumulative (P, R)
3. Plot the resulting points

**Shape**: P typically decreases as R increases (zig-zag pattern). The curve starts near P≈1, R≈0 and ends at P≈(total_GT / total_DT), R≈1.

### 4.2 Interpolation Methods

Before computing AP (area under PR curve), the raw PR curve is interpolated to remove zig-zag fluctuations:

#### All-Point Interpolation (COCO Standard — Current)

```
P_interp(r) = max_{r' ≥ r} P(r')
```

For every recall level `r`, take the maximum precision at any recall ≥ r. This makes the PR curve monotonically decreasing.

#### 101-Point Interpolation (COCO Legacy — Deprecated)

```
P_interp(r) = max_{r' ≥ r} P(r')
```
Sampled at 101 uniformly spaced recall levels: `[0.00, 0.01, 0.02, ..., 1.00]`.

Historical note: This was used in earlier COCO challenges (pre-2017). Modern COCO evaluation and DataFlow-CV use **all-point interpolation** by default.

### 4.3 Interpolation Comparison

| Property | All-Point | 101-Point |
|----------|-----------|-----------|
| Sample points | Every unique recall value in the data | 101 fixed recall levels |
| Accuracy | More accurate (uses all data) | Slightly underestimates AP |
| AP difference | — | Typically 0.1-0.5 AP lower |
| pycocotools | Default | Available via parameter change |

## 5. AP — Average Precision

### 5.1 Definition

AP is the area under the interpolated PR curve:

```
AP = ∫[0,1] P_interp(r) dr
```

In discrete form (all-point interpolation):

```
AP = Σ[i=0 to N-1] (R_{i+1} - R_i) × P_interp(R_{i+1})
```

Where `(R_i, P_i)` are the points on the interpolated PR curve, sorted by recall ascending.

### 5.2 Interpretation

AP summarizes the PR curve into a single number:
- Range: `[0, 1]`
- Higher is better
- AP = 1 requires P=1 at all recall levels (perfect detection)
- A model that outputs many detections may have high recall but lower AP (due to FPs)
- A model that outputs few, high-confidence detections may have high precision but lower AP (due to FNs)

### 5.3 Per-Category AP

AP is computed per category `c`:

```
AP_c = area under PR curve for category c only
```

Where matching only considers GTs and DTs of category `c`.

## 6. mAP — Mean Average Precision

### 6.1 Definition

mAP is the arithmetic mean of per-category AP across all `K` categories:

```
mAP = (1/K) × Σ[c=1 to K] AP_c
```

Each category contributes equally, regardless of its GT count. This prevents categories with many annotations from dominating the metric.

### 6.2 Variants by IoU Threshold

| Notation | IoU Threshold(s) | Description |
|----------|-----------------|-------------|
| **mAP50** | τ = 0.50 only | Single IoU threshold. Historically common (Pascal VOC standard). Lenient — moderate overlap counts as correct. |
| **mAP75** | τ = 0.75 only | Strict threshold. Requires precise localization. |
| **mAP** (or mAP50_95) | τ ∈ [0.50, 0.55, ..., 0.95] | COCO standard. Average over 10 IoU thresholds. Balances lenient and strict matching. This is the **default** meaning of "AP" in modern evaluation. |

### 6.3 mAP50_95 Computation

```
mAP50_95 = (1/10) × Σ[τ ∈ thresholds] mAP(τ)

thresholds = {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95}
```

Where `mAP(τ)` is the mAP computed with IoU threshold `τ`.

**Why average over IoU?**: A single IoU threshold (like 0.5) gives an incomplete picture:
- mAP50 = 0.80 but mAP75 = 0.30 → model finds objects but localization is sloppy
- mAP50 = 0.80 and mAP75 = 0.78 → model has both detection and localization ability

The 10-threshold average captures this spectrum.

### 6.4 Edge Case: Category with Zero GT

If a category `c` has zero GT annotations:
- AP_c is undefined (no recall levels to compute)
- This category is **excluded** from mAP averaging
- mAP = mean over categories with GT > 0

## 7. AR — Average Recall

### 7.1 Definition

AR measures the maximum recall the model can achieve, given a limit on the number of detections per image (`maxDets`):

```
AR(maxDets) = (2/10) × Σ[τ ∈ thresholds] Recall(τ, maxDets)
```

Where `Recall(τ, maxDets)` is the recall achieved when taking the top `maxDets` detections (by score) per image, evaluated at IoU threshold `τ`.

### 7.2 Why AR Uses maxDets

Unlike AP which evaluates across all confidence levels, AR measures the recall ceiling:

| maxDets | Meaning |
|---------|---------|
| 1 | Can the model find the object with its top-1 detection? |
| 10 | Can the model find objects with up to 10 detections per image? |
| 100 | Upper bound — with many detections, what's the maximum recall? |

A model with AR@1 = 0.30 and AR@100 = 0.75 has decent recognition ability but poor confidence ranking (the correct detection often isn't ranked #1).

### 7.3 AR by Object Scale

Same as AP, AR is stratified by object size:
- **AR_small**: GT area < 32² px
- **AR_medium**: 32² ≤ GT area < 96² px
- **AR_large**: GT area ≥ 96² px

## 8. Scale Stratification

### 8.1 Area Thresholds

Object size categories are defined by **GT segmentation area** (or bbox area if segmentation is absent):

| Label | Area Range | Typical Objects |
|-------|-----------|----------------|
| **small** | area < 32² = 1024 px² | Distant persons, traffic signs, small animals |
| **medium** | 1024 ≤ area < 96² = 9216 px² | Pedestrians, cars at moderate distance |
| **large** | area ≥ 9216 px² | Close-up objects, large vehicles |

### 8.2 Scale-Stratified AP & AR

For each scale `s ∈ {small, medium, large}`:

```
AP_s  = AP considering only GT of scale s
AR_s  = AR considering only GT of scale s
```

DT matching is not filtered by DT size — a large DT can match a small GT (and vice versa), though the IoU will naturally be lower if size mismatch is severe.

### 8.3 Interpretation

Scale-stratified metrics reveal a model's blind spots:

| Pattern | Interpretation |
|---------|---------------|
| APl >> APs | Model struggles with small objects (common for CNN-based detectors) |
| APs ≈ APl ≈ APm | Good multi-scale detection |
| ARs ≈ ARl but APs << APl | Model finds small objects (high recall) but localizes them poorly (low precision) |

## 9. Metric Summary Table

| Metric | Parameters | Category Aggregation | IoU Thresholds |
|--------|-----------|---------------------|----------------|
| P | iou_thr, conf_thr | Overall or per-class | Single |
| R | iou_thr, conf_thr | Overall or per-class | Single |
| F1 | iou_thr, conf_thr | Overall or per-class | Single |
| AP | — | Per-class | Single or averaged (10 thresholds) |
| mAP50 | — | Mean over classes | Single (0.50) |
| mAP75 | — | Mean over classes | Single (0.75) |
| mAP (mAP50_95) | — | Mean over classes | 10 thresholds [0.50:0.95] |
| AP_small | — | Mean over classes | 10 thresholds, GT area < 1024 |
| AP_medium | — | Mean over classes | 10 thresholds, GT area ∈ [1024, 9216) |
| AP_large | — | Mean over classes | 10 thresholds, GT area ≥ 9216 |
| AR@maxDets | maxDets ∈ {1, 10, 100} | Mean over classes | 10 thresholds |
| AR_small/_medium/_large | maxDets=100 | Mean over classes | 10 thresholds, GT area filter |

## 10. Relationship to pycocotools

This specification defines metrics consistent with the **COCO evaluation protocol** as implemented in pycocotools (`COCOeval` class). The implementation in `dataflow/evaluate/` wraps pycocotools and:

1. Calls `COCOeval.evaluate()`, `COCOeval.accumulate()`, `COCOeval.summarize()` for COCO-standard evaluation
2. Extracts `COCOeval.stats` for the 12 standard metrics
3. Computes per-class breakdowns from `COCOeval.eval` arrays
4. Provides the `compute_pr_f1()` convenience method for single-threshold P/R/F1

The formulas in this spec are the **contract**; pycocotools is the **reference implementation**.
