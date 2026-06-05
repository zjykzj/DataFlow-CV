# Evaluation Task Specification

> **Version:** 1.0
> **Status:** Draft
> **Layer:** Evaluate
> **Dependencies:** `spec_evaluate_fundamentals.md` (IoU, matching, TP/FP/FN), `spec_evaluate_metrics.md` (P/R/F1, AP/mAP, AR, scales)

## 1. Document Scope

This document applies the metrics framework (defined in fundamentals + metrics specs) to concrete evaluation tasks: object detection and instance segmentation. It defines input requirements, the COCO standard 12-metric table, and provides a task selection guide.

## 2. Task 1: Object Detection Evaluation

### 2.1 Overview

**IoU type**: `bbox` (`iouType='bbox'`)

Object detection evaluation measures how well predicted bounding boxes match ground truth bounding boxes. The model outputs per-image detection results; each detection includes a bbox, category, and confidence score.

### 2.2 Input Format

Both GT and DT must be in COCO JSON format. This is the **de facto standard exchange format** for detection evaluation, regardless of the original annotation format (YOLO, LabelMe, etc.).

**GT JSON requirements:**
- `images` array with `id`, `width`, `height`
- `categories` array with `id`, `name`
- `annotations` array with `id`, `image_id`, `category_id`, `bbox`

**DT JSON requirements:**
- Same structure as GT, plus `score` field in each annotation
- `score`: float ∈ [0, 1], higher = more confident detection

### 2.3 Matching Method

Bbox IoU (§2.1 of fundamentals spec) between GT `[x, y, w, h]` and DT `[x, y, w, h]`. Both are in absolute pixel coordinates (COCO standard).

### 2.4 Metrics Produced

All metrics from `spec_evaluate_metrics.md` apply:
- Full 12 COCO standard metrics (§4)
- Per-class AP, AP50, AP75, P, R, F1
- Optional: single-threshold P/R/F1 via `compute_pr_f1()` API

### 2.5 Input Requirements Summary

| GT Annotation Field | Required | DT Annotation Field | Required |
|--------------------|----------|--------------------|----------|
| `image_id` | Yes | `image_id` | Yes |
| `category_id` | Yes | `category_id` | Yes |
| `bbox` | Yes | `bbox` | Yes |
| `segmentation` | No | `score` | **Yes** (DT only) |
| `area` | Recommended | `segmentation` | No |

## 3. Task 2: Instance Segmentation Evaluation

### 3.1 Overview

**IoU type**: `segm` (`iouType='segm'`)

Instance segmentation evaluation measures how well predicted masks match ground truth masks. The primary difference from detection (§2) is the IoU type: mask IoU instead of bbox IoU.

### 3.2 Input Format

Same COCO JSON structure as detection, with the addition of `segmentation` data:

**GT JSON requirements:**
- All detection requirements, plus
- `annotations[].segmentation`: polygon `[[x1,y1,...], ...]` or RLE `{"size": [h,w], "counts": "..."}`
- `annotations[].area`: mask area in square pixels

**DT JSON requirements:**
- Same as detection DT, plus `segmentation` field (polygon or RLE format)
- `score` field (unchanged from detection)

### 3.3 Matching Method

Mask IoU (§2.2 of fundamentals spec) computed via:
1. GT mask → decoded from polygon/RLE to binary mask
2. DT mask → decoded from polygon/RLE to binary mask
3. Pixel-level IoU computation

**Important**: This requires pycocotools to be installed. Without pycocotools, mask IoU cannot be computed. The Evaluate module MUST raise a clear error if `iouType='segm'` is requested but pycocotools is not available.

### 3.4 Metrics Produced

Identical metric structure to detection — all 12 COCO standard metrics, per-class breakdown, P/R/F1. The only difference is that IoU = mask IoU instead of bbox IoU.

### 3.5 Input Requirements Summary

| GT Annotation Field | Required | DT Annotation Field | Required |
|--------------------|----------|--------------------|----------|
| `image_id` | Yes | `image_id` | Yes |
| `category_id` | Yes | `category_id` | Yes |
| `bbox` | Yes (for area) | `bbox` | Yes (for area) |
| `segmentation` | **Yes** | `segmentation` | **Yes** |
| `area` | **Yes** | `score` | **Yes** (DT only) |
| `iscrowd` | Yes (crowd regions) | — | — |

## 4. COCO Standard 12-Metric Table

The following 12 metrics are the standard output of COCO evaluation. These are produced by both detection and segmentation evaluation.

### 4.1 Average Precision (AP) Metrics

| # | Metric | IoU | Area | maxDets | Description |
|---|--------|-----|------|---------|-------------|
| 0 | **AP** | 0.50:0.95 | all | 100 | Primary COCO metric. AP averaged over 10 IoU thresholds, all object sizes. |
| 1 | **AP50** | 0.50 | all | 100 | AP at single IoU=0.50 (Pascal VOC metric). Less strict on localization. |
| 2 | **AP75** | 0.75 | all | 100 | AP at single IoU=0.75. Strict localization requirement. |
| 3 | **AP_small** | 0.50:0.95 | small (<32²) | 100 | AP for small objects only. |
| 4 | **AP_medium** | 0.50:0.95 | medium (32²–96²) | 100 | AP for medium objects only. |
| 5 | **AP_large** | 0.50:0.95 | large (≥96²) | 100 | AP for large objects only. |

### 4.2 Average Recall (AR) Metrics

| # | Metric | IoU | Area | maxDets | Description |
|---|--------|-----|------|---------|-------------|
| 6 | **AR@1** | 0.50:0.95 | all | 1 | Max recall given 1 detection per image. Measures top-1 confidence quality. |
| 7 | **AR@10** | 0.50:0.95 | all | 10 | Max recall given 10 detections per image. |
| 8 | **AR@100** | 0.50:0.95 | all | 100 | Max recall given 100 detections per image. Upper-bound recall. |
| 9 | **AR_small** | 0.50:0.95 | small (<32²) | 100 | Max recall for small objects. |
| 10 | **AR_medium** | 0.50:0.95 | medium (32²–96²) | 100 | Max recall for medium objects. |
| 11 | **AR_large** | 0.50:0.95 | large (≥96²) | 100 | Max recall for large objects. |

### 4.3 Metric Selection Guide

| Evaluation Goal | Primary Metric | Secondary Metric |
|----------------|---------------|-----------------|
| General model comparison | AP (mAP50_95) | AP50, AP75 |
| Deployment with fixed IoU requirement | AP50 or AP75 | AP |
| Small object detection | AP_small | AR_small |
| Recall-oriented task (e.g., search) | AR@100 | AR@10 |
| Confidence calibration check | AR@1 vs AR@100 | — |
| Per-class performance analysis | Per-class AP | Per-class AP50, F1 |

## 5. Detection vs Segmentation — Side-by-Side

| Dimension | Detection | Segmentation |
|-----------|-----------|-------------|
| IoU type (`iouType`) | `bbox` | `segm` |
| IoU computation | Geometric (O(1)) | Pixel-level (O(H×W)) |
| Additional input needed | None | Polygon or RLE segmentation per GT and DT |
| pycocotools required | Yes (for eval) | Yes (for eval + mask encoding) |
| RLE encoding | N/A | Required for mask IoU computation |
| `iscrowd` handling | N/A | Crowd regions excluded from matching |
| Speed | Fast | Slower (mask rasterization overhead) |
| Use case | Bounding box detectors (YOLO, Faster R-CNN, etc.) | Instance segmentation models (Mask R-CNN, SOLO, etc.) |

## 6. Task Selection Guide

### 6.1 Decision Matrix

| GT Annotations Have | DT Annotations Have | Recommended Task |
|--------------------|--------------------|--------------------|
| bbox only | bbox + score | **Detection** (`iouType='bbox'`) |
| segm only | segm + score | **Segmentation** (`iouType='segm'`) |
| bbox + segm | bbox + score | **Detection** (can also evaluate segm models) |
| bbox + segm | segm + bbox + score | **Segmentation** (primary) + **Detection** (secondary, for bbox quality) |

### 6.2 Mixed Datasets

When a dataset contains some annotations with segmentation and some without:

1. **Detection evaluation**: All annotations are usable. Those without segmentation are evaluated with bbox IoU.
2. **Segmentation evaluation**: Annotations without segmentation are **excluded** from evaluation with a warning.
3. The Evaluate module should log a warning when the dataset is mixed, to alert the user.

### 6.3 Running Both Evaluations

It is valid to run both detection and segmentation evaluation on the same GT/DT pair:

```python
det_result = evaluator_det.evaluate(gt, dt)    # iouType='bbox'
seg_result = evaluator_seg.evaluate(gt, dt)    # iouType='segm'
```

The results are independent and complementary — bbox AP measures localization, mask AP measures pixel-level accuracy.

## 7. Non-COCO Input Conversion

### 7.1 YOLO Format

YOLO annotations must be converted to COCO format before evaluation. Use the Convert module with the `--prediction` flag for prediction files:

```
YOLO GT (.txt, 5 tokens)  → [Convert: yolo2coco]                 → COCO JSON GT (anno.json)
YOLO DT (.txt, 6 tokens)  → [Convert: yolo2coco --prediction]    → COCO JSON DT (pred.json)
```

**Format details:**

| Step | YOLO Format | Tokens | CLI Command |
|------|------------|--------|-------------|
| Ground Truth | `class_id cx cy w h` | 5 | `yolo2coco images/ labels/ classes.txt anno.json` |
| Detection Prediction | `class_id cx cy w h confidence` | 6 | `yolo2coco --prediction images/ preds/ classes.txt pred.json` |
| Segmentation Prediction | `class_id x1 y1 ... xn yn confidence` | even > 6 | `yolo2coco --prediction images/ preds/ classes.txt pred.json` |

The `--prediction` flag tells the YoloHandler to parse prediction format (even token count with trailing confidence) instead of label format (odd token count). The confidence value is preserved as the COCO `score` field in the output JSON.

For segmentation predictions, the output COCO JSON stores segmentation as polygon format by default. pycocotools `COCOeval` converts polygon → RLE internally during evaluation. The `--do-rle` flag is available for smaller file size but is not required.

### 7.2 LabelMe Format

Same pattern — convert to COCO first:

```
LabelMe GT (.json per image) → [Convert: labelme2coco] → COCO JSON GT
LabelMe DT (.json per image + score) → [Convert: labelme2coco] → COCO JSON DT
```

### 7.3 Conversion Note

The Evaluate module does **not** perform format conversion. It accepts COCO-format input only (by file path or in-memory dict). Users are responsible for converting their data to COCO format before evaluation.

The CLI may optionally accept non-COCO formats and perform conversion automatically (see `spec_cli.md`), but the core Evaluate module's contract is COCO-in/COCO-out.
