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

Both GT and DT use COCO JSON as the **de facto standard exchange format** for detection evaluation, regardless of the original annotation format (YOLO, LabelMe, etc.).

**GT JSON requirements:**
- Must be a full COCO dict with `images`, `categories`, and `annotations` arrays
- `images`: each entry has `id`, `width`, `height`
- `categories`: each entry has `id`, `name`
- `annotations`: each entry has `id`, `image_id`, `category_id`, `bbox`

**DT JSON requirements — two valid formats:**

1. **Full COCO dict** (same structure as GT) — each annotation additionally includes `score` (float ∈ [0,1]). This is the output of Convert module tools (e.g., `yolo2coco --prediction`).

2. **Plain annotation list** (JSON array) — a top-level `[{...}, {...}, ...]` array where each entry is an annotation dict with `image_id`, `category_id`, `bbox`, and `score`. No `images` or `categories` arrays. This is the most common output from model inference frameworks (Detectron2, MMDetection, custom scripts). At load time, `images` and `categories` are sourced from GT via `COCO.loadRes()`.

**Common to both formats:**
- `score`: float ∈ [0, 1], higher = more confident detection (required on every DT annotation)

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

## 6. Mixed Datasets

When a dataset contains some annotations with segmentation and some without:

1. **Detection evaluation**: All annotations are usable. Those without segmentation are evaluated with bbox IoU.
2. **Segmentation evaluation**: Annotations without segmentation are **excluded** from evaluation with a warning.
3. The Evaluate module should log a warning when the dataset is mixed, to alert the user.
