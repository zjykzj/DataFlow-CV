# Evaluate Layer — Specification Index

> **Version:** v1.1 | **Last Updated:** 2026-07-30
> **Status:** Canonical — these documents define the authoritative evaluation metric contracts for DataFlow-CV.

## What This Layer Covers

The Evaluate layer defines **what** constitutes correct evaluation of object detection and instance segmentation algorithms. These specs are the ground truth for:

- **Metric calculation** — ensuring mAP, AP50, AR, and other metrics are computed to COCO standard
- **Matching rules** — defining IoU-based greedy matching between ground truth and detections
- **Task differentiation** — clarifying the difference between detection (bbox IoU) and segmentation (mask IoU) evaluation
- **Implementation compliance** — verifying that `dataflow/evaluate/` produces correct results

## Layer Architecture

```
Evaluate Layer (WHAT)              Modules Layer (HOW)
─────────────────────           ─────────────────────
spec_evaluate_fundamentals.md   spec_evaluate.md
spec_evaluate_metrics.md  ──▶
spec_evaluate_tasks.md

  "What is correct?"             "How does the code achieve it?"
```

The Evaluate layer defines the metric computing contract. The [Modules layer](../modules/index.md) defines the `dataflow/evaluate/` module that implements it.

## Relationship to Other Layers

```
specs/
├── evaluate/          # WHAT — metric definitions (THIS LAYER)
│   ├── index.md                          ← You are here
│   ├── spec_evaluate_fundamentals.md     # IoU, matching, TP/FP/FN, confusion matrix
│   ├── spec_evaluate_metrics.md          # P/R/F1, PR curve, AP/mAP/AR, scale stratification
│   └── spec_evaluate_tasks.md            # Detection vs segmentation, COCO 12 metrics, task guide
│
├── formats/           # WHAT — data format contracts (independent of evaluate)
│   ├── index.md
│   ├── spec_yolo_format.md
│   ├── spec_labelme_format.md
│   ├── spec_coco_format.md
│   └── spec_conversion.md
│
└── modules/           # HOW — module implementations
    ├── index.md
    ├── spec_label.md
    ├── spec_convert.md
    ├── spec_visualize.md
    ├── spec_evaluate.md                # Evaluate module implementation
    └── spec_cli.md
```

**Key design**: The Evaluate layer is **independent** of the Formats layer. Evaluation works on COCO-format annotations (the de facto exchange standard for detection/segmentation evaluation), regardless of how the data was originally stored (YOLO, LabelMe, or COCO). The Evaluate module internally converts input to COCO format when needed.

## Documents

| # | Document | Purpose |
|---|----------|---------|
| 1 | [`spec_evaluate_fundamentals.md`](spec_evaluate_fundamentals.md) | **Fundamental concepts** — bbox IoU, mask IoU, greedy matching rules, TP/FP/FN definitions, confusion matrix, why TN is inapplicable in detection |
| 2 | [`spec_evaluate_metrics.md`](spec_evaluate_metrics.md) | **Core metrics** — Precision & Recall, F1-score, PR curve (all-point vs 101-point interpolation), AP/mAP/mAP50/mAP75/mAP50_95, AR, scale stratification (small/medium/large) |
| 3 | [`spec_evaluate_tasks.md`](spec_evaluate_tasks.md) | **Task application** — detection vs segmentation evaluation differences, COCO standard 12-metric table, task selection guide (bbox-only, segm-only, mixed) |

## Reading Order

- **New to evaluation metrics?** Read all three in order: fundamentals → metrics → tasks.

- **Implementing the evaluate module?** Read all three evaluate specs, then [`spec_evaluate.md`](../modules/spec_evaluate.md) for the module contract.
- **Using evaluation in CLI?** Read `spec_evaluate_tasks.md` for the task guide, then [`spec_cli.md`](../modules/spec_cli.md) for command signatures.
- **Auditing metric correctness?** Use `spec_evaluate_metrics.md` formulas as your ground truth.

## Conventions

### Metric Notation

| Symbol | Meaning |
|--------|---------|
| AP | Average Precision (generic, context-dependent IoU threshold) |
| mAP | Mean Average Precision (AP averaged over all categories) |
| AP50 | AP at IoU = 0.50 |
| AP75 | AP at IoU = 0.75 |
| AP50_95 (or AP) | AP averaged over 10 IoU thresholds [0.50:0.05:0.95] |
| AR | Average Recall |
| APs/APm/APl | AP for small / medium / large objects |

Unless explicitly stated otherwise, "AP" in this layer refers to AP50_95 (COCO default).

### IoU Type Notation

| Symbol | Meaning |
|--------|---------|
| `iouType='bbox'` | Use bounding box IoU for matching (detection) |
| `iouType='segm'` | Use mask IoU for matching (segmentation) |
