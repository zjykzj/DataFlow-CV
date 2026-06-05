# Formats Layer — Specification Index

> **Status:** Canonical — these documents define the authoritative external format contracts for DataFlow-CV.

## What This Layer Covers

The Formats layer defines **what** constitutes valid annotation data in each external format. These specs are the ground truth for:

- **Handler validation** — determining whether an input file is valid
- **Conversion correctness** — verifying coordinate transforms produce valid output
- **SSD Agent compliance** — checking that read/write operations conform to the format contract

## Documents

| # | Document | Purpose |
|---|----------|---------|
| 1 | [`spec_yolo_format.md`](spec_yolo_format.md) | YOLO `.txt` format authority — file structure, detection/segmentation line formats, center-based normalized coordinate system, validation constraints |
| 2 | [`spec_labelme_format.md`](spec_labelme_format.md) | LabelMe `.json` format authority — JSON structure, rectangle/polygon shape types, corner-order agnosticism, optional fields |
| 3 | [`spec_coco_format.md`](spec_coco_format.md) | COCO `.json` format authority — JSON schema, top-left absolute-pixel bbox, polygon/RLE segmentation, latin1 RLE encoding, crowd annotations |
| 4 | [`spec_conversion.md`](spec_conversion.md) | Conversion rules authority — coordinate transform formulas for all 6 directions, category mapping, explicit precision documentation, round-trip fidelity matrix |

## Relationship to Modules Layer

```
Formats Layer (WHAT)              Modules Layer (HOW)
─────────────────────           ─────────────────────
spec_yolo_format.md             spec_label.md
spec_coco_format.md    ──▶      spec_convert.md
spec_labelme_format.md          spec_visualize.md
spec_conversion.md              spec_evaluate.md
                                spec_cli.md

  "What is correct?"             "How does the code achieve it?"
```

The Formats layer defines the data contract. The [Modules layer](../modules/index.md) defines the software components that read, write, convert, visualize, and evaluate that data.

Note: The [Evaluate layer](../evaluate/index.md) is an independent third layer — it defines evaluation metric contracts (WHAT) that the Evaluate module (HOW, in `specs/modules/spec_evaluate.md`) implements.

## Reading Order

- **New to annotation formats?** Start with all four in order (YOLO → LabelMe → COCO → Conversion).
- **New to the project?** Read [`SSD_AGENT.md`](../SSD_AGENT.md) first for the development workflow.
- **Implementing a handler?** Read the relevant format spec + `spec_conversion.md` for coordinate transforms.
- **Auditing compliance?** Use each format spec's Validation Rules section as your checklist.
