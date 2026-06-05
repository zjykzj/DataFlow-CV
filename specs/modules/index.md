# Modules Layer — Specification Index

> **Status:** Canonical — these documents define the authoritative module architecture and interface contracts for DataFlow-CV.

## What This Layer Covers

The Modules layer defines **how** the software components work — their architecture, interfaces, pipelines, and constraints. These specs are the ground truth for:

- **Architecture compliance** — verifying that modules respect dependency boundaries
- **Interface contracts** — ensuring public APIs are stable and correctly implemented
- **SSD Agent development** — understanding where to add new features or fix bugs

## Architecture Constraint

```
┌──────────────────────────────────────────────────────────────┐
│                           CLI                                 │
│  (calls Convert, Visualize & Evaluate public APIs)            │
└──────┬─────────────────────┬──────────────────┬──────────────┘
       │                     │                  │
       ▼                     ▼                  ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│   Convert    │    │    Visualize     │    │   Evaluate   │
│  (pipeline)  │    │  (rendering)     │    │  (metrics)   │
└──────┬───────┘    └───────┬──────────┘    └──────┬───────┘
       │                    │                      │
       │    ZERO CROSS-     │    ZERO CROSS-       │
       │    DEPENDENCY      │    DEPENDENCY        │
       │                    │                      │
       ▼                    ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│                         Label                                 │
│  Data Models + Handlers (read/write/validate)                 │
└──────────────────────────────────────────────────────────────┘
```

**Hard constraints enforced by this architecture:**

1. **Convert ↔ Visualize**: Zero dependency. They do not import from each other.
2. **Evaluate ↔ Convert**: Zero dependency. They do not import from each other.
3. **Evaluate ↔ Visualize**: Zero dependency. They do not import from each other.
4. **Convert → Label**: Converters import handlers and models from the Label module only through public interfaces (`Handler.read()`, `Handler.write()`).
5. **Visualize → Label**: Visualizers import handlers and models from the Label module only through public interfaces.
6. **Evaluate → Label**: Evaluators import COCO handler and models from the Label module only through public interfaces.
7. **CLI → Convert/Visualize/Evaluate**: CLI commands only call converter/visualizer/evaluator public APIs. CLI must NOT import label handlers directly.

## Documents

| # | Document | Purpose |
|---|----------|---------|
| 1 | [`spec_label.md`](spec_label.md) | **Label module** — format-aware data models (`DatasetAnnotations` with `format` field, `BoundingBox`, `Segmentation` in native coordinates), handler interface (`BaseAnnotationHandler`), and concrete handlers (`YoloHandler`, `CocoHandler`, `LabelMeHandler`) |
| 2 | [`spec_convert.md`](spec_convert.md) | **Convert module** — `BaseConverter` pipeline, `ConversionResult`, three converter classes, RLE converter, state management contract |
| 3 | [`spec_visualize.md`](spec_visualize.md) | **Visualize module** — `BaseVisualizer` rendering pipeline, `ColorManager`, three visualizers, display/save modes, keyboard interaction |
| 4 | [`spec_evaluate.md`](spec_evaluate.md) | **Evaluate module** — `BaseEvaluator` pipeline, `DetectionEvaluator` / `SegmentationEvaluator`, `EvaluationResult` / `PRF1Result` data models, pycocotools wrapper, per-class metrics, P/R/F1 API |
| 5 | [`spec_cli.md`](spec_cli.md) | **CLI module** — Click-based command structure, 11 subcommands (6 convert + 3 visualize + 2 evaluate), option decorators, exception hierarchy, exit code system |

## Relationship to Formats Layer

```
Modules Layer (HOW)               Formats Layer (WHAT)
─────────────────────           ─────────────────────
spec_label.md                   spec_yolo_format.md
spec_convert.md       ──▶       spec_labelme_format.md
spec_visualize.md               spec_coco_format.md
spec_evaluate.md                spec_conversion.md
spec_cli.md

  "How does the code              "What is correct behavior?"
   achieve the behavior?"

Modules Layer (HOW)               Evaluate Layer (WHAT)
─────────────────────           ────────────────────────
spec_evaluate.md      ──▶       spec_evaluate_fundamentals.md
                                spec_evaluate_metrics.md
                                spec_evaluate_tasks.md

  "How does the code              "What metrics are correct?"
   achieve evaluation?"
```

The [Formats layer](../formats/index.md) defines the data contracts that the Label and Convert modules must implement correctly. The [Evaluate layer](../evaluate/index.md) defines the metric contracts that the Evaluate module must implement correctly.

## Reading Order

- **New to the codebase?** Start with [`SSD_AGENT.md`](../SSD_AGENT.md) for the development methodology, then `spec_label.md` (foundation), then `spec_convert.md`, then `spec_visualize.md`, then `spec_evaluate.md`, then `spec_cli.md`.
- **Adding a conversion direction?** Read `spec_convert.md` first, then reference `spec_label.md` for handler interfaces.
- **Adding a visualization?** Read `spec_visualize.md` and `spec_label.md`.
- **Adding evaluation functionality?** Read the [Evaluate layer](../evaluate/index.md) for metric definitions, then `spec_evaluate.md` for the module contract.
- **Adding a CLI command?** Read `spec_cli.md` and the relevant module spec (convert, visualize, or evaluate).
