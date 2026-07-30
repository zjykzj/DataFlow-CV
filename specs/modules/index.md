# Modules Layer — Specification Index

> **Version:** v1.2 | **Last Updated:** 2026-07-30
> **Status:** Canonical — these documents define the authoritative module architecture and interface contracts for DataFlow-CV.

## What This Layer Covers

The Modules layer defines **how** the software components work — their architecture, interfaces, pipelines, and constraints. These specs are the ground truth for:

- **Architecture compliance** — verifying that modules respect dependency boundaries
- **Interface contracts** — ensuring public APIs are stable and correctly implemented
- **SDD Agent development** — understanding where to add new features or fix bugs

## Architecture Constraint

```
┌──────────────────────────────────────────────────────────────┐
│                           CLI                                 │
│  (passes LogConfig to modules; click.echo() for terminal UI)  │
└──┬──────────┬──────────────┬──────────────┬──────────────────┘
   │          │              │              │
   ▼          ▼              ▼              ▼
┌──────────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐
│   Analyse    │ │Convert │ │Visualize │ │Evaluate  │
│(introspection)│ │(pipeline)│ │(rendering)│ │(metrics) │
│  LogManager  │ │LogMgr  │ │LogMgr    │ │LogMgr    │
└──────┬───────┘ └──┬─────┘ └────┬─────┘ └────┬─────┘
       │            │            │            │
       │   ZERO CROSS-DEPENDENCY │            │
       │            │            │            │
       ▼            ▼            ▼            ▼
┌──────────────────────────────────────────────────────────────┐
│                         Label                                 │
│  Data Models + Handlers (read/write/validate)                 │
│  (receive logger from calling module)                         │
└──────────────────────────────────────────────────────────────┘
       │            │            │            │
       └────────────┼────────────┼────────────┘
                    │            │
                    ▼            ▼
┌──────────────────────────────────────────────────────────────┐
│                    util/logging.py                             │
│  LogManager + format helpers (shared infrastructure)           │
└──────────────────────────────────────────────────────────────┘
```

**Hard constraints enforced by this architecture:**

1. **Analyse ↔ Convert/Visualize/Evaluate**: Zero dependency. Analyse does not import from any of them, and vice versa.
2. **Convert ↔ Visualize**: Zero dependency. They do not import from each other.
3. **Evaluate ↔ Convert**: Zero dependency. They do not import from each other.
4. **Evaluate ↔ Visualize**: Zero dependency. They do not import from each other.
5. **Analyse → Label**: Analysers import handlers and models from the Label module only through public interfaces.
6. **Convert → Label**: Converters import handlers and models from the Label module only through public interfaces (`Handler.read()`, `Handler.write()`).
7. **Visualize → Label**: Visualizers import handlers and models from the Label module only through public interfaces.
8. **Evaluate → Label**: Evaluators import COCO handler and models from the Label module only through public interfaces.
9. **CLI → Analyse/Convert/Visualize/Evaluate**: CLI commands only call module public APIs. CLI must NOT import label handlers directly.
10. **Logging ownership**: All log output is produced by modules, not CLI. CLI passes `LogConfig` to module constructors and uses `click.echo()` for terminal UI. See [`spec_logging.md`](spec_logging.md).

## Documents

| # | Document | Purpose |
|---|----------|---------|
| 1 | [`spec_label.md`](spec_label.md) | **Label module** — format-aware data models (`DatasetAnnotations` with `format` field, `BoundingBox`, `Segmentation` in native coordinates), handler interface (`BaseAnnotationHandler`), and concrete handlers (`YoloHandler`, `CocoHandler`, `LabelMeHandler`) |
| 2 | [`spec_analyse.md`](spec_analyse.md) | **Analyse module** — `BaseAnalyser` + `StatsAnalyser` + `SplitAnalyser`, `AnalysisResult` / `StatsResult` / `SplitResult` data models, format auto-detection, dataset statistics, train/test split |
| 3 | [`spec_convert.md`](spec_convert.md) | **Convert module** — `BaseConverter` pipeline, `ConversionResult`, three converter classes, RLE converter, state management contract |
| 4 | [`spec_visualize.md`](spec_visualize.md) | **Visualize module** — `BaseVisualizer` rendering pipeline, `ColorManager`, three visualizers, display/save modes, keyboard interaction |
| 5 | [`spec_evaluate.md`](spec_evaluate.md) | **Evaluate module** — `BaseEvaluator` pipeline, `DetectionEvaluator` / `SegmentationEvaluator`, `EvaluationResult` / `PRF1Result` data models, pycocotools wrapper, per-class metrics, P/R/F1 API |
| 6 | [`spec_cli.md`](spec_cli.md) | **CLI module** — Click-based command structure, 16 subcommands (5 analyse + 6 convert + 3 visualize + 2 evaluate), option decorators, exception hierarchy, exit code system |
| 7 | [`spec_logging.md`](spec_logging.md) | **Logging module** — `LogManager` unified logging infrastructure, `LogConfig`, format helpers, module log templates, CLI logging contract |

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

- **New to the codebase?** Start with `spec_label.md` (foundation), then `spec_logging.md` (infrastructure), then `spec_analyse.md`, then `spec_convert.md`, then `spec_visualize.md`, then `spec_evaluate.md`, then `spec_cli.md`.
- **Adding a conversion direction?** Read `spec_convert.md` first, then reference `spec_label.md` for handler interfaces.
- **Adding a visualization?** Read `spec_visualize.md` and `spec_label.md`.
- **Adding evaluation functionality?** Read the [Evaluate layer](../evaluate/index.md) for metric definitions, then `spec_evaluate.md` for the module contract.
- **Adding dataset analysis or splitting?** Read `spec_analyse.md` and `spec_label.md`.
- **Adding a CLI command?** Read `spec_cli.md` and the relevant module spec (convert, visualize, evaluate, or analyse).
- **Working on logging?** Read `spec_logging.md` for the `LogManager` contract and module integration patterns.
