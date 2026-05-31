# Modules Layer — Specification Index

> **Status:** Canonical — these documents define the authoritative module architecture and interface contracts for DataFlow-CV.

## What This Layer Covers

The Modules layer defines **how** the software components work — their architecture, interfaces, pipelines, and constraints. These specs are the ground truth for:

- **Architecture compliance** — verifying that modules respect dependency boundaries
- **Interface contracts** — ensuring public APIs are stable and correctly implemented
- **SSD Agent development** — understanding where to add new features or fix bugs

## Architecture Constraint

```
┌──────────────────────────────────────────────┐
│                    CLI                        │
│  (calls Convert & Visualize public APIs)      │
└──────┬─────────────────────┬─────────────────┘
       │                     │
       ▼                     ▼
┌──────────────┐    ┌──────────────────┐
│   Convert    │    │    Visualize     │
│  (pipeline)  │    │  (rendering)     │
└──────┬───────┘    └───────┬──────────┘
       │                    │
       │    ZERO CROSS-     │
       │    DEPENDENCY      │
       │                    │
       ▼                    ▼
┌──────────────────────────────────────────────┐
│                   Label                       │
│  Data Models + Handlers (read/write/validate) │
└──────────────────────────────────────────────┘
```

**Hard constraints enforced by this architecture:**

1. **Convert ↔ Visualize**: Zero dependency. They do not import from each other.
2. **Convert → Label**: Converters import handlers and models from the Label module only through public interfaces (`Handler.read()`, `Handler.write()`).
3. **Visualize → Label**: Visualizers import handlers and models from the Label module only through public interfaces.
4. **CLI → Convert/Visualize**: CLI commands only call converter/visualizer public APIs. CLI must NOT import label handlers directly.

## Documents

| # | Document | Purpose |
|---|----------|---------|
| 1 | [`spec_label.md`](spec_label.md) | **Label module** — data models (`DatasetAnnotations`, `BoundingBox`, `Segmentation`, `OriginalData`), handler interface (`BaseAnnotationHandler`), and concrete handlers (`YoloHandler`, `CocoHandler`, `LabelMeHandler`) |
| 2 | [`spec_convert.md`](spec_convert.md) | **Convert module** — `BaseConverter` pipeline, `ConversionResult`, three converter classes, RLE converter, state management contract |
| 3 | [`spec_visualize.md`](spec_visualize.md) | **Visualize module** — `BaseVisualizer` rendering pipeline, `ColorManager`, three visualizers, display/save modes, keyboard interaction |
| 4 | [`spec_cli.md`](spec_cli.md) | **CLI module** — Click-based command structure, 9 subcommands (6 convert + 3 visualize), option decorators, exception hierarchy, exit code system |

## Relationship to Formats Layer

```
Modules Layer (HOW)               Formats Layer (WHAT)
─────────────────────           ─────────────────────
spec_label.md                   spec_yolo_format.md
spec_convert.md       ──▶       spec_coco_format.md
spec_visualize.md               spec_labelme_format.md
spec_cli.md                     spec_conversion.md

  "How does the code              "What is correct behavior?"
   achieve the behavior?"
```

The [Formats layer](../formats/index.md) defines the data contracts that the Modules layer must implement correctly.

## Reading Order

- **New to the codebase?** Start with [`SSD_AGENT.md`](../SSD_AGENT.md) for the development methodology, then `spec_label.md` (foundation), then `spec_convert.md`, then `spec_visualize.md`, then `spec_cli.md`.
- **Adding a conversion direction?** Read `spec_convert.md` first, then reference `spec_label.md` for handler interfaces.
- **Adding a visualization?** Read `spec_visualize.md` and `spec_label.md`.
- **Adding a CLI command?** Read `spec_cli.md` and the relevant module spec (convert or visualize).
