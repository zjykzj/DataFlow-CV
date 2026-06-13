# CLI Module Specification

> **Version:** 2.0
> **Layer:** Modules
> **Dependencies:** Convert module + Visualize module + Evaluate module (public APIs only) + Logging module (LogConfig only)

## 1. Module Overview

The CLI module (`dataflow/cli/`) provides a command-line interface built on Click. It is a **thin wrapper** over the Convert, Visualize, and Evaluate modules — CLI commands only call public APIs and never reach into internal module details.

### 1.1 Module Contract

- **CLI → Convert**: Calls `Converter.convert()` only. Does NOT import label handlers.
- **CLI → Visualize**: Calls `Visualizer.visualize()` only. Does NOT import label handlers.
- **CLI → Evaluate**: Calls `Evaluator.evaluate()` only. Does NOT import label handlers or pycocotools directly.
- **CLI responsibility**: Parameter parsing, validation, constructing `LogConfig`, error formatting, terminal UI via `click.echo()`. Zero business logic. Zero log output — all logging is owned by modules.

### 1.2 File Map

```
dataflow/cli/
├── main.py                  # Entry point, root CLI group
├── commands/
│   ├── __init__.py
│   ├── convert.py           # Convert subcommand group (6 commands)
│   ├── visualize.py         # Visualize subcommand group (3 commands)
│   ├── evaluate.py          # Evaluate subcommand group (2 commands)
│   ├── utils.py             # Shared decorators, validators, FormattedCommand
│   └── exceptions.py        # Exception hierarchy with exit codes
```

## 2. Entry Point

### 2.1 Root Command

```
dataflow-cv [OPTIONS] COMMAND [ARGS]...
```

**Global options:**

| Option | Type | Description |
|--------|------|-------------|
| `--version`, `-v` | Flag | Display version and exit |
| `--help`, `-h` | Flag | Show help |

**Context initialization (`cli` group callback):**
- `ctx.obj["verbose"]` = `False`
- `ctx.obj["log_dir"]` = `Path("./logs")`
- `ctx.obj["strict"]` = `True`
- No logger is created — logging is owned by modules

### 2.2 Help Configuration

- Help option names: `-h`, `--help`
- Max content width: 100 characters
- Default values are shown in help text

## 3. Command Structure

```
dataflow-cv
├── convert
│   ├── yolo2coco     IMAGE_DIR LABEL_DIR CLASS_FILE OUTPUT_FILE [--do-rle] [--verbose] [--no-strict]
│   ├── yolo2labelme  IMAGE_DIR LABEL_DIR CLASS_FILE OUTPUT_DIR [--verbose] [--no-strict]
│   ├── labelme2yolo  LABELME_DIR CLASS_FILE OUTPUT_DIR [--verbose] [--no-strict]
│   ├── labelme2coco  LABELME_DIR CLASS_FILE OUTPUT_FILE [--do-rle] [--verbose] [--no-strict]
│   ├── coco2yolo     COCO_FILE OUTPUT_DIR [--verbose] [--no-strict]
│   └── coco2labelme  COCO_FILE OUTPUT_DIR [--verbose] [--no-strict]
│
├── visualize
│   ├── yolo     IMAGE_DIR LABEL_DIR CLASS_FILE [--save DIR] [--verbose] [--display/--no-display]
│   ├── labelme  IMAGE_DIR LABEL_DIR [--save DIR] [--verbose] [--display/--no-display]
│   └── coco     IMAGE_DIR COCO_FILE [--save DIR] [--verbose] [--display/--no-display]
│
└── evaluate
    ├── detection     GT_JSON DT_JSON [--verbose] [--prf1] [--prf1-iou FLOAT] [--prf1-conf FLOAT] [--prf1-method STR] [--output PATH]
    └── segmentation  GT_JSON DT_JSON [--verbose] [--prf1] [--prf1-iou FLOAT] [--prf1-conf FLOAT] [--prf1-method STR] [--output PATH]
```

## 4. Convert Subcommands

### 4.1 Shared Decorator: `@add_common_options`

Adds to every convert subcommand:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--verbose` | Flag | False | Enable verbose log output (file logging to `logs/`) |
| `--no-strict` | Flag | False | Disable strict mode (skip invalid annotations) |
| `--log-dir` | Path | `./logs` | Log file output directory |

**Behavior on apply:**
1. Sets `ctx.obj["verbose"]` and `ctx.obj["strict"]` (= `not no_strict`) and `ctx.obj["log_dir"]`
2. Does NOT create a logger — modules handle logging internally via `LogConfig`

### 4.2 Command Signatures

#### `yolo2coco`

```
dataflow-cv convert yolo2coco [OPTIONS] IMAGE_DIR LABEL_DIR CLASS_FILE OUTPUT_FILE
```

| Argument | Type | Description |
|----------|------|-------------|
| `IMAGE_DIR` | Path (exists) | Image file directory |
| `LABEL_DIR` | Path (exists) | YOLO label/prediction directory |
| `CLASS_FILE` | Path (exists) | Class file (`classes.txt`) |
| `OUTPUT_FILE` | Path | Output COCO JSON file |

| Option | Description |
|--------|-------------|
| `--do-rle` | Use RLE encoding for COCO segmentation |
| `--prediction` | Treat input as prediction format (with confidence). Output includes `score` fields. |

**Validates:** `class_file` and `image_dir` are required. `pycocotools` must be installed if `--do-rle` is set.

#### `yolo2labelme`

```
dataflow-cv convert yolo2labelme [OPTIONS] IMAGE_DIR LABEL_DIR CLASS_FILE OUTPUT_DIR
```

| Argument | Type | Description |
|----------|------|-------------|
| `IMAGE_DIR` | Path (exists) | Image file directory |
| `LABEL_DIR` | Path (exists) | YOLO label directory |
| `CLASS_FILE` | Path (exists) | Class file |
| `OUTPUT_DIR` | Path | Output LabelMe directory |

#### `coco2yolo`

```
dataflow-cv convert coco2yolo [OPTIONS] COCO_FILE OUTPUT_DIR
```

| Argument | Type | Description |
|----------|------|-------------|
| `COCO_FILE` | Path (exists) | Input COCO JSON file |
| `OUTPUT_DIR` | Path | Output directory (creates `labels/` + `images/` + `classes.txt`) |

#### `coco2labelme`

```
dataflow-cv convert coco2labelme [OPTIONS] COCO_FILE OUTPUT_DIR
```

| Argument | Type | Description |
|----------|------|-------------|
| `COCO_FILE` | Path (exists) | Input COCO JSON file |
| `OUTPUT_DIR` | Path | Output LabelMe directory |

#### `labelme2yolo`

```
dataflow-cv convert labelme2yolo [OPTIONS] LABELME_DIR CLASS_FILE OUTPUT_DIR
```

| Argument | Type | Description |
|----------|------|-------------|
| `LABELME_DIR` | Path (exists) | LabelMe annotation directory |
| `CLASS_FILE` | Path (exists) | Class file |
| `OUTPUT_DIR` | Path | Output directory |

#### `labelme2coco`

```
dataflow-cv convert labelme2coco [OPTIONS] LABELME_DIR CLASS_FILE OUTPUT_FILE
```

| Argument | Type | Description |
|----------|------|-------------|
| `LABELME_DIR` | Path (exists) | LabelMe annotation directory |
| `CLASS_FILE` | Path (exists) | Class file |
| `OUTPUT_FILE` | Path | Output COCO JSON file |

| Option | Description |
|--------|-------------|
| `--do-rle` | Use RLE encoding for COCO segmentation |

### 4.3 Convert Command Flow (All 6 Commands)

```
1. Extract ctx.obj (verbose, strict, log_dir)
2. Construct LogConfig(name=f"dataflow.{cmd}", verbose=verbose, log_dir=log_dir)
3. Call validate_convert_params() — format-specific checks
4. Instantiate converter class with log_config=log_config, strict_mode=strict
5. Call converter.convert() → ConversionResult
   (converter handles ALL logging internally)
6. If success: click.echo(f"✓ {result.get_summary()}")
7. If verbose and result.log_path: click.echo(f"Log saved to: {result.log_path}")
8. If failure: raise RuntimeCLIError with first error message
```

## 5. Visualize Subcommands

### 5.1 Shared Decorator: `@add_visualize_options`

Adds to every visualize subcommand:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--verbose` | Flag | False | Enable verbose log output |
| `--display`/`--no-display` | Flag | `--display` | Show/hide visualization window |
| `--log-dir` | Path | `./logs` | Log file output directory |

**Behavior on apply:**
1. Sets `ctx.obj["verbose"]`, `ctx.obj["is_show"]` (= display), `ctx.obj["log_dir"]`
2. Does NOT create a logger — visualizer handles logging internally via `LogConfig`

### 5.2 Command Signatures

#### `yolo`

```
dataflow-cv visualize yolo [OPTIONS] IMAGE_DIR LABEL_DIR CLASS_FILE
```

| Argument | Type | Description |
|----------|------|-------------|
| `IMAGE_DIR` | Path (exists) | Image directory |
| `LABEL_DIR` | Path (exists) | YOLO label directory |
| `CLASS_FILE` | Path (exists) | Class file |

| Option | Description |
|--------|-------------|
| `--save`, `-s` PATH | Save rendered images to directory |

#### `coco`

```
dataflow-cv visualize coco [OPTIONS] IMAGE_DIR COCO_FILE
```

| Argument | Type | Description |
|----------|------|-------------|
| `IMAGE_DIR` | Path (exists) | Image directory |
| `COCO_FILE` | Path (exists) | COCO JSON annotation file |

| Option | Description |
|--------|-------------|
| `--save`, `-s` PATH | Save rendered images to directory |

#### `labelme`

```
dataflow-cv visualize labelme [OPTIONS] IMAGE_DIR LABEL_DIR
```

| Argument | Type | Description |
|----------|------|-------------|
| `IMAGE_DIR` | Path (exists) | Image directory |
| `LABEL_DIR` | Path (exists) | LabelMe annotation directory |

| Option | Description |
|--------|-------------|
| `--save`, `-s` PATH | Save rendered images to directory |

### 5.3 Visualize Command Flow (All 3 Commands)

```
1. Extract ctx.obj (verbose, is_show, log_dir)
2. Construct LogConfig(name=f"dataflow.{cmd}", verbose=verbose, log_dir=log_dir)
3. Call validate_visualize_params() — checks paths exist, creates output dir
4. Instantiate visualizer class with log_config=log_config
5. Call visualizer.visualize() → VisualizationResult
   (visualizer handles ALL logging internally)
6. If success: click.echo(f"✓ processed {result.data.get('processed_count', 0)} images")
7. If verbose and result.log_path: click.echo(f"Log saved to: {result.log_path}")
8. If failure: raise RuntimeCLIError with error message
```

## 6. Evaluate Subcommands

### 6.1 Shared Options

Evaluate subcommands share these options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--verbose` | Flag | False | Enable verbose output: per-class metrics table + file logging |
| `--prf1` | Flag | False | Compute P/R/F1 instead of mAP. When set, COCOeval is skipped entirely — only single-threshold P/R/F1 is computed via manual greedy matching. Mutually exclusive with the mAP path (no flag). |
| `--prf1-iou` | Float | 0.5 | IoU threshold for P/R/F1 calculation |
| `--prf1-conf` | Float | 0.0 | Confidence threshold for P/R/F1 calculation |
| `--prf1-method` | Choice | "macro" | Aggregation method for overall P/R/F1: ``"macro"`` or ``"micro"``. See `spec_evaluate_metrics.md` §2.3 |
| `--output`, `-o` | Path | None | Save full `EvaluationResult` as JSON to this path |

### 6.2 Command Signatures

#### `detection`

```
dataflow-cv evaluate detection [OPTIONS] GT_JSON DT_JSON
```

Evaluates object detection results using bbox IoU (`iouType='bbox'`).

| Argument | Type | Description |
|----------|------|-------------|
| `GT_JSON` | Path (exists) | COCO format Ground Truth JSON file |
| `DT_JSON` | Path (exists) | COCO format Detection/Prediction JSON file (annotations must include `score`) |

#### `segmentation`

```
dataflow-cv evaluate segmentation [OPTIONS] GT_JSON DT_JSON
```

Evaluates instance segmentation results using mask IoU (`iouType='segm'`).

| Argument | Type | Description |
|----------|------|-------------|
| `GT_JSON` | Path (exists) | COCO format Ground Truth JSON file (annotations must include `segmentation`) |
| `DT_JSON` | Path (exists) | COCO format Prediction JSON file (annotations must include `segmentation` + `score`) |

### 6.3 Evaluate Command Flow (Both Commands)

Two mutually exclusive paths, selected by `--prf1`:

**Path A — mAP (default, no `--prf1`):**

```
1. Extract ctx.obj (verbose, log_dir)
2. Construct LogConfig(name=f"dataflow.{cmd}", verbose=verbose, log_dir=log_dir)
3. Validate: GT_JSON, DT_JSON exist; pycocotools installed
4. Instantiate DetectionEvaluator or SegmentationEvaluator (with log_config=log_config)
5. Call evaluator.evaluate(GT_JSON, DT_JSON) → EvaluationResult
   (runs full COCOeval: 10 IoU × 101 recall thresholds)
6. If success:
   a. Print 12 COCO standard metrics table (click.echo)
   b. If verbose: print per-class breakdown table (click.echo)
   c. If --output: write EvaluationResult as JSON to file
   d. If result.log_path: click.echo(f"Log saved to: {result.log_path}")
7. If failure: raise RuntimeCLIError with first error message
```

**Path B — P/R/F1 (`--prf1`):**

```
1. Same validation steps (1-3)
2. Call compute_pr_f1(GT_JSON, DT_JSON, iou_threshold=prf1_iou,
     confidence_threshold=prf1_conf, iou_type=..., method=prf1_method)
   (single IoU threshold, manual greedy matching — COCOeval NOT invoked)
3. If success:
   a. Print P/R/F1 per-class table + overall summary (click.echo)
   b. If result.log_path: click.echo(f"Log saved to: {result.log_path}")
4. If failure: raise RuntimeCLIError
```

**Key design rule**: `--prf1` and the mAP path are mutually exclusive. If the user
wants both metrics, they run the command twice — once without `--prf1` for mAP,
once with `--prf1` for P/R/F1. This avoids forcing expensive COCOeval computation
on users who only need single-threshold P/R/F1.```

### 6.4 Output Format

**Default output (always printed):**

```
Evaluation: detection (bbox)
Ground Truth: 500 images, 3250 annotations, 10 categories
Detections:   500 images, 4100 detections, 10 categories

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.352
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.568
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.371
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.152
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.389
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.524
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.289
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.452
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.467
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.213
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.501
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.689
```

**Verbose output (with `--verbose`):**

Additional per-class table after the 12 metrics:

```
Per-Class Breakdown (IoU: 0.50:0.95):
───────────────────────────────────────────────────────────────────────────
 Class          GT    DT     TP    FP    FN     AP     AP50   AP75   P      R      F1
 person         520   610   487   123    33   0.432  0.689  0.451  0.798  0.937  0.862
 car            380   450   342   108    38   0.401  0.634  0.422  0.760  0.900  0.824
 bicycle        150   180   128    52    22   0.321  0.521  0.338  0.711  0.853  0.775
 ...
───────────────────────────────────────────────────────────────────────────
```

### 6.5 P/R/F1 Output (with `--prf1`)

When `--prf1` is specified, mAP is **not** computed. Only P/R/F1 is output.

Default (macro) mode:
```
Precision / Recall / F1-Score (IoU=0.50, Conf=0.00, Method=macro):
────────────────────────────────────────────────────────────────
Class             GT    TP    FP    FN       P       R      F1
────────────────────────────────────────────────────────────────
person            520   487   123    33  0.7980  0.9370  0.8620
car               380   342   108    38  0.7600  0.9000  0.8240
────────────────────────────────────────────────────────────────
  Overall:  P=0.756  R=0.912  F1=0.826  TP=1250  FP=403  FN=120
```

Micro mode (`--prf1-method micro`):
```
Precision / Recall / F1-Score (IoU=0.50, Conf=0.00, Method=micro):
  Overall:  P=0.756  R=0.912  F1=0.826  TP=1250  FP=403  FN=120
```

In macro mode, overall P/R are means of per-class values. In micro mode, overall
P/R are computed from summed TP/FP/FN totals. The TP/FP/FN shown are always the
summed totals. Per-class P/R/F1 values are identical in both modes.

### 6.5 JSON Output (`--output`)

When `--output PATH` is specified, the full `EvaluationResult` is serialized to JSON:

```json
{
  "success": true,
  "iou_type": "bbox",
  "metrics": {
    "ap": 0.352, "ap50": 0.568, "ap75": 0.371,
    "ap_small": 0.152, "ap_medium": 0.389, "ap_large": 0.524,
    "ar_max_1": 0.289, "ar_max_10": 0.452, "ar_max_100": 0.467,
    "ar_small": 0.213, "ar_medium": 0.501, "ar_large": 0.689
  },
  "per_class": {
    "1": {"class_id": 1, "class_name": "person", "gt_count": 520, "dt_count": 610, "tp": 487, "fp": 123, "fn": 33, "ap": 0.432, "ap50": 0.689, "ap75": 0.451, "precision": 0.798, "recall": 0.937, "f1_score": 0.862}
  },
  "gt_stats": {"images": 500, "annotations": 3250, "categories": 10},
  "dt_stats": {"images": 500, "annotations": 4100, "categories": 10},
  "warnings": [],
  "errors": []
}
```

The JSON output is intended for programmatic consumption (CI pipelines, experiment tracking, etc.).

## 7. Exception Hierarchy

All CLI exceptions extend `click.ClickException` with specific exit codes:

```
CLIError (base, exit_code configurable)
├── ParameterError  (exit 1)  — Invalid/missing command-line parameters
├── InputError      (exit 2)  — Input file/directory does not exist
├── OutputError     (exit 3)  — Cannot create/write output
├── RuntimeCLIError (exit 4)  — API execution failed (converter, visualizer, or evaluator)
└── SystemError     (exit 5)  — System-level failure (disk full, pycocotools not installed, etc.)
```

### 7.1 Usage in Commands

- `validate_convert_params()` raises `InputError` for missing/invalid inputs
- `validate_evaluate_params()` raises `InputError` for missing/invalid inputs
- `validate_path_exists()` raises `InputError` for non-existent paths
- Post-conversion/visualization/evaluation failures raise `RuntimeCLIError`
- `ParameterError`, `OutputError`, `SystemError` are defined but used sparingly
- `SystemError` is used when pycocotools is not installed for evaluate commands



### 8.1 `validate_convert_params(source_format, target_format, input_path, output_path, image_dir, class_file)`

Format-specific required parameter checks:

| Direction | image_dir | class_file |
|-----------|-----------|------------|
| YOLO → COCO | **Required** | **Required** |
| YOLO → LabelMe | **Required** | **Required** |
| COCO → YOLO | Optional | Optional |
| COCO → LabelMe | Optional | Optional |
| LabelMe → YOLO | N/A | **Required** |
| LabelMe → COCO | N/A | **Required** |

Missing required parameters raise `InputError`.

### 8.2 `validate_visualize_params(input_path, image_dir, output_dir)`

- Validates `input_path` exists
- Validates `image_dir` exists (if provided)
- Creates `output_dir` if it doesn't exist (if provided)

## 9. Dependency Contract

```
CLI module imports FROM:
├── dataflow.convert.yolo_and_coco       (YoloAndCocoConverter)
├── dataflow.convert.labelme_and_yolo    (LabelMeAndYoloConverter)
├── dataflow.convert.coco_and_labelme    (CocoAndLabelMeConverter)
├── dataflow.visualize.yolo_visualizer   (YOLOVisualizer)
├── dataflow.visualize.coco_visualizer   (COCOVisualizer)
├── dataflow.visualize.labelme_visualizer (LabelMeVisualizer)
├── dataflow.evaluate.evaluator          (DetectionEvaluator, SegmentationEvaluator)
├── dataflow.evaluate.metrics            (compute_pr_f1)
├── dataflow.util.logging                (LogConfig — for constructing config passed to modules)
├── dataflow.cli.exceptions              (InputError, RuntimeCLIError, SystemError)
└── click                                (Framework)

CLI module does NOT import FROM:
├── dataflow.label.*                     (FORBIDDEN — must go through Convert/Visualize/Evaluate)
├── dataflow.convert.rle_converter       (FORBIDDEN — internal to Convert)
├── dataflow.convert.utils               (FORBIDDEN — internal to Convert)
├── dataflow.visualize.base              (FORBIDDEN — internal to Visualize; only concrete classes)
├── dataflow.evaluate.base               (FORBIDDEN — internal to Evaluate; only concrete classes)
├── dataflow.evaluate.result             (FORBIDDEN — internal to Evaluate)
├── dataflow.util.logging_util           (REMOVED — no longer needed; LogConfig replaces both classes)
└── pycocotools                          (FORBIDDEN — Evaluate dependency; CLI must not import directly)
```

### 9.1 Validation

The CLI contract can be verified by checking that:
1. `dataflow/cli/commands/convert.py` only imports from `dataflow.convert.*` (not `dataflow.label.*`)
2. `dataflow/cli/commands/visualize.py` only imports from `dataflow.visualize.*` (not `dataflow.label.*`)
3. `dataflow/cli/commands/evaluate.py` only imports from `dataflow.evaluate.*` (not `dataflow.label.*` or `pycocotools`)
4. No CLI file imports cross-module internals (e.g., convert commands don't import from visualize or evaluate internals)
5. No CLI file imports `LoggingOperations` or `VerboseLoggingOperations` — only `LogConfig`

## 10. Logging Contract

Logging is **module-owned** — CLI does not write log messages. See [`spec_logging.md`](spec_logging.md) for the full contract.

**CLI responsibilities:**
1. Parse `--verbose` and `--log-dir` from command-line options
2. Construct a `LogConfig(name=f"dataflow.{command_name}", verbose=verbose, log_dir=log_dir)`
3. Pass `log_config` to the module constructor (Converter / Visualizer / Evaluator)
4. After the operation completes, if the result includes `log_path`, print it via `click.echo()`

**Module responsibilities:**
1. Create `LogManager` from the provided `LogConfig` (or a default)
2. Handle ALL log output — start, progress, phases, results, errors, warnings
3. Record `log_path` in the result object for CLI to report

**Output destinations:**
- **Console**: Always active (INFO level, compact format). Module writes structured log messages.
- **File**: Active only when `verbose=True` (DEBUG level, detailed format). Path: `{log_dir}/log_{timestamp}.log`.
- **Terminal UI**: `click.echo()` from CLI — only the final summary and log file path.

## 11. FormattedCommand

Custom `click.Command` subclass that formats Arguments in the help output to match the style of Options:

- Uses `formatter.write_dl()` for aligned argument display
- Maps argument parameter names to human-readable help text via `_get_argument_help()`
- Preserves standard Click formatting for usage, description, options, and epilog

## 12. Change History

### v2.0 → v2.1: --prf1 Computes P/R/F1 Only

| Aspect | v2.0 | v2.1 |
|--------|------|------|
| `--prf1` behavior | Additionally compute P/R/F1 on top of mAP | Compute P/R/F1 instead of mAP |
| mAP path | Always runs COCOeval (10×101 IoU/recall) | Only runs when `--prf1` NOT set |
| P/R/F1 path | Appended to mAP output | Standalone — skips COCOeval entirely |
| Both metrics | One command | Two separate commands (run twice) |

**Rationale**: mAP and P/R/F1 answer different questions. Forcing COCOeval on
users who only want single-threshold P/R/F1 was wasteful. The two paths are now
mutually exclusive — run the command twice if both metrics are needed.

### v1.1 → v2.0: Module-Owned Logging

| Aspect | v1.1 | v2.0 |
|--------|------|------|
| CLI logging | CLI creates `LoggingOperations`/`VerboseLoggingOperations` and writes log messages | CLI creates `LogConfig` only, passes to modules — zero log output |
| `add_common_options` | Creates logger, stores in `ctx.obj["logger"]` | Sets `verbose`/`strict`/`log_dir` in `ctx.obj`, no logger |
| `add_visualize_options` | Creates logger, stores in `ctx.obj` | Sets `verbose`/`is_show`/`log_dir` in `ctx.obj`, no logger |
| Command flow | Logs "Starting...", "Completed...", "Log saved to..." | `click.echo()` for terminal output only |
| Dependency imports | `LoggingOperations`, `VerboseLoggingOperations` | `LogConfig` only |
| Converter instantiation | `Converter(verbose=verbose)` — no logger passed | `Converter(log_config=log_config)` |
| Visualizer instantiation | `Visualizer(verbose=verbose, logger=logger, log_file_path=...)` | `Visualizer(log_config=log_config)` |
| New CLI option | — | `--log-dir` (default `./logs`) |
