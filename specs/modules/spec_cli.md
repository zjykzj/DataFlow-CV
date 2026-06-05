# CLI Module Specification

> **Version:** 1.0
> **Layer:** Modules
> **Dependencies:** Convert module + Visualize module (public APIs only)

## 1. Module Overview

The CLI module (`dataflow/cli/`) provides a command-line interface built on Click. It is a **thin wrapper** over the Convert, Visualize, and Evaluate modules — CLI commands only call public APIs and never reach into internal module details.

### 1.1 Module Contract

- **CLI → Convert**: Calls `Converter.convert()` only. Does NOT import label handlers.
- **CLI → Visualize**: Calls `Visualizer.visualize()` only. Does NOT import label handlers.
- **CLI → Evaluate**: Calls `Evaluator.evaluate()` only. Does NOT import label handlers or pycocotools directly.
- **CLI responsibility**: Parameter parsing, validation, logging configuration, error formatting. Zero business logic.

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
- Default logger configured via `LoggingOperations`

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
    ├── detection     GT_JSON DT_JSON [--verbose] [--prf1] [--prf1-iou FLOAT] [--prf1-conf FLOAT] [--output PATH]
    └── segmentation  GT_JSON DT_JSON [--verbose] [--prf1] [--prf1-iou FLOAT] [--prf1-conf FLOAT] [--output PATH]
```

## 4. Convert Subcommands

### 4.1 Shared Decorator: `@add_common_options`

Adds to every convert subcommand:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--verbose` | Flag | False | Enable verbose log output (file logging to `logs/`) |
| `--no-strict` | Flag | False | Disable strict mode (skip invalid annotations) |

**Behavior on apply:**
1. Sets `ctx.obj["verbose"]` and `ctx.obj["strict"]` (= `not no_strict`)
2. Reconfigures logger: verbose → `VerboseLoggingOperations.get_verbose_logger()`, else → `LoggingOperations.get_logger()`
3. Stores `log_file_path` in `ctx.obj`

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
1. Extract ctx.obj (logger, verbose, strict)
2. Log start message
3. Call validate_convert_params() — format-specific checks
4. Instantiate converter class
5. Call converter.convert()
6. If verbose: log file path
7. If success: log summary
8. If failure: raise RuntimeCLIError with first error message
```

## 5. Visualize Subcommands

### 5.1 Shared Decorator: `@add_visualize_options`

Adds to every visualize subcommand:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--verbose` | Flag | False | Enable verbose log output |
| `--display`/`--no-display` | Flag | `--display` | Show/hide visualization window |

**Behavior on apply:**
1. Sets `ctx.obj["verbose"]`, `ctx.obj["is_show"]` (= display), `ctx.obj["strict"]` (= True, always)
2. Reconfigures logger same as `add_common_options`
3. Visualize commands do NOT support `--no-strict` (strict is always True)

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
1. Extract ctx.obj (logger, verbose, strict, is_show, log_file_path)
2. Log start message
3. Call validate_visualize_params() — checks paths exist, creates output dir
4. Instantiate visualizer class (with logger + log_file_path from context)
5. Call visualizer.visualize()
6. If verbose: log file path
7. If success: log processed count
8. If failure: raise RuntimeCLIError with error message
```

## 6. Evaluate Subcommands

### 6.1 Shared Options

Evaluate subcommands share these options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--verbose` | Flag | False | Enable verbose output: per-class metrics table + file logging |
| `--prf1` | Flag | False | Additionally compute and display P/R/F1 at specified IoU threshold |
| `--prf1-iou` | Float | 0.5 | IoU threshold for P/R/F1 calculation |
| `--prf1-conf` | Float | 0.0 | Confidence threshold for P/R/F1 calculation |
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

```
1. Extract ctx.obj (logger, verbose, strict, log_file_path)
2. Log start message with GT_JSON, DT_JSON paths
3. Validate: GT_JSON exists and is valid COCO JSON
4. Validate: DT_JSON exists and is valid COCO JSON
5. Validate: pycocotools is installed (raise SystemError if not)
6. Instantiate DetectionEvaluator or SegmentationEvaluator (with logger + verbose)
7. Call evaluator.evaluate(GT_JSON, DT_JSON) → EvaluationResult
8. If success:
   a. Print 12 COCO standard metrics table
   b. If verbose: print per-class breakdown table
   c. If --prf1: call compute_pr_f1() and print results
   d. If --output: write EvaluationResult as JSON to file
   e. Log file path (if verbose)
9. If failure: raise RuntimeCLIError with first error message
```

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

**PRF1 output (with `--prf1`):**

```
Precision / Recall / F1-Score (IoU=0.50, Conf=0.00):
  Overall:  P=0.756  R=0.912  F1=0.826  TP=1250  FP=403  FN=120
```

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

### 6.1 Usage in Commands

- `validate_convert_params()` raises `InputError` for missing/invalid inputs
- `validate_path_exists()` raises `InputError` for non-existent paths
- Post-conversion/visualization failures raise `RuntimeCLIError`
- `ParameterError`, `OutputError`, `SystemError` are defined but used sparingly

## 8. Parameter Validation

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
├── dataflow.util.logging_util           (LoggingOperations, VerboseLoggingOperations)
├── dataflow.cli.exceptions              (InputError, RuntimeCLIError, SystemError)
└── click                                (Framework)

CLI module does NOT import FROM:
├── dataflow.label.*                     (FORBIDDEN — must go through Convert/Visualize/Evaluate)
├── dataflow.convert.rle_converter       (FORBIDDEN — internal to Convert)
├── dataflow.convert.utils               (FORBIDDEN — internal to Convert)
├── dataflow.visualize.base              (FORBIDDEN — internal to Visualize; only concrete classes)
├── dataflow.evaluate.base               (FORBIDDEN — internal to Evaluate; only concrete classes)
├── dataflow.evaluate.result             (FORBIDDEN — internal to Evaluate)
└── pycocotools                          (FORBIDDEN — Evaluate dependency; CLI must not import directly)
```

### 9.1 Validation

The CLI contract can be verified by checking that:
1. `dataflow/cli/commands/convert.py` only imports from `dataflow.convert.*` (not `dataflow.label.*`)
2. `dataflow/cli/commands/visualize.py` only imports from `dataflow.visualize.*` (not `dataflow.label.*`)
3. `dataflow/cli/commands/evaluate.py` only imports from `dataflow.evaluate.*` (not `dataflow.label.*` or `pycocotools`)
4. No CLI file imports cross-module internals (e.g., convert commands don't import from visualize or evaluate internals)

## 10. Verbose Logging Contract

When `--verbose` is specified on any command:
1. Logger is created via `VerboseLoggingOperations.get_verbose_logger()`
2. Log file is written to `logs/` directory with timestamped filename
3. Console output uses `DEFAULT_FORMAT` (timestamps)
4. File output includes filename and line numbers (DEBUG level)
5. `log_file_path` is stored in `ctx.obj` and passed to the converter/visualizer
6. On completion, the log file path is printed to the user

When `--verbose` is NOT specified:
1. Console-only logging via `LoggingOperations.get_logger()` (INFO level)
2. No log files created
3. `log_file_path` is `None`

## 11. FormattedCommand

Custom `click.Command` subclass that formats Arguments in the help output to match the style of Options:

- Uses `formatter.write_dl()` for aligned argument display
- Maps argument parameter names to human-readable help text via `_get_argument_help()`
- Preserves standard Click formatting for usage, description, options, and epilog
