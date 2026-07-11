# CLI Module Specification

> **Version:** 2.1 | **Last Updated:** 2026-07-01
> **Layer:** Modules
> **Dependencies:** Convert module + Visualize module + Evaluate module (public APIs only) + Logging module (LogConfig only)

## 1. Module Contract

The CLI module (`dataflow/cli/`) is a **thin wrapper** over Convert, Visualize, and Evaluate. It owns zero business logic and zero log output.

- **CLI → Analyse**: Calls `StatsAnalyser.analyse()` / `SplitAnalyser.analyse()` only. Does NOT import label handlers.
- **CLI → Convert**: Calls `Converter.convert()` only. Does NOT import label handlers.
- **CLI → Visualize**: Calls `Visualizer.visualize()` only. Does NOT import label handlers.
- **CLI → Evaluate**: Calls `Evaluator.evaluate()` only. Does NOT import label handlers or `pycocotools` directly.
- **CLI responsibility**: Parameter parsing, validation, constructing `LogConfig`, error formatting, terminal UI via `click.echo()`.
- **Module responsibility**: All business logic and all log output.

## 2. Convert Subcommands

### 2.1 Shared Decorator: `@add_common_options`

Every convert subcommand receives:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--verbose` | Flag | False | Enable verbose log output (file logging to `logs/`) |
| `--no-strict` | Flag | False | Disable strict mode (skip invalid annotations) |
| `--log-dir` | Path | `./logs` | Log file output directory |

Sets `ctx.obj["verbose"]`, `ctx.obj["strict"]` (= `not no_strict`), `ctx.obj["log_dir"]`. Does NOT create a logger — modules handle logging internally via `LogConfig`.

### 2.2 Command Signatures

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

## 3. Visualize Subcommands

### 3.1 Shared Decorator: `@add_visualize_options`

Every visualize subcommand receives:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--verbose` | Flag | False | Enable verbose log output |
| `--display`/`--no-display` | Flag | `--display` | Show/hide visualization window |
| `--log-dir` | Path | `./logs` | Log file output directory |

Sets `ctx.obj["verbose"]`, `ctx.obj["is_show"]` (= display), `ctx.obj["log_dir"]`. Does NOT create a logger.

### 3.2 Command Signatures

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

## 4. Evaluate Subcommands

### 4.1 Shared Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--verbose` | Flag | False | Enable verbose output: per-class metrics table + file logging |
| `--log-dir` | Path | `./logs` | Log file output directory (only when `--verbose` is set) |
| `--prf1` | Flag | False | Compute P/R/F1 instead of mAP. Mutually exclusive with the mAP path. |
| `--prf1-iou` | Float | 0.5 | IoU threshold for P/R/F1 calculation |
| `--prf1-conf` | Float | 0.0 | Confidence threshold for P/R/F1 calculation |
| `--prf1-method` | Choice | `"macro"` | Aggregation: `"macro"` or `"micro"` |
| `--output`, `-o` | Path | None | Save full `EvaluationResult` as JSON to this path |

**Design rule**: `--prf1` and the mAP path are mutually exclusive. When `--prf1` is set, COCOeval is skipped entirely — only single-threshold P/R/F1 is computed via manual greedy matching. If both metrics are needed, run the command twice.

### 4.2 Command Signatures

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

### 4.3 Output

**Default (mAP)** — 12 COCO standard metrics table. With `--verbose`, an additional per-class breakdown table.

**`--prf1` mode** — P/R/F1 per-class table + overall summary. mAP is NOT computed in this mode.

**`--output` mode** — Full `EvaluationResult` serialized as JSON, containing: 12 metrics, per-class data (class_id, class_name, gt/dt/tp/fp/fn counts, AP/AP50/AP75, P/R/F1), GT/DT stats, warnings, errors. Intended for programmatic consumption (CI pipelines, experiment tracking).

## 5. Analyse Subcommands

### 5.1 Shared Options

Both ``stats`` and ``split`` subcommands use the `@add_common_options` decorator (§2.1) which provides `--verbose`, `--no-strict`, and `--log-dir`.  (The ``--no-strict`` flag is accepted but has no effect — analyse operations always run in non-strict mode.)

Both subcommands accept `--class-file` and `--image-dir`. The `_add_analyse_options` decorator (defined in ``commands/analyse.py``) provides these options and is used by the ``stats`` subcommand, while the ``split`` subcommand defines them inline (both as ``@click.option`` with identical signatures).

The ``stats`` subcommand additionally accepts `--sort-by` (`id`|`count`) and `--descending/--ascending` to control per-class output ordering when `--class-file` is not provided.

### 5.2 Command Signatures

#### `stats`

```
dataflow-cv analyse stats [OPTIONS] LABEL_PATH
```

Compute dataset statistics. Auto-detects the annotation format from `LABEL_PATH`.

| Argument | Type | Description |
|----------|------|-------------|
| `LABEL_PATH` | Path (exists) | Path to labels — directory (YOLO/LabelMe) or JSON file (COCO) |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--class-file`, `-c` | Path | None | Classes.txt for class name mapping and output ordering |
| `--image-dir` | Path | None | Image directory for YOLO format (auto-detected if omitted) |
| `--sort-by` | Choice | `"id"` | Sort key: `"id"` (class_id) or `"count"`. Ignored when `--class-file` is provided. |
| `--descending/--ascending` | Flag | ascending | Sort direction (default: ascending) |

#### `split`

```
dataflow-cv analyse split [OPTIONS] LABEL_PATH OUTPUT_DIR
```

Split dataset into train/val subsets with deterministic shuffling.

| Argument | Type | Description |
|----------|------|-------------|
| `LABEL_PATH` | Path (exists) | Path to labels — directory (YOLO/LabelMe) or JSON file (COCO) |
| `OUTPUT_DIR` | Path | Output root directory (``train/`` and ``val/`` created inside) |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--ratio`, `-r` | Float | 0.8 | Train proportion |
| `--seed`, `-s` | Int | 42 | Random seed |
| `--class-file`, `-c` | Path | None | Classes.txt (auto-generated for YOLO if omitted, copied to output dirs) |
| `--image-dir` | Path | None | Image directory for YOLO format (auto-detected if omitted) |

### 5.3 Output

**`stats`** — Summary block (total files, total annotations, category count) + per-class count table. Output ordering: class-file order if `--class-file` is provided, otherwise controlled by `--sort-by` + `--descending/--ascending` (default: class_id ascending).

**`split`** — Split summary block (train/val counts, output directories). Creates ``OUTPUT_DIR/train/`` and ``OUTPUT_DIR/val/``. For COCO, produces ``train.json`` and ``val.json``. For YOLO/LabelMe, produces per-file output via ``write_one()``.

## 6. Exception Hierarchy

All CLI exceptions extend `click.ClickException` with specific exit codes:

```
CLIError (base, exit_code configurable)
├── ParameterError  (exit 1)  — Invalid/missing command-line parameters
├── InputError      (exit 2)  — Input file/directory does not exist
├── OutputError     (exit 3)  — Cannot create/write output
├── RuntimeCLIError (exit 4)  — API execution failed (converter, visualizer, or evaluator)
└── SystemError     (exit 5)  — System-level failure (disk full, pycocotools not installed, etc.)
```

**Usage:**
- `validate_convert_params()` / `validate_evaluate_params()` raise `InputError` for missing/invalid inputs
- `validate_path_exists()` raises `InputError` for non-existent paths
- Post-conversion/visualization/evaluation failures raise `RuntimeCLIError`
- `SystemError` is used when pycocotools is not installed for evaluate commands
- `ParameterError`, `OutputError` are defined but used sparingly

## 7. Validators

### 7.1 `validate_convert_params`

Required parameter matrix:

| Direction | image_dir | class_file |
|-----------|-----------|------------|
| YOLO → COCO | **Required** | **Required** |
| YOLO → LabelMe | **Required** | **Required** |
| COCO → YOLO | Optional | Optional |
| COCO → LabelMe | Optional | Optional |
| LabelMe → YOLO | N/A | **Required** |
| LabelMe → COCO | N/A | **Required** |

Missing required parameters raise `InputError`.

### 7.2 `validate_visualize_params`

- Validates `input_path` exists
- Validates `image_dir` exists (if provided)
- Creates `output_dir` if it doesn't exist (if provided)

## 8. Dependency Contract

```
CLI module imports FROM:
├── dataflow.analyse                (StatsAnalyser, SplitAnalyser)
├── dataflow.convert                (YoloAndCocoConverter, LabelMeAndYoloConverter, CocoAndLabelMeConverter)
├── dataflow.visualize              (YOLOVisualizer, COCOVisualizer, LabelMeVisualizer)
├── dataflow.evaluate               (DetectionEvaluator, SegmentationEvaluator, compute_pr_f1)
├── dataflow.evaluate.utils         (format_prf1_output, format_metric_table, format_per_class_table)
├── dataflow.util.logging           (LogConfig only)
├── dataflow.cli.exceptions         (InputError, RuntimeCLIError, SystemError)
└── click                           (Framework)

CLI module does NOT import FROM:
├── dataflow.label.*                (FORBIDDEN — must go through Analyse/Convert/Visualize/Evaluate)
├── dataflow.analyse.base           (FORBIDDEN — internal to Analyse)
├── dataflow.analyse.utils          (FORBIDDEN — internal to Analyse)
├── dataflow.convert.rle_converter  (FORBIDDEN — internal to Convert)
├── dataflow.convert.utils          (FORBIDDEN — internal to Convert)
├── dataflow.visualize.base         (FORBIDDEN — internal to Visualize)
├── dataflow.evaluate.base          (FORBIDDEN — internal to Evaluate)
├── dataflow.evaluate.result        (FORBIDDEN — internal to Evaluate)
└── pycocotools                     (FORBIDDEN — Evaluate dependency)
```

**Compliance verification:**
1. No CLI file imports from `dataflow.label.*`
2. No CLI file imports from `pycocotools`
3. No CLI file imports cross-module internals (e.g., convert commands don't import from visualize internals)

## 9. Logging Contract

Logging is **module-owned**. CLI does not write log messages.

**CLI responsibilities:**
1. Parse `--verbose` and `--log-dir` from command-line options
2. Construct `LogConfig(name=f"dataflow.{command_name}", verbose=verbose, log_dir=log_dir)`
3. Pass `log_config` to the module constructor (Converter / Visualizer / Evaluator)
4. After operation completes, print final summary and log path via `click.echo()`

**Module responsibilities:**
1. Create `LogManager` from the provided `LogConfig`
2. Handle ALL log output — start, progress, phases, results, errors, warnings
3. Record `log_path` in the result object for CLI to report

**Output destinations:**
- **Console**: Always active (INFO level). Module writes structured log messages.
- **File**: Active only when `verbose=True` (DEBUG level). Path: `{log_dir}/log_{timestamp}.log`.
- **Terminal UI**: `click.echo()` from CLI — only final summary and log file path.
