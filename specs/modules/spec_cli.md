# CLI Module Specification

> **Version:** 1.0
> **Layer:** Modules
> **Dependencies:** Convert module + Visualize module (public APIs only)

## 1. Module Overview

The CLI module (`dataflow/cli/`) provides a command-line interface built on Click. It is a **thin wrapper** over the Convert and Visualize modules — CLI commands only call public APIs and never reach into internal module details.

### 1.1 Module Contract

- **CLI → Convert**: Calls `Converter.convert()` only. Does NOT import label handlers.
- **CLI → Visualize**: Calls `Visualizer.visualize()` only. Does NOT import label handlers.
- **CLI responsibility**: Parameter parsing, validation, logging configuration, error formatting. Zero business logic.

### 1.2 File Map

```
dataflow/cli/
├── main.py                  # Entry point, root CLI group
├── commands/
│   ├── __init__.py
│   ├── convert.py           # Convert subcommand group (6 commands)
│   ├── visualize.py         # Visualize subcommand group (3 commands)
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
└── visualize
    ├── yolo     IMAGE_DIR LABEL_DIR CLASS_FILE [--save DIR] [--verbose] [--display/--no-display]
    ├── labelme  IMAGE_DIR LABEL_DIR [--save DIR] [--verbose] [--display/--no-display]
    └── coco     IMAGE_DIR COCO_FILE [--save DIR] [--verbose] [--display/--no-display]
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
| `LABEL_DIR` | Path (exists) | YOLO label directory |
| `CLASS_FILE` | Path (exists) | Class file (`classes.txt`) |
| `OUTPUT_FILE` | Path | Output COCO JSON file |

| Option | Description |
|--------|-------------|
| `--do-rle` | Use RLE encoding for COCO segmentation |

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

## 6. Exception Hierarchy

All CLI exceptions extend `click.ClickException` with specific exit codes:

```
CLIError (base, exit_code configurable)
├── ParameterError  (exit 1)  — Invalid/missing command-line parameters
├── InputError      (exit 2)  — Input file/directory does not exist
├── OutputError     (exit 3)  — Cannot create/write output
├── RuntimeCLIError (exit 4)  — API execution failed (converter or visualizer)
└── SystemError     (exit 5)  — System-level failure (disk full, etc.)
```

### 6.1 Usage in Commands

- `validate_convert_params()` raises `InputError` for missing/invalid inputs
- `validate_path_exists()` raises `InputError` for non-existent paths
- Post-conversion/visualization failures raise `RuntimeCLIError`
- `ParameterError`, `OutputError`, `SystemError` are defined but used sparingly

## 7. Parameter Validation

### 7.1 `validate_convert_params(source_format, target_format, input_path, output_path, image_dir, class_file)`

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

### 7.2 `validate_visualize_params(input_path, image_dir, output_dir)`

- Validates `input_path` exists
- Validates `image_dir` exists (if provided)
- Creates `output_dir` if it doesn't exist (if provided)

## 8. Dependency Contract

```
CLI module imports FROM:
├── dataflow.convert.yolo_and_coco       (YoloAndCocoConverter)
├── dataflow.convert.labelme_and_yolo    (LabelMeAndYoloConverter)
├── dataflow.convert.coco_and_labelme    (CocoAndLabelMeConverter)
├── dataflow.visualize.yolo_visualizer   (YOLOVisualizer)
├── dataflow.visualize.coco_visualizer   (COCOVisualizer)
├── dataflow.visualize.labelme_visualizer (LabelMeVisualizer)
├── dataflow.util.logging_util           (LoggingOperations, VerboseLoggingOperations)
├── dataflow.cli.exceptions              (InputError, RuntimeCLIError, etc.)
└── click                                (Framework)

CLI module does NOT import FROM:
├── dataflow.label.*                     (FORBIDDEN — must go through Convert/Visualize)
├── dataflow.convert.rle_converter       (FORBIDDEN — internal to Convert)
├── dataflow.convert.utils               (FORBIDDEN — internal to Convert)
└── dataflow.visualize.base              (FORBIDDEN — internal to Visualize; only concrete classes)
```

### 8.1 Validation

The CLI contract can be verified by checking that:
1. `dataflow/cli/commands/convert.py` only imports from `dataflow.convert.*` (not `dataflow.label.*`)
2. `dataflow/cli/commands/visualize.py` only imports from `dataflow.visualize.*` (not `dataflow.label.*`)
3. No CLI file imports from both `dataflow.convert.*` and `dataflow.visualize.*` simultaneously (they don't need to know about each other)

## 9. Verbose Logging Contract

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

## 10. FormattedCommand

Custom `click.Command` subclass that formats Arguments in the help output to match the style of Options:

- Uses `formatter.write_dl()` for aligned argument display
- Maps argument parameter names to human-readable help text via `_get_argument_help()`
- Preserves standard Click formatting for usage, description, options, and epilog
