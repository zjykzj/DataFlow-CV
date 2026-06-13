# Logging Module Specification

> **Version:** 1.0
> **Status:** Draft — unified logging infrastructure replacing `LoggingOperations` + `VerboseLoggingOperations`
> **Layer:** Modules
> **Dependencies:** None (foundation module, parallel to Label)

## 1. Module Overview

The Logging module (`dataflow/util/logging.py`) provides a **single unified logging infrastructure**
used by all other modules (Label, Convert, Visualize, Evaluate) and the CLI layer.

### 1.1 Design Principle: Module-Owned Logging

**Logging is entirely the responsibility of the processing modules, not the CLI.**

The CLI layer does NOT create loggers or write log messages. Instead:

1. CLI parses `--verbose` and `--log-dir` from the command line
2. CLI passes these values as parameters to module constructors
3. Modules create their own `LogManager` internally, which owns all log output
4. Modules log their entire lifecycle (start, progress, phases, results, errors)
5. CLI only uses `click.echo()` for terminal UI output

This ensures:

- **Python API users** get identical logging behavior without CLI involvement
- **No duplicate loggers** — one `LogManager` per module instance, one log file in verbose mode
- **No logger parameter threading** through constructor chains — `LogManager` is created once at the module entry point and propagated internally

### 1.2 Architecture Position

```
┌──────────────────────────────────────────────────────────────┐
│                           CLI                                 │
│  click.echo(result)  ← 终端 UI 输出                           │
│  传递 verbose + log_dir 参数                                  │
└──────┬─────────────────────┬──────────────────┬──────────────┘
       │                     │                  │
       ▼                     ▼                  ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│   Convert    │    │    Visualize     │    │   Evaluate   │
│  (pipeline)  │    │  (rendering)     │    │  (metrics)   │
│  LogManager  │    │  LogManager      │    │  LogManager  │
└──────┬───────┘    └───────┬──────────┘    └──────┬───────┘
       │                    │                      │
       │                    │                      │
       ▼                    ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│                         Label                                 │
│  Data Models + Handlers (receive logger from caller)          │
└──────────────────────────────────────────────────────────────┘
       │                    │                      │
       └────────────────────┼──────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                      util/logging.py                           │
│  LogManager + 格式化辅助函数                                    │
│  (所有模块的共享基础设施)                                       │
└──────────────────────────────────────────────────────────────┘
```

All modules (Label, Convert, Visualize, Evaluate) and CLI may import from `dataflow.util.logging`.
The `LogManager` is the single entry point — it replaces both `LoggingOperations` and `VerboseLoggingOperations`.

## 2. Core Classes

### 2.1 `LogConfig`

Immutable configuration object passed to `LogManager`:

```python
@dataclass(frozen=True)
class LogConfig:
    """Logging configuration.

    Attributes:
        name: Logger name, e.g. ``"convert.yolo_to_coco"``.
            Used as the Python logger name for hierarchical filtering.
        verbose: If True, enable file logging (RotatingFileHandler at DEBUG level).
            If False, console-only output at INFO level.
        log_dir: Directory for log files. Only used when ``verbose=True``.
            Default: ``Path("./logs")``.
    """
    name: str
    verbose: bool = False
    log_dir: Path = Path("./logs")
```

### 2.2 `LogManager`

Unified logging manager — the single replacement for `LoggingOperations` and `VerboseLoggingOperations`.

```python
class LogManager:
    """Unified logging manager.

    One ``LogManager`` per module instance.  Created at module entry point
    and propagated to child components via ``self.logger`` or ``.child()``.
    """

    def __init__(self, config: LogConfig) -> None: ...
```

#### 2.2.1 Properties

| Property | Type | Description |
|----------|------|-------------|
| `logger` | `logging.Logger` | The configured logger — use directly for all logging calls |
| `log_path` | `Optional[str]` | Log file path when `verbose=True`, else `None` |

#### 2.2.2 `child(suffix) → logging.Logger`

Create a child logger for sub-components (e.g., handlers, converters):

```python
def child(self, suffix: str) -> logging.Logger:
    """Create a child logger.

    Example:
        handler_logger = log_manager.child("handler")
        # → logger name "convert.yolo_to_coco.handler"
    """
```

Returns `self._logger.getChild(suffix)`.

#### 2.2.3 Handler Configuration

The logger is configured once in `__init__` with two handlers:

**Console handler** (always active):

| Attribute | Value |
|-----------|-------|
| Stream | `sys.stdout` |
| Level | `INFO` |
| Format | `"%(asctime)s  %(levelname)-7s  %(message)s"` |
| Date format | `"%H:%M:%S"` |

**File handler** (only when `verbose=True`):

| Attribute | Value |
|-----------|-------|
| Type | `RotatingFileHandler` |
| Max bytes | 10 MB |
| Backup count | 5 |
| Encoding | UTF-8 |
| Level | `DEBUG` |
| Format | `"%(asctime)s  %(levelname)-7s  %(name)s:%(lineno)d  %(message)s"` |
| Date format | `"%Y-%m-%d %H:%M:%S"` |
| File path | `{log_dir}/log_{timestamp}.log` |

**Key design decisions:**

- Console format is **compact** — time + level + message. No module name or line number in console (reduces noise). This is the INFO-level user-facing output.
- File format is **verbose** — includes module name and line number at DEBUG level. This is the developer-facing diagnostic output.
- Both handlers coexist when `verbose=True` — console shows INFO-level progress, file captures full DEBUG details.
- `RotatingFileHandler` prevents log files from growing unbounded (10 MB × 5 backups).

#### 2.2.4 Logger Behavior Contract

1. **Logger is reusable**: Multiple calls to create a `LogManager` with the same `config.name` return a logger with the same Python `logging.getLogger(name)`, but each `LogManager` instance adds its own handlers. Normal usage creates one `LogManager` per module instance, so this is not an issue.

2. **Logger is NOT a global singleton**: Each module creates its own `LogManager`. There is no cross-module logger sharing.

3. **Propagation is disabled**: The logger has `propagate = False` to prevent double-logging to the root logger.

4. **Existing handlers are cleared**: `logger.handlers.clear()` is called before adding new handlers to prevent duplicate output when the same Python logger name is reused.

### 2.3 Legacy Classes (Removed)

The following classes are **removed** in this version:

| Removed Class | Replacement |
|---------------|-------------|
| `LoggingOperations` | `LogManager` — single unified class |
| `VerboseLoggingOperations` | `LogManager` with `verbose=True` |
| `logging_error_or_raise()` | Each base class implements its own error handling with `self.logger` |
| `detect_image_error()` | Moved to `BaseVisualizer` (only consumer) or a small util function |

Rationale: Two classes with overlapping responsibility created confusion and led to
inconsistent behavior across modules. One `LogManager` with a `verbose` flag is simpler
and sufficient for all use cases.

## 3. Format Helpers

Module-level formatting functions for structured log output. These are pure functions —
no logger dependency, no side effects. Modules call them and pass the result to `self.logger.info()`.

### 3.1 General-Purpose Helpers

```python
def format_divider(char: str = "─", width: int = 60) -> str:
    """Return a divider line. ``"───...───"``"""

def format_section(title: str) -> str:
    """Return a section header. ``"── {title} ──"``"""

def format_kv(key: str, value: Any, indent: int = 2) -> str:
    """Return a key-value line. ``"  key: value"``"""

def format_result_block(
    status: str,
    items: Dict[str, Any],
    log_path: Optional[str] = None,
) -> str:
    """Return a result summary block with optional log path line."""
```

### 3.2 Table Helpers (for Evaluate)

```python
def format_table(
    headers: List[str],
    rows: List[List[str]],
    col_widths: Optional[List[int]] = None,
) -> str:
    """Render a formatted table with aligned columns and borders."""
```

---

## 4. Module-Specific Log Templates

Each processing module defines its own log templates in a `log_templates.py` file.
These templates are formatting functions that produce **strings** — the caller decides
the log level and passes the result to `self.logger.info()` / `.debug()` / etc.

### 4.1 Structure

```
dataflow/
├── convert/
│   └── log_templates.py      # Convert-specific log formatting
├── visualize/
│   └── log_templates.py      # Visualize-specific log formatting
├── evaluate/
│   └── log_templates.py      # Evaluate-specific log formatting
└── util/
    └── logging.py             # Shared LogManager + general-purpose helpers
```

### 4.2 Convert Log Templates (`dataflow/convert/log_templates.py`)

```python
def format_convert_header(
    source_format: str,
    target_format: str,
    source_path: str,
    target_path: str,
    mode: str,           # "label" or "prediction"
    strict: bool,
    options: Dict[str, Any],  # e.g. {"do_rle": True}
) -> str:
    """Header block shown at conversion start."""

def format_convert_phase(
    phase: str,          # "Read", "Convert", "Write"
    stats: Dict[str, Any],
) -> str:
    """Phase marker with statistics."""

def format_convert_result(result: "ConversionResult") -> str:
    """Final result block with status, counts, duration, warnings."""
```

### 4.3 Visualize Log Templates (`dataflow/visualize/log_templates.py`)

```python
def format_viz_header(
    format_name: str,
    label_dir: str,
    image_dir: str,
    is_show: bool,
    is_save: bool,
    output_dir: Optional[str],
) -> str:
    """Header block shown at visualization start."""

def format_viz_progress(
    index: int,
    image_name: str,
    n_objects: int,
    status: str,         # "✓" or "✗"
) -> str:
    """Single-line progress for streaming output."""

def format_viz_result(stats: Dict[str, Any]) -> str:
    """Final result block with counts and duration."""
```

### 4.4 Evaluate Log Templates (`dataflow/evaluate/log_templates.py`)

```python
def format_eval_header(
    iou_type: str,       # "bbox" or "segm"
    gt_stats: Dict[str, int],
    dt_stats: Dict[str, int],
) -> str:
    """Header block shown at evaluation start."""

def format_metric_table(metrics: "EvaluationMetrics") -> str:
    """12 COCO standard metrics as a formatted table."""

def format_per_class_table(per_class: Dict[int, "PerClassMetrics"]) -> str:
    """Per-class breakdown as a formatted table."""

def format_prf1_output(result: "PRF1Result") -> str:
    """P/R/F1 results as a formatted block."""

def format_eval_result(status: str, duration_sec: float, log_path: Optional[str]) -> str:
    """Final result block."""
```

---

## 5. Module Integration Contract

### 5.1 Constructor Pattern

Every module base class follows this pattern:

```python
class BaseModule:
    def __init__(
        self,
        ...
        log_config: Optional[LogConfig] = None,  # NEW — unified entry point
    ):
        if log_config is None:
            log_config = LogConfig(name=self._default_log_name())
        self._log_manager = LogManager(log_config)
        self.logger = self._log_manager.logger
        self._log_path = self._log_manager.log_path
```

Where `_default_log_name()` provides a sensible default (e.g., `"convert.yolo_to_coco"`).

**The old `verbose`, `logger`, and `log_file_path` constructor parameters are removed** —
they are replaced by a single `log_config: Optional[LogConfig]` parameter.

### 5.2 Result Object Contract

Result objects carry the log file path so callers can report it:

| Result Class | Field | Type |
|-------------|-------|------|
| `ConversionResult` | `log_path` | `Optional[str]` |
| `VisualizationResult` | `log_path` | `Optional[str]` |
| `EvaluationResult` | `log_path` | `Optional[str]` |

### 5.3 Log Propagation to Handlers

Modules pass their logger (or a child logger) to Label handlers:

```python
handler = YoloAnnotationHandler(
    ...,
    logger=self._log_manager.child("handler"),
)
```

This ensures handler log messages appear under the module's logger hierarchy
and are captured in the same log file.

---

## 6. CLI Contract

### 6.1 What CLI Does

```python
# CLI parses command-line options
log_config = LogConfig(
    name=f"dataflow.{command_name}",
    verbose=verbose,
    log_dir=log_dir,
)

# CLI passes config to module constructor
converter = YoloAndCocoConverter(
    ...,
    log_config=log_config,
    strict_mode=strict,
)

# Module handles ALL logging internally
result = converter.convert(...)

# CLI only uses click.echo() for terminal output
if result.success:
    click.echo(f"✓ {result.get_summary()}")
else:
    click.echo(f"✗ Failed: {result.errors[0]}", err=True)

if result.log_path:
    click.echo(f"Log saved to: {result.log_path}")
```

### 6.2 What CLI Does NOT Do

- CLI does **NOT** create `LoggingOperations` or `VerboseLoggingOperations`
- CLI does **NOT** write log messages (`logger.info(...)`)
- CLI does **NOT** pass raw `logger` objects to module constructors
- CLI does **NOT** store `log_file_path` in `ctx.obj`

### 6.3 Decorator Simplification

The `add_common_options` / `add_visualize_options` decorators no longer create loggers.
They only parse and store parameter values:

```python
def add_common_options(func):
    @click.option("--verbose", is_flag=True, ...)
    @click.option("--no-strict", is_flag=True, ...)
    @click.option("--log-dir", default="./logs", ...)
    @click.pass_context
    @wraps(func)
    def wrapper(ctx, verbose, no_strict, log_dir, *args, **kwargs):
        ctx.obj["verbose"] = verbose
        ctx.obj["strict"] = not no_strict
        ctx.obj["log_dir"] = Path(log_dir)
        # No logger creation here
        return func(ctx, *args, **kwargs)
    return wrapper
```

---

## 7. Error Handling Contract

### 7.1 Module-Level Error Logging

Each module base class implements its own error handling. The shared `logging_error_or_raise()`
is removed — each module defines its own `_log_error()` with behavior specific to its contract:

| Module | `_log_error()` behavior |
|--------|------------------------|
| Label | Log ERROR + raise `ValueError` in strict mode; image errors → WARNING (no raise) |
| Convert | Log ERROR + raise `ValueError` in strict mode |
| Visualize | Log ERROR (never raises — read-only operation); image errors → WARNING |
| Evaluate | Log ERROR + always raise `ValueError` |

### 7.2 `detect_image_error()`

Moved to `dataflow/visualize/utils.py` (only consumer) or kept as a small standalone function
in `dataflow/util/logging.py` since the Label module also uses it via `BaseAnnotationHandler`.

**Decision**: Keep in `dataflow/util/logging.py` as a pure utility function — both Label and
Visualize modules need image error detection.

---

## 8. Migration from Old API

| Old Pattern | New Pattern |
|-------------|-------------|
| `LoggingOperations().get_logger(name)` | `LogManager(LogConfig(name=name)).logger` |
| `VerboseLoggingOperations().get_verbose_logger(name, verbose=True, log_dir=...)` | `LogManager(LogConfig(name=name, verbose=True, log_dir=...))` |
| `logging_error_or_raise(msg, logger, strict_mode, is_image_error)` | `self._log_error(msg)` — each base class defines its own |
| `LoggingOperations().get_logger(name, log_file=...)` | `LogManager(LogConfig(name=name, verbose=bool(log_file)))` |
| `VerboseLoggingOperations().create_progress_logger(name)` | `log_manager.child("progress")` |
| `VerboseLoggingOperations().log_summary(logger, title, data)` | `logger.info(format_result_block(title, data))` |
| Constructor accepts `verbose`, `logger`, `log_file_path` | Constructor accepts `log_config: LogConfig` |

---

## 9. Dependency Contract

```
dataflow/util/logging.py imports FROM:
└── logging (stdlib)
└── pathlib (stdlib)
└── datetime (stdlib)
└── sys (stdlib)

dataflow/util/logging.py is imported BY:
├── dataflow.label.base           (LogManager → handler logger)
├── dataflow.convert.base         (LogManager → converter pipeline)
├── dataflow.visualize.base       (LogManager → visualization pipeline)
├── dataflow.evaluate.base        (LogManager → evaluation pipeline)
├── dataflow.cli.commands.*       (LogConfig only — passed to module constructors)
└── dataflow.cli.main             (LogConfig only)
```

`LogManager` does NOT import from any DataFlow-CV module — it is a pure leaf dependency.
