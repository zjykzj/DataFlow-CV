"""
Unified logging infrastructure for DataFlow-CV.

Provides ``LogManager`` — the single entry point for all module logging,
replacing the old ``LoggingOperations`` and ``VerboseLoggingOperations``.
Also provides general-purpose format helpers and ``detect_image_error``.

Usage::

    from dataflow.util.logging import LogConfig, LogManager

    config = LogConfig(name="convert.yolo_to_coco", verbose=True)
    manager = LogManager(config)
    logger = manager.logger

    logger.info("Processing started")
    # ... do work ...
    logger.info("Processing completed")

    if manager.log_path:
        print(f"Log saved to: {manager.log_path}")
"""

import datetime
import logging
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogConfig:
    """Immutable logging configuration.

    Attributes:
        name: Logger name, e.g. ``"convert.yolo_to_coco"``.
            Used as the Python logger name for hierarchical filtering.
        verbose: If ``True``, enable file logging (RotatingFileHandler
            at DEBUG level). If ``False``, console-only output at INFO
            level.
        log_dir: Directory for log files. Only used when
            ``verbose=True``.  Default: ``Path("./logs")``.
    """

    name: str
    verbose: bool = False
    log_dir: Path = Path("./logs")


# ---------------------------------------------------------------------------
# LogManager
# ---------------------------------------------------------------------------


class LogManager:
    """Unified logging manager.

    One ``LogManager`` per module instance.  Created at module entry
    point and propagated to child components via ``self.logger`` or
    :meth:`child`.

    Replaces both ``LoggingOperations`` and ``VerboseLoggingOperations``
    from the old ``dataflow.util.logging_util`` module.
    """

    # Console format (compact — time + level + message)
    _CONSOLE_FORMAT = "%(asctime)s  %(levelname)-7s  %(message)s"
    _CONSOLE_DATE_FORMAT = "%H:%M:%S"

    # File format (verbose — includes module name and line number)
    _FILE_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)s:%(lineno)d  %(message)s"
    _FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # Rotating file limits
    _FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
    _FILE_BACKUP_COUNT = 5

    def __init__(self, config: LogConfig) -> None:
        self._config = config
        self._log_path: Optional[str] = None
        self._logger = self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> logging.Logger:
        """Create and configure the internal logger."""
        logger = logging.getLogger(self._config.name)
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # Console handler (always active, INFO level, compact format)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter(self._CONSOLE_FORMAT, datefmt=self._CONSOLE_DATE_FORMAT)
        )
        logger.addHandler(console_handler)

        # File handler (only when verbose)
        if self._config.verbose:
            log_dir = Path(self._config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"log_{timestamp}.log"

            file_handler = RotatingFileHandler(
                str(log_file),
                maxBytes=self._FILE_MAX_BYTES,
                backupCount=self._FILE_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter(self._FILE_FORMAT, datefmt=self._FILE_DATE_FORMAT)
            )
            logger.addHandler(file_handler)

            self._log_path = str(log_file)

        return logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def logger(self) -> logging.Logger:
        """The configured logger — use for all logging calls."""
        return self._logger

    @property
    def log_path(self) -> Optional[str]:
        """Log file path when ``verbose=True``, else ``None``."""
        return self._log_path

    def child(self, suffix: str) -> logging.Logger:
        """Create a child logger for sub-components.

        Example:
            handler_logger = manager.child("handler")
            # → logger name "convert.yolo_to_coco.handler"
        """
        return self._logger.getChild(suffix)


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

_DEFAULT_WIDTH = 60
_DEFAULT_CHAR = "─"  # ─


def format_divider(char: str = _DEFAULT_CHAR, width: int = _DEFAULT_WIDTH) -> str:
    """Return a horizontal divider line."""
    return char * width


def format_section(title: str) -> str:
    """Return a section header.

    Example: ``"── Load ──"``
    """
    return f"{_DEFAULT_CHAR * 2} {title} {_DEFAULT_CHAR * 2}"


def format_kv(key: str, value: Any, indent: int = 2) -> str:
    """Return a key-value line.

    Example: ``"  Source: yolo_labels/"``
    """
    prefix = " " * indent
    return f"{prefix}{key}: {value}"


def format_result_block(
    status: str,
    items: Dict[str, Any],
    log_path: Optional[str] = None,
    width: int = _DEFAULT_WIDTH,
) -> str:
    """Return a formatted result summary block.

    Args:
        status: Status line text, e.g. ``"✓ Success"``.
        items: Key-value pairs to display.
        log_path: Optional log file path to append at end.
        width: Total width of the block.

    Returns:
        Formatted string.
    """
    lines: List[str] = []
    lines.append(format_divider(width=width))
    lines.append(format_section("Result"))
    lines.append(f"  Status:   {status}")
    for key, value in items.items():
        lines.append(f"  {key}: {str(value):>{width - len(key) - 7}}")
    if log_path:
        lines.append("")
        lines.append(f"  Log saved to: {log_path}")
    lines.append(format_divider(width=width))
    return "\n".join(lines)


def format_table(
    headers: List[str],
    rows: List[List[str]],
    col_widths: Optional[List[int]] = None,
) -> str:
    """Render a formatted table with aligned columns.

    Args:
        headers: Column header strings.
        rows: Row data — each row is a list of cell strings.
        col_widths: Optional pre-computed column widths.  If ``None``,
            widths are computed from headers and row data.

    Returns:
        Formatted table string with borders.
    """
    if not headers:
        return ""

    # Compute column widths
    if col_widths is None:
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

    def _format_row(cells: List[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            if i < len(col_widths):
                parts.append(f" {str(cell):<{col_widths[i]}} ")
        return "│" + "│".join(parts) + "│"

    # Top border
    border = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
    # Separator
    sep = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
    # Bottom border
    bottom = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"

    output = [border, _format_row(headers), sep]
    for row in rows:
        # Pad short rows
        padded = list(row) + [""] * (len(headers) - len(row))
        output.append(_format_row(padded))
    output.append(bottom)

    return "\n".join(output)


# ---------------------------------------------------------------------------
# Error detection utility
# ---------------------------------------------------------------------------

# Keywords that indicate an image-related error (always downgraded to
# warnings, never cause a raise even in strict mode).
_IMAGE_ERROR_KEYWORDS: Tuple[str, ...] = (
    "not found",
    "failed to load",
    "failed to read",
    "invalid",
    "error getting",
    "no corresponding",
    "does not exist",
)


def detect_image_error(message: str) -> bool:
    """Check if an error message indicates an image-related error.

    Image errors should always be treated as warnings regardless of
    ``strict_mode``.  This function checks for known image-error
    keywords.

    Args:
        message: The error message to inspect.

    Returns:
        ``True`` if the message appears to be image-related.
    """
    msg_lower = message.lower()
    return "image" in msg_lower and any(kw in msg_lower for kw in _IMAGE_ERROR_KEYWORDS)
