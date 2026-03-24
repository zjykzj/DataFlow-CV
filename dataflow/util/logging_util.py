"""
Logging utilities for DataFlow-CV.

Provides unified logging configuration with support for console and file output.
"""

import datetime
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional


class LoggingOperations:
    """Logging operations utility class."""

    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        self.loggers = {}

    def get_logger(
        self,
        name: str = "dataflow",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        console: bool = True,
    ) -> logging.Logger:
        """Get a configured logger."""
        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Clear existing handlers
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            self.DEFAULT_FORMAT, datefmt=self.DEFAULT_DATE_FORMAT
        )

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler
        if log_file:
            self._ensure_log_dir(log_file)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self.loggers[name] = logger
        return logger

    def _ensure_log_dir(self, log_file: str):
        """Ensure log directory exists."""
        log_path = Path(log_file)
        if log_path.parent:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def setup_root_logger(
        self, level: int = logging.INFO, log_file: Optional[str] = None
    ):
        """Setup root logger configuration."""
        # Clear any existing handlers on root logger to ensure basicConfig works
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        handlers: List[logging.Handler] = []

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # File handler
        if log_file:
            self._ensure_log_dir(log_file)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            handlers.append(file_handler)

        handlers.append(console_handler)

        # Configure root logger
        logging.basicConfig(
            level=level,
            format=self.DEFAULT_FORMAT,
            datefmt=self.DEFAULT_DATE_FORMAT,
            handlers=handlers,
        )

    def create_log_file(
        self, base_name: str = "dataflow", directory: str = "./logs"
    ) -> str:
        """Create a log file path with timestamp."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = dir_path / f"{base_name}_{timestamp}.log"

        return str(log_file)

    def set_log_level(self, logger_name: str, level: str):
        """Set log level for a logger."""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        if level not in level_map:
            raise ValueError(f"Invalid log level: {level}")

        logger = logging.getLogger(logger_name)
        logger.setLevel(level_map[level])

        # Update all handlers' levels
        for handler in logger.handlers:
            handler.setLevel(level_map[level])


class VerboseLoggingOperations(LoggingOperations):
    """Enhanced logging operations with verbose mode support."""

    VERBOSE_FORMAT = (
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    SUMMARY_FORMAT = "%(message)s"

    def __init__(self):
        super().__init__()
        self.verbose_loggers = {}

    def get_verbose_logger(
        self, name: str = "dataflow", verbose: bool = False, log_dir: str = "./logs"
    ) -> logging.Logger:
        """
        Get a configured verbose mode logger.

        Args:
            name: logger name
            verbose: whether to enable verbose logging mode
            log_dir: log file directory

        Returns:
            Configured logger instance
        """
        if name in self.verbose_loggers:
            return self.verbose_loggers[name]

        logger = logging.getLogger(f"{name}.verbose")

        # Clear existing handlers
        logger.handlers.clear()

        # Add console handler (always added)
        console_formatter = logging.Formatter(self.DEFAULT_FORMAT)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # If verbose=True, add file handler
        if verbose:
            # Ensure log directory exists
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            # Create timestamped log filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = Path(log_dir) / f"log_{timestamp}.log"

            # Use RotatingFileHandler to prevent oversized files
            file_handler = RotatingFileHandler(
                str(log_file),
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(self.VERBOSE_FORMAT)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        logger.setLevel(logging.DEBUG)
        self.verbose_loggers[name] = logger
        return logger

    def create_progress_logger(self, name: str = "dataflow.progress") -> logging.Logger:
        """
        Create a progress report dedicated logger.

        Args:
            name: logger name

        Returns:
            Progress logger instance
        """
        logger = logging.getLogger(name)
        logger.handlers.clear()

        # Progress logger uses simple format
        formatter = logging.Formatter("%(message)s")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        return logger

    def log_summary(self, logger: logging.Logger, title: str, data: Dict[str, Any]):
        """
        Log formatted summary information.

        Args:
            logger: logger to use
            title: summary title
            data: summary data dictionary
        """
        summary = self._format_summary(title, data)
        logger.info(summary)

    def _format_summary(self, title: str, data: Dict[str, Any]) -> str:
        """Format summary information into beautiful text."""
        lines = []
        lines.append("=" * 60)
        lines.append(title.center(60))
        lines.append("=" * 60)

        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  • {sub_key}: {sub_value}")
            else:
                lines.append(f"• {key}: {value}")

        lines.append("=" * 60)
        return "\n".join(lines)
