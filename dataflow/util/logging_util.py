"""
Logging utilities for DataFlow-CV.

Provides unified logging configuration with support for console and file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import datetime


class LoggingOperations:
    """Logging operations utility class."""

    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self):
        self.loggers = {}

    def get_logger(self, name: str = "dataflow",
                  level: int = logging.INFO,
                  log_file: Optional[str] = None,
                  console: bool = True) -> logging.Logger:
        """Get a configured logger."""
        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Clear existing handlers
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            self.DEFAULT_FORMAT,
            datefmt=self.DEFAULT_DATE_FORMAT
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
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
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

    def setup_root_logger(self, level: int = logging.INFO,
                         log_file: Optional[str] = None):
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
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            handlers.append(file_handler)

        handlers.append(console_handler)

        # Configure root logger
        logging.basicConfig(
            level=level,
            format=self.DEFAULT_FORMAT,
            datefmt=self.DEFAULT_DATE_FORMAT,
            handlers=handlers
        )

    def create_log_file(self, base_name: str = "dataflow",
                       directory: str = "./logs") -> str:
        """Create a log file path with timestamp."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = dir_path / f"{base_name}_{timestamp}.log"

        return str(log_file)

    def set_log_level(self, logger_name: str, level: str):
        """Set log level for a logger."""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        if level not in level_map:
            raise ValueError(f"Invalid log level: {level}")

        logger = logging.getLogger(logger_name)
        logger.setLevel(level_map[level])

        # Update all handlers' levels
        for handler in logger.handlers:
            handler.setLevel(level_map[level])