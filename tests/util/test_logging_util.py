"""
Unit tests for logging_util.py
"""

import logging
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from dataflow.util.logging_util import (LoggingOperations,
                                        VerboseLoggingOperations)


class TestLoggingOperations:
    """Test suite for LoggingOperations class."""

    @pytest.fixture
    def log_ops(self):
        """Create a LoggingOperations instance for testing."""
        return LoggingOperations()

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for log files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_logger_creates_new(self, log_ops):
        """Test get_logger creates new logger."""
        logger_name = "test_logger"
        logger = log_ops.get_logger(logger_name)

        assert logger.name == logger_name
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1  # Console handler by default
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_get_logger_caches(self, log_ops):
        """Test get_logger returns cached logger."""
        logger_name = "cached_logger"
        logger1 = log_ops.get_logger(logger_name)
        logger2 = log_ops.get_logger(logger_name)

        assert logger1 is logger2

    def test_get_logger_with_file(self, log_ops, temp_log_dir):
        """Test get_logger with file output."""
        log_file = temp_log_dir / "test.log"
        logger = log_ops.get_logger("file_logger", log_file=str(log_file))

        assert len(logger.handlers) == 2  # Console + file
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert log_file.exists()

    def test_get_logger_no_console(self, log_ops):
        """Test get_logger without console output."""
        logger = log_ops.get_logger("no_console", console=False)

        assert len(logger.handlers) == 0

    def test_get_logger_custom_level(self, log_ops):
        """Test get_logger with custom log level."""
        logger = log_ops.get_logger("debug_logger", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

        # Check handler level
        for handler in logger.handlers:
            assert handler.level == logging.DEBUG

    def test_setup_root_logger(self, log_ops, temp_log_dir):
        """Test setup_root_logger configures root logger."""
        log_file = temp_log_dir / "root.log"
        log_ops.setup_root_logger(level=logging.WARNING, log_file=str(log_file))

        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING
        assert len(root_logger.handlers) >= 1

        # Log a message to ensure it works
        test_message = "Test root logger message"
        logging.warning(test_message)

        # Force flush all handlers to ensure message is written
        for handler in root_logger.handlers:
            handler.flush()

        # Check file was created and contains message
        assert log_file.exists()
        log_content = log_file.read_text()
        assert test_message in log_content

    def test_create_log_file(self, log_ops, temp_log_dir):
        """Test create_log_file generates timestamped log file path."""
        log_path = log_ops.create_log_file("test_app", directory=str(temp_log_dir))

        assert Path(log_path).parent == temp_log_dir
        assert log_path.startswith(str(temp_log_dir / "test_app_"))
        assert log_path.endswith(".log")

        # Check timestamp format (YYYYMMDD_HHMMSS)
        import re

        filename = Path(log_path).name
        timestamp = filename.replace("test_app_", "").replace(".log", "")
        assert re.match(r"\d{8}_\d{6}", timestamp)

    def test_set_log_level_valid(self, log_ops):
        """Test set_log_level with valid level."""
        logger_name = "level_test"
        logger = log_ops.get_logger(logger_name, level=logging.INFO)

        # Change to DEBUG
        log_ops.set_log_level(logger_name, "DEBUG")
        assert logger.level == logging.DEBUG

        # Change to ERROR
        log_ops.set_log_level(logger_name, "ERROR")
        assert logger.level == logging.ERROR

    def test_set_log_level_invalid(self, log_ops):
        """Test set_log_level with invalid level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            log_ops.set_log_level("test", "INVALID_LEVEL")

    def test_logger_output(self, log_ops, temp_log_dir):
        """Test logger actually outputs messages."""
        log_file = temp_log_dir / "output_test.log"
        logger = log_ops.get_logger("output_test", log_file=str(log_file))

        test_message = "This is a test message"
        logger.info(test_message)

        # Force flush
        for handler in logger.handlers:
            handler.flush()

        assert log_file.exists()
        log_content = log_file.read_text()
        assert test_message in log_content
        assert "INFO" in log_content
        assert "output_test" in log_content

    def test_multiple_loggers_independent(self, log_ops):
        """Test multiple loggers can have different configurations."""
        logger1 = log_ops.get_logger("logger1", level=logging.DEBUG, console=False)
        logger2 = log_ops.get_logger("logger2", level=logging.ERROR)

        assert logger1.level == logging.DEBUG
        assert logger2.level == logging.ERROR
        assert len(logger1.handlers) == 0
        assert len(logger2.handlers) == 1
        assert logger1 is not logger2


class TestVerboseLoggingOperations:
    """Test suite for VerboseLoggingOperations class."""

    @pytest.fixture
    def verbose_log_ops(self):
        """Create a VerboseLoggingOperations instance for testing."""
        return VerboseLoggingOperations()

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for log files."""
        import shutil
        import tempfile

        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_verbose_logger_creation(self, verbose_log_ops):
        """Test verbose logger creation."""
        logger = verbose_log_ops.get_verbose_logger("test", verbose=True)
        assert logger.name == "test.verbose"
        # Should have 2 handlers: console + file
        assert len(logger.handlers) == 2

    def test_verbose_logger_without_verbose(self, verbose_log_ops):
        """Test verbose logger without verbose flag."""
        logger = verbose_log_ops.get_verbose_logger("test", verbose=False)
        assert logger.name == "test.verbose"
        # Should have only console handler when verbose=False
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_log_file_creation(self, verbose_log_ops, temp_log_dir):
        """Test log file creation."""
        logger = verbose_log_ops.get_verbose_logger(
            "test", verbose=True, log_dir=str(temp_log_dir)
        )

        # Log a message to trigger file creation
        logger.info("Test message")
        for handler in logger.handlers:
            handler.flush()

        # Check if log files were created
        log_files = list(temp_log_dir.glob("log_*.log"))
        assert len(log_files) > 0

        # Check file content
        log_content = log_files[0].read_text()
        assert "Test message" in log_content

    def test_progress_logger(self, verbose_log_ops):
        """Test progress logger."""
        logger = verbose_log_ops.create_progress_logger()
        assert logger.name == "dataflow.progress"
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_log_summary(self, verbose_log_ops, capsys):
        """Test log_summary method."""
        logger = logging.getLogger("summary_test")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)

        # Create console handler that outputs to sys.stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        # Use simple formatter like log_summary expects
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

        test_data = {
            "module": "TestModule",
            "runtime": "12.34s",
            "stats": {"processed": 150, "failed": 0},
        }

        verbose_log_ops.log_summary(logger, "Test Summary", test_data)

        # Force flush to ensure output is captured
        for handler in logger.handlers:
            handler.flush()

        # Capture output
        captured = capsys.readouterr()
        assert "Test Summary" in captured.out
        assert "TestModule" in captured.out
        assert "processed" in captured.out

    def test_verbose_logger_caching(self, verbose_log_ops):
        """Test verbose logger caching."""
        logger1 = verbose_log_ops.get_verbose_logger("cached_test", verbose=True)
        logger2 = verbose_log_ops.get_verbose_logger("cached_test", verbose=True)

        assert logger1 is logger2
