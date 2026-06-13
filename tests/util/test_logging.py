"""Tests for the unified LogManager and format helpers."""

import logging
import tempfile
from pathlib import Path

import pytest

from dataflow.util.logging import (
    LogConfig,
    LogManager,
    detect_image_error,
    format_divider,
    format_kv,
    format_result_block,
    format_section,
    format_table,
)


class TestLogConfig:
    """Test LogConfig frozen dataclass."""

    def test_defaults(self):
        config = LogConfig(name="test")
        assert config.name == "test"
        assert config.verbose is False
        assert config.log_dir == Path("./logs")

    def test_verbose(self):
        config = LogConfig(name="test", verbose=True, log_dir=Path("/tmp/logs"))
        assert config.verbose is True
        assert config.log_dir == Path("/tmp/logs")

    def test_immutable(self):
        config = LogConfig(name="test")
        with pytest.raises(Exception):
            config.verbose = True  # type: ignore


class TestLogManager:
    """Test LogManager."""

    def test_non_verbose(self):
        config = LogConfig(name="test_non_verbose")
        mgr = LogManager(config)
        assert mgr.logger is not None
        assert isinstance(mgr.logger, logging.Logger)
        assert mgr.log_path is None

    def test_verbose(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(name="test_verbose", verbose=True, log_dir=Path(tmpdir))
            mgr = LogManager(config)
            assert mgr.log_path is not None
            assert tmpdir in mgr.log_path
            mgr.logger.info("test message")

    def test_log_file_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(name="test_content", verbose=True, log_dir=Path(tmpdir))
            mgr = LogManager(config)
            test_msg = "hello from test_log_file_content"
            mgr.logger.info(test_msg)

            with open(mgr.log_path, "r") as f:
                content = f.read()
            assert "INFO" in content

    def test_child_logger(self):
        config = LogConfig(name="test_parent")
        mgr = LogManager(config)
        child = mgr.child("handler")
        assert child is not None
        assert child.name == "test_parent.handler"

    def test_handlers_cleared(self):
        """Re-creating with the same name clears old handlers."""
        config1 = LogConfig(name="test_reuse")
        mgr1 = LogManager(config1)
        n_handlers1 = len(mgr1.logger.handlers)

        config2 = LogConfig(name="test_reuse")
        mgr2 = LogManager(config2)
        n_handlers2 = len(mgr2.logger.handlers)

        assert n_handlers1 == n_handlers2
        assert n_handlers1 >= 1

    def test_propagation_disabled(self):
        config = LogConfig(name="test_propagate")
        mgr = LogManager(config)
        assert mgr.logger.propagate is False


class TestLogManagerVerbose:
    """Test LogManager in verbose mode with file logging."""

    def test_file_handler_added(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(name="test_file_handler", verbose=True, log_dir=Path(tmpdir))
            mgr = LogManager(config)
            file_handlers = [
                h for h in mgr.logger.handlers
                if isinstance(h, logging.FileHandler)
                or type(h).__name__ == "RotatingFileHandler"
            ]
            assert len(file_handlers) == 1

    def test_console_and_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LogConfig(name="test_both", verbose=True, log_dir=Path(tmpdir))
            mgr = LogManager(config)
            assert len(mgr.logger.handlers) == 2

    def test_no_file_handler_when_not_verbose(self):
        config = LogConfig(name="test_no_file")
        mgr = LogManager(config)
        file_handlers = [
            h for h in mgr.logger.handlers
            if isinstance(h, logging.FileHandler)
            or type(h).__name__ == "RotatingFileHandler"
        ]
        assert len(file_handlers) == 0


class TestDetectImageError:
    """Test detect_image_error utility."""

    def test_image_not_found(self):
        assert detect_image_error("image not found: img_001.jpg") is True

    def test_image_failed_to_load(self):
        assert detect_image_error("image failed to load: img_001.jpg") is True

    def test_no_image_keyword(self):
        assert detect_image_error("file not found: classes.txt") is False

    def test_image_but_no_error_keyword(self):
        assert detect_image_error("image dimensions: 640x480") is False


class TestFormatHelpers:
    """Test general-purpose format helper functions."""

    def test_format_divider(self):
        result = format_divider()
        assert len(result) == 60
        assert all(c == "─" for c in result)

    def test_format_divider_custom(self):
        result = format_divider(char="=", width=10)
        assert result == "=" * 10

    def test_format_section(self):
        result = format_section("Load")
        assert "Load" in result
        assert result.startswith("──")

    def test_format_kv(self):
        result = format_kv("Source", "/path/to/data")
        assert "Source" in result
        assert "/path/to/data" in result

    def test_format_kv_custom_indent(self):
        result = format_kv("Key", "Value", indent=4)
        assert result.startswith("    Key:")

    def test_format_result_block(self):
        items = {"Images": 500, "Objects": 3240, "Duration": "2.15s"}
        result = format_result_block("✓ Success", items)
        assert "✓ Success" in result
        assert "Images" in result
        assert "500" in result

    def test_format_result_block_with_log_path(self):
        items = {"Images": 500}
        result = format_result_block("✓ Success", items, log_path="/tmp/log.log")
        assert "Log saved to" in result
        assert "/tmp/log.log" in result

    def test_format_table(self):
        headers = ["Name", "Count"]
        rows = [["cat", "10"], ["dog", "5"]]
        result = format_table(headers, rows)
        assert "Name" in result
        assert "Count" in result
        assert "cat" in result
        assert "dog" in result
        assert "┌" in result
        assert "└" in result

    def test_format_table_empty_headers(self):
        result = format_table([], [["a", "b"]])
        assert result == ""
