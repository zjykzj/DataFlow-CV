# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/14
@File    : test_cli.py
@Author  : DataFlow Team
@Description: Tests for CLI interface
"""

import os
import sys
import tempfile
import unittest
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataflow import __version__
from dataflow.config import Config


class TestCLI(unittest.TestCase):
    """Test cases for CLI interface."""

    def test_version_option(self):
        """Test --version option."""
        result = subprocess.run(
            [sys.executable, "-m", "dataflow.cli", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        self.assertIn(__version__, result.stdout)
        self.assertEqual(result.returncode, 0)

    def test_help_option(self):
        """Test --help option."""
        result = subprocess.run(
            [sys.executable, "-m", "dataflow.cli", "--help"],
            capture_output=True,
            text=True,
            check=True
        )
        self.assertIn("DataFlow-CV", result.stdout)
        self.assertIn("Usage:", result.stdout)
        self.assertEqual(result.returncode, 0)

    def test_convert_help(self):
        """Test convert command help."""
        result = subprocess.run(
            [sys.executable, "-m", "dataflow.cli", "convert", "--help"],
            capture_output=True,
            text=True,
            check=True
        )
        self.assertIn("Convert between different annotation formats", result.stdout)
        self.assertEqual(result.returncode, 0)

    def test_visualize_help(self):
        """Test visualize command help."""
        result = subprocess.run(
            [sys.executable, "-m", "dataflow.cli", "visualize", "--help"],
            capture_output=True,
            text=True,
            check=True
        )
        self.assertIn("Visualize annotations on images", result.stdout)
        self.assertEqual(result.returncode, 0)

    def test_config_command(self):
        """Test config command."""
        result = subprocess.run(
            [sys.executable, "-m", "dataflow.cli", "config"],
            capture_output=True,
            text=True,
            check=True
        )
        self.assertIn("DataFlow-CV Configuration", result.stdout)
        self.assertIn("Global Configuration:", result.stdout)
        self.assertEqual(result.returncode, 0)

    def test_invalid_command(self):
        """Test invalid command handling."""
        result = subprocess.run(
            [sys.executable, "-m", "dataflow.cli", "invalidcommand"],
            capture_output=True,
            text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No such command", result.stderr)

    def test_convert_coco2yolo_help(self):
        """Test convert coco2yolo help."""
        result = subprocess.run(
            [sys.executable, "-m", "dataflow.cli", "convert", "coco2yolo", "--help"],
            capture_output=True,
            text=True,
            check=True
        )
        self.assertIn("Convert COCO JSON to YOLO format", result.stdout)
        self.assertEqual(result.returncode, 0)

    def test_visualize_yolo_help(self):
        """Test visualize yolo help."""
        result = subprocess.run(
            [sys.executable, "-m", "dataflow.cli", "visualize", "yolo", "--help"],
            capture_output=True,
            text=True,
            check=True
        )
        self.assertIn("Visualize YOLO format annotations", result.stdout)
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()