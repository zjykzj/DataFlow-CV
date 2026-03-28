"""Tests for CLI main module."""

from click.testing import CliRunner
from dataflow.cli.main import cli


def test_cli_help():
    """Test that CLI help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "DataFlow-CV" in result.output


def test_cli_version():
    """Test that CLI version works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


