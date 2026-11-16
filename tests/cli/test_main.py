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


def test_python_m_main_module(tmp_path):
    """`python -m dataflow.cli` works (__main__.py exists)."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "dataflow.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),  # outside the repo root so cwd shadowing can't mask it
    )
    assert proc.returncode == 0
    assert "DataFlow-CV" in proc.stdout
    assert "__main__" not in proc.stderr  # no module-name warning
