"""CLI smoke tests for analyse commands."""

from click.testing import CliRunner

from dataflow.cli.main import cli


def test_analyse_help():
    """``analyse --help`` shows command group description."""
    runner = CliRunner()
    result = runner.invoke(cli, ["analyse", "--help"])
    assert result.exit_code == 0
    assert "Dataset analysis" in result.output


def test_stats_help():
    """``analyse stats --help`` shows subcommand description."""
    runner = CliRunner()
    result = runner.invoke(cli, ["analyse", "stats", "--help"])
    assert result.exit_code == 0
    assert "statistics" in result.output.lower()


def test_split_help():
    """``analyse split --help`` shows subcommand description."""
    runner = CliRunner()
    result = runner.invoke(cli, ["analyse", "split", "--help"])
    assert result.exit_code == 0
    assert "split" in result.output.lower()
