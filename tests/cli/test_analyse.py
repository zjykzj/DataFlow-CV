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


def test_filter_help():
    """``analyse filter --help`` shows subcommand description."""
    runner = CliRunner()
    result = runner.invoke(cli, ["analyse", "filter", "--help"])
    assert result.exit_code == 0
    assert "filter" in result.output.lower()


def test_filter_missing_args():
    """``analyse filter`` with no args shows an error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["analyse", "filter"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Stats: multi-path and --recursive
# ---------------------------------------------------------------------------


def test_stats_help_shows_recursive():
    """``analyse stats --help`` shows --recursive option."""
    runner = CliRunner()
    result = runner.invoke(cli, ["analyse", "stats", "--help"])
    assert result.exit_code == 0
    assert "--recursive" in result.output


def test_stats_multi_path(tmp_path):
    """``analyse stats`` with multiple LABEL_PATH args succeeds."""
    from pathlib import Path

    test_data = (
        Path(__file__).parent.parent.parent / "assets" / "test_data"
    )
    yolo_labels = test_data / "det" / "yolo" / "labels"
    classes = test_data / "det" / "yolo" / "classes.txt"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "analyse", "stats",
        "-c", str(classes),
        str(yolo_labels),
        str(yolo_labels),  # same path twice
    ])
    assert result.exit_code == 0


def test_stats_no_paths_errors():
    """``analyse stats`` with no LABEL_PATH shows usage error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["analyse", "stats"])
    assert result.exit_code != 0


def test_stats_recursive_flag_accepted(tmp_path):
    """``analyse stats --recursive`` is accepted as a valid option."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        "analyse", "stats", "--recursive", str(tmp_path),
    ])
    # Will fail because tmp_path has no label files, but the flag itself
    # should be recognised (not an "unknown option" error)
    assert "No such option" not in result.output
