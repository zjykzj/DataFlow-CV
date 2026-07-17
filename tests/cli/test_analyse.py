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


# ---------------------------------------------------------------------------
# Partition CLI tests
# ---------------------------------------------------------------------------


def test_partition_help():
    """``analyse partition --help`` shows subcommand description."""
    runner = CliRunner()
    result = runner.invoke(cli, ["analyse", "partition", "--help"])
    assert result.exit_code == 0
    assert "partition" in result.output.lower()


def test_partition_missing_num():
    """``analyse partition`` without --num shows an error."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        "analyse", "partition", "/tmp/out",
    ])
    assert result.exit_code != 0


def test_partition_no_input_error():
    """``analyse partition`` without --label-dir or --image-dir errors."""
    runner = CliRunner()
    result = runner.invoke(cli, [
        "analyse", "partition", "/tmp/out", "--num", "2",
    ])
    assert result.exit_code != 0
    assert "at least one" in result.output.lower()


def test_partition_move_declined(tmp_path):
    """``analyse partition --move`` declined by user aborts."""
    from pathlib import Path

    test_data = (
        Path(__file__).parent.parent.parent / "assets" / "test_data"
    )
    yolo_labels = test_data / "det" / "yolo" / "labels"
    classes = test_data / "det" / "yolo" / "classes.txt"
    output_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "analyse", "partition",
        str(output_dir),
        "--num", "2",
        "--label-dir", str(yolo_labels),
        "-c", str(classes),
        "--move",
    ], input="n\n")
    # User declined → abort, exit code 1 (click.Abort)
    assert result.exit_code != 0


def test_partition_yolo_smoke(tmp_path):
    """``analyse partition`` with YOLO labels succeeds."""
    from pathlib import Path

    test_data = (
        Path(__file__).parent.parent.parent / "assets" / "test_data"
    )
    yolo_labels = test_data / "det" / "yolo" / "labels"
    classes = test_data / "det" / "yolo" / "classes.txt"
    output_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(cli, [
        "analyse", "partition",
        str(output_dir),
        "--num", "2",
        "--label-dir", str(yolo_labels),
        "-c", str(classes),
    ])
    assert result.exit_code == 0
    assert (output_dir / "part_1").is_dir()
    assert (output_dir / "part_2").is_dir()
