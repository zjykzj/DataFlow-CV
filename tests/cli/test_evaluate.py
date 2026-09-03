"""CLI tests for evaluate subcommands."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from dataflow.cli.main import cli

TEST_DATA = Path(__file__).parent.parent.parent / "assets" / "test_data" / "evaluate"
GT_JSON = str(TEST_DATA / "gt_coco.json")
DT_JSON = str(TEST_DATA / "dt_coco.json")
GT_SEGM_JSON = str(TEST_DATA / "gt_coco_segm.json")
DT_SEGM_JSON = str(TEST_DATA / "dt_coco_segm.json")


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class TestEvaluateDetection:
    """Test 'evaluate detection' CLI command."""

    def test_help(self, runner):
        result = runner.invoke(cli, ["evaluate", "detection", "--help"])
        assert result.exit_code == 0
        assert "GT_JSON" in result.output
        assert "DT_JSON" in result.output

    def test_basic(self, runner):
        result = runner.invoke(cli, ["evaluate", "detection", GT_JSON, DT_JSON])
        assert result.exit_code == 0
        assert "AP" in result.output

    def test_verbose(self, runner):
        result = runner.invoke(cli, ["evaluate", "detection", "--verbose", GT_JSON, DT_JSON])
        assert result.exit_code == 0

    def test_prf1(self, runner):
        result = runner.invoke(
            cli,
            ["evaluate", "detection", "--prf1", GT_JSON, DT_JSON],
        )
        assert result.exit_code == 0
        assert "F1" in result.output

    def test_prf1_micro(self, runner):
        result = runner.invoke(
            cli,
            [
                "evaluate",
                "detection",
                "--prf1",
                "--prf1-method",
                "micro",
                GT_JSON,
                DT_JSON,
            ],
        )
        assert result.exit_code == 0
        assert "Method=micro" in result.output

    def test_prf1_custom_thresholds(self, runner):
        result = runner.invoke(
            cli,
            [
                "evaluate",
                "detection",
                "--prf1",
                "--prf1-iou",
                "0.75",
                "--prf1-conf",
                "0.5",
                GT_JSON,
                DT_JSON,
            ],
        )
        assert result.exit_code == 0
        assert "IoU=0.75" in result.output

    def test_output_json(self, runner, tmp_path):
        output_file = tmp_path / "result.json"
        result = runner.invoke(
            cli,
            [
                "evaluate",
                "detection",
                "--output",
                str(output_file),
                GT_JSON,
                DT_JSON,
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

    def test_missing_gt(self, runner):
        result = runner.invoke(cli, ["evaluate", "detection", "/nonexistent/gt.json", DT_JSON])
        assert result.exit_code != 0

    def test_missing_dt(self, runner):
        result = runner.invoke(cli, ["evaluate", "detection", GT_JSON, "/nonexistent/dt.json"])
        assert result.exit_code != 0

    def test_missing_both_args(self, runner):
        result = runner.invoke(cli, ["evaluate", "detection"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


class TestEvaluateSegmentation:
    """Test 'evaluate segmentation' CLI command."""

    def test_help(self, runner):
        result = runner.invoke(cli, ["evaluate", "segmentation", "--help"])
        assert result.exit_code == 0

    def test_basic(self, runner):
        result = runner.invoke(cli, ["evaluate", "segmentation", GT_SEGM_JSON, DT_SEGM_JSON])
        assert result.exit_code == 0

    def test_verbose(self, runner):
        result = runner.invoke(
            cli,
            ["evaluate", "segmentation", "--verbose", GT_SEGM_JSON, DT_SEGM_JSON],
        )
        assert result.exit_code == 0

    def test_prf1(self, runner):
        result = runner.invoke(
            cli,
            ["evaluate", "segmentation", "--prf1", GT_SEGM_JSON, DT_SEGM_JSON],
        )
        assert result.exit_code == 0
        assert "F1" in result.output

    def test_missing_gt(self, runner):
        result = runner.invoke(
            cli,
            ["evaluate", "segmentation", "/nonexistent/gt.json", DT_SEGM_JSON],
        )
        assert result.exit_code != 0

    def test_missing_args(self, runner):
        result = runner.invoke(cli, ["evaluate", "segmentation"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Evaluate group
# ---------------------------------------------------------------------------


class TestEvaluateGroup:
    """Test the 'evaluate' command group itself."""

    def test_help(self, runner):
        result = runner.invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "detection" in result.output
        assert "segmentation" in result.output
