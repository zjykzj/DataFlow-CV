"""Tests for CLI convert commands."""

from click.testing import CliRunner
from dataflow.cli.main import cli


def test_convert_help():
    """Test that convert command help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "--help"])
    assert result.exit_code == 0
    assert "格式转换命令组" in result.output


def test_convert_yolo2coco_help():
    """Test that convert yolo2coco subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "yolo2coco", "--help"])
    assert result.exit_code == 0
    assert "YOLO格式转COCO格式" in result.output


def test_convert_yolo2labelme_help():
    """Test that convert yolo2labelme subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "yolo2labelme", "--help"])
    assert result.exit_code == 0
    assert "YOLO格式转LabelMe格式" in result.output


def test_convert_coco2yolo_help():
    """Test that convert coco2yolo subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "coco2yolo", "--help"])
    assert result.exit_code == 0
    assert "COCO格式转YOLO格式" in result.output


def test_convert_coco2labelme_help():
    """Test that convert coco2labelme subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "coco2labelme", "--help"])
    assert result.exit_code == 0
    assert "COCO格式转LabelMe格式" in result.output


def test_convert_labelme2yolo_help():
    """Test that convert labelme2yolo subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "labelme2yolo", "--help"])
    assert result.exit_code == 0
    assert "LabelMe格式转YOLO格式" in result.output


def test_convert_labelme2coco_help():
    """Test that convert labelme2coco subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "labelme2coco", "--help"])
    assert result.exit_code == 0
    assert "LabelMe格式转COCO格式" in result.output