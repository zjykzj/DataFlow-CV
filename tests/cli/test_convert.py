"""Tests for CLI convert commands."""

from click.testing import CliRunner
from dataflow.cli.main import cli


def test_convert_help():
    """Test that convert command help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "--help"])
    assert result.exit_code == 0
    assert "Format conversion command group" in result.output


def test_convert_yolo2coco_help():
    """Test that convert yolo2coco subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "yolo2coco", "--help"])
    assert result.exit_code == 0
    assert "Convert YOLO format to COCO format" in result.output


def test_convert_yolo2labelme_help():
    """Test that convert yolo2labelme subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "yolo2labelme", "--help"])
    assert result.exit_code == 0
    assert "Convert YOLO format to LabelMe format" in result.output


def test_convert_coco2yolo_help():
    """Test that convert coco2yolo subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "coco2yolo", "--help"])
    assert result.exit_code == 0
    assert "Convert COCO format to YOLO format" in result.output
    assert "--save-crowd" in result.output


def test_convert_coco2labelme_help():
    """Test that convert coco2labelme subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "coco2labelme", "--help"])
    assert result.exit_code == 0
    assert "Convert COCO format to LabelMe format" in result.output
    assert "--save-crowd" in result.output


def test_convert_coco2yolo_save_crowd(tmp_path):
    """End-to-end: crowd annotations skipped by default, --save-crowd keeps them."""
    import json

    coco = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[10, 10, 60, 10, 60, 60, 10, 60]],
                "bbox": [10, 10, 50, 50],
                "area": 2500.0,
                "iscrowd": 1,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[400, 300, 450, 300, 450, 350, 400, 350]],
                "bbox": [400, 300, 50, 50],
                "area": 2500.0,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "obj"}],
    }
    src = tmp_path / "ann.json"
    src.write_text(json.dumps(coco), encoding="utf-8")

    runner = CliRunner()
    out_skip = tmp_path / "out_skip"
    result = runner.invoke(cli, ["convert", "coco2yolo", str(src), str(out_skip)])
    assert result.exit_code == 0, result.output
    rows = (out_skip / "labels" / "img1.txt").read_text().strip().split("\n")
    assert len(rows) == 1

    out_keep = tmp_path / "out_keep"
    result = runner.invoke(cli, ["convert", "coco2yolo", "--save-crowd", str(src), str(out_keep)])
    assert result.exit_code == 0, result.output
    rows = (out_keep / "labels" / "img1.txt").read_text().strip().split("\n")
    assert len(rows) == 2


def test_convert_labelme2yolo_help():
    """Test that convert labelme2yolo subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "labelme2yolo", "--help"])
    assert result.exit_code == 0
    assert "Convert LabelMe format to YOLO format" in result.output


def test_convert_labelme2coco_help():
    """Test that convert labelme2coco subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["convert", "labelme2coco", "--help"])
    assert result.exit_code == 0
    assert "Convert LabelMe format to COCO format" in result.output
