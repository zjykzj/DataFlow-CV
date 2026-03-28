"""Tests for CLI visualize commands."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from click.testing import CliRunner
from dataflow.cli.main import cli


def test_visualize_help():
    """Test that visualize command help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["visualize", "--help"])
    assert result.exit_code == 0
    assert "可视化命令组" in result.output


def test_visualize_yolo_help():
    """Test that visualize yolo subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["visualize", "yolo", "--help"])
    assert result.exit_code == 0
    assert "可视化YOLO格式标签" in result.output


def test_visualize_coco_help():
    """Test that visualize coco subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["visualize", "coco", "--help"])
    assert result.exit_code == 0
    assert "可视化COCO格式标签" in result.output


def test_visualize_labelme_help():
    """Test that visualize labelme subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["visualize", "labelme", "--help"])
    assert result.exit_code == 0
    assert "可视化LabelMe格式标签" in result.output


def test_visualize_yolo_success():
    """Test visualize yolo command with mock success."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test files
        input_dir = tmpdir_path / "input"
        input_dir.mkdir()

        class_file = tmpdir_path / "classes.txt"
        class_file.write_text("person\ncar\n")

        output_dir = tmpdir_path / "output"

        # Mock YOLOVisualizer
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = {"processed_images": 5}
        mock_result.message = "Success"

        mock_visualizer = Mock()
        mock_visualizer.visualize.return_value = mock_result

        with patch('dataflow.visualize.yolo_visualizer.YOLOVisualizer', return_value=mock_visualizer):
            result = runner.invoke(
                cli,
                [
                    "visualize",
                    "yolo",
                    str(input_dir),
                    "--class-file",
                    str(class_file),
                    "--output-dir",
                    str(output_dir),
                    "--save",
                ]
            )

            # Check command executed successfully
            assert result.exit_code == 0
            # Visualizer should have been called
            mock_visualizer.visualize.assert_called_once()


def test_visualize_yolo_missing_class_file():
    """Test visualize yolo command with missing required class file."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test directory
        input_dir = tmpdir_path / "input"
        input_dir.mkdir()

        # Run command without required --class-file
        result = runner.invoke(
            cli,
            [
                "visualize",
                "yolo",
                str(input_dir),
            ]
        )

        # Should fail with error about missing class file
        assert result.exit_code != 0
        assert "需要" in result.output or "required" in result.output or "class-file" in result.output


def test_visualize_coco_success():
    """Test visualize coco command with mock success."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test files
        input_file = tmpdir_path / "annotations.json"
        input_file.write_text('{"images": [], "annotations": []}')

        output_dir = tmpdir_path / "output"

        # Mock COCOVisualizer
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = {"processed_images": 3}
        mock_result.message = "Success"

        mock_visualizer = Mock()
        mock_visualizer.visualize.return_value = mock_result

        with patch('dataflow.visualize.coco_visualizer.COCOVisualizer', return_value=mock_visualizer):
            result = runner.invoke(
                cli,
                [
                    "visualize",
                    "coco",
                    str(input_file),
                    "--output-dir",
                    str(output_dir),
                    "--save",
                ]
            )

            # Check command executed successfully
            assert result.exit_code == 0
            # Visualizer should have been called
            mock_visualizer.visualize.assert_called_once()


def test_visualize_labelme_success():
    """Test visualize labelme command with mock success."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test directory
        input_dir = tmpdir_path / "input"
        input_dir.mkdir()

        output_dir = tmpdir_path / "output"

        # Mock LabelMeVisualizer
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = {"processed_images": 2}
        mock_result.message = "Success"

        mock_visualizer = Mock()
        mock_visualizer.visualize.return_value = mock_result

        with patch('dataflow.visualize.labelme_visualizer.LabelMeVisualizer', return_value=mock_visualizer):
            result = runner.invoke(
                cli,
                [
                    "visualize",
                    "labelme",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--save",
                ]
            )

            # Check command executed successfully
            assert result.exit_code == 0
            # Visualizer should have been called
            mock_visualizer.visualize.assert_called_once()


def test_visualize_failure():
    """Test visualize command with mock failure."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test files
        input_dir = tmpdir_path / "input"
        input_dir.mkdir()

        class_file = tmpdir_path / "classes.txt"
        class_file.write_text("person\ncar\n")

        output_dir = tmpdir_path / "output"

        # Mock YOLOVisualizer with failure
        mock_result = Mock()
        mock_result.success = False
        mock_result.message = "Visualization failed: test error"

        mock_visualizer = Mock()
        mock_visualizer.visualize.return_value = mock_result

        with patch('dataflow.visualize.yolo_visualizer.YOLOVisualizer', return_value=mock_visualizer):
            result = runner.invoke(
                cli,
                [
                    "visualize",
                    "yolo",
                    str(input_dir),
                    "--class-file",
                    str(class_file),
                    "--output-dir",
                    str(output_dir),
                    "--save",
                ]
            )

            # Should fail with error
            assert result.exit_code != 0
            assert "失败" in result.output or "failed" in result.output