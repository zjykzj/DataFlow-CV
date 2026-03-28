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
    assert "Visualization command group" in result.output


def test_visualize_yolo_help():
    """Test that visualize yolo subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["visualize", "yolo", "--help"])
    assert result.exit_code == 0
    assert "Visualize YOLO format labels" in result.output


def test_visualize_coco_help():
    """Test that visualize coco subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["visualize", "coco", "--help"])
    assert result.exit_code == 0
    assert "Visualize COCO format labels" in result.output


def test_visualize_labelme_help():
    """Test that visualize labelme subcommand help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["visualize", "labelme", "--help"])
    assert result.exit_code == 0
    assert "Visualize LabelMe format labels" in result.output


def test_visualize_yolo_success():
    """Test visualize yolo command with mock success."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test files
        image_dir = tmpdir_path / "images"
        image_dir.mkdir()
        label_dir = tmpdir_path / "labels"
        label_dir.mkdir()
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
                    str(image_dir),
                    str(label_dir),
                    str(class_file),
                    "--save",
                    str(output_dir),
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
        image_dir = tmpdir_path / "images"
        image_dir.mkdir()
        label_dir = tmpdir_path / "labels"
        label_dir.mkdir()
        # No class file created

        # Run command without required class file (positional argument)
        result = runner.invoke(
            cli,
            [
                "visualize",
                "yolo",
                str(image_dir),
                str(label_dir),
                # Missing class_file argument
            ]
        )

        # Should fail with error about missing class file
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "required" in result.output or "class-file" in result.output


def test_visualize_coco_success():
    """Test visualize coco command with mock success."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test files
        image_dir = tmpdir_path / "images"
        image_dir.mkdir()
        coco_file = tmpdir_path / "annotations.json"
        coco_file.write_text('{"images": [], "annotations": []}')

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
                    str(image_dir),
                    str(coco_file),
                    "--save",
                    str(output_dir),
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
        image_dir = tmpdir_path / "images"
        image_dir.mkdir()
        label_dir = tmpdir_path / "labels"
        label_dir.mkdir()

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
                    str(image_dir),
                    str(label_dir),
                    "--save",
                    str(output_dir),
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
        image_dir = tmpdir_path / "images"
        image_dir.mkdir()
        label_dir = tmpdir_path / "labels"
        label_dir.mkdir()
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
                    str(image_dir),
                    str(label_dir),
                    str(class_file),
                    "--save",
                    str(output_dir),
                ]
            )

            # Should fail with error
            assert result.exit_code != 0
            assert "failed" in result.output