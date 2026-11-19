"""
Unit tests for pose (keypoint) visualization.

Covers the P2 contracts (spec_visualize.md §2.1/§2.3/§4.3): keypoint +
skeleton conversion to RenderData, skeleton sources per format, drawing
behavior, and bbox-from-keypoints fallback.
"""

import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from dataflow.visualize import COCOVisualizer, LabelMeVisualizer, YOLOVisualizer

KPT_NAMES = ["nose", "left_eye", "right_eye"]
SKELETON = [[0, 1], [0, 2]]


def _write_kpts_config(tmp_path: Path, with_skeleton: bool = True) -> Path:
    config = tmp_path / "kpts.yaml"
    lines = ["keypoints:", "  - nose", "  - left_eye", "  - right_eye"]
    if with_skeleton:
        lines += ["skeleton:", "  - [0, 1]", "  - [0, 2]"]
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config


class TestYOLOVisualizerPose:
    """YOLO pose visualizer (spec_visualize.md §5.1)."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp(prefix="test_pose_viz_yolo_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_pose_data(self, temp_dir):
        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\n")
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        label_dir = temp_dir / "labels"
        label_dir.mkdir()
        img = np.zeros((100, 200, 3), dtype=np.uint8)  # 200x100 image
        cv2.imwrite(str(image_dir / "test.jpg"), img)
        # 3-keypoint pose row (14 tokens): kpt2 v=0
        (label_dir / "test.txt").write_text("0 0.5 0.5 0.2 0.2 0.48 0.45 2 0.52 0.45 1 0.9 0.9 0\n")
        return {
            "class_file": class_file,
            "image_dir": image_dir,
            "label_dir": label_dir,
        }

    def _make_visualizer(self, data, kpts_config=None):
        return YOLOVisualizer(
            label_dir=str(data["label_dir"]),
            image_dir=str(data["image_dir"]),
            class_file=str(data["class_file"]),
            pose=True,
            kpts_config=str(kpts_config) if kpts_config else None,
            is_show=False,
            is_save=False,
        )

    def test_convert_keypoints_with_skeleton(self, sample_pose_data, temp_dir):
        visualizer = self._make_visualizer(sample_pose_data, _write_kpts_config(temp_dir))
        handler = visualizer._create_handler()
        image_ann = next(handler.iter_images())
        render_data = visualizer._convert_to_render_data(image_ann)

        assert len(render_data.annotations) == 1
        render_ann = render_data.annotations[0]
        assert render_ann.keypoints == [
            (int(0.48 * 200), int(0.45 * 100), 2),  # (96, 45, 2)
            (int(0.52 * 200), int(0.45 * 100), 1),  # (104, 45, 1)
            (0, 0, 0),  # v=0 normalized to (0, 0)
        ]
        assert render_ann.skeleton == [(0, 1), (0, 2)]
        # bbox present from the pose row
        assert render_ann.bbox is not None

    def test_convert_keypoints_without_config(self, sample_pose_data):
        visualizer = self._make_visualizer(sample_pose_data)
        handler = visualizer._create_handler()
        image_ann = next(handler.iter_images())
        render_data = visualizer._convert_to_render_data(image_ann)
        render_ann = render_data.annotations[0]
        assert render_ann.keypoints is not None
        assert render_ann.skeleton is None

    def test_bbox_from_keypoints_fallback(self, sample_pose_data):
        visualizer = self._make_visualizer(sample_pose_data)
        handler = visualizer._create_handler()
        image_ann = next(handler.iter_images())
        obj = image_ann.objects[0]
        obj.bbox = None
        render_data = visualizer._convert_to_render_data(image_ann)
        render_ann = render_data.annotations[0]
        # Labeled point extent: x 96-104, y 45-45 (v=0 point excluded)
        assert render_ann.bbox == (96, 45, 104, 45)


class TestCOCOVisualizerPose:
    """COCO pose visualizer (spec_visualize.md §5.2)."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp(prefix="test_pose_viz_coco_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_coco_pose(self, temp_dir):
        coco_file = temp_dir / "pose.json"
        data = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 200, "height": 100}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [50.0, 20.0, 60.0, 80.0],
                    "segmentation": [],
                    "area": 4800.0,
                    "iscrowd": 0,
                    "keypoints": [96, 45, 2, 104, 45, 1, 0, 0, 0],
                    "num_keypoints": 2,
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "keypoints": KPT_NAMES,
                    "skeleton": SKELETON,
                }
            ],
        }
        coco_file.write_text(json.dumps(data))
        return coco_file

    def test_convert_keypoints_with_categories_skeleton(self, sample_coco_pose, temp_dir):
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        visualizer = COCOVisualizer(
            annotation_file=str(sample_coco_pose),
            image_dir=str(image_dir),
            is_show=False,
            is_save=False,
        )
        handler = visualizer._create_handler()
        # visualize() stores the handler on the visualizer — simulate it
        visualizer._handler = handler
        image_ann = next(handler.iter_images())
        render_data = visualizer._convert_to_render_data(image_ann)

        render_ann = render_data.annotations[0]
        assert render_ann.keypoints == [(96, 45, 2), (104, 45, 1), (0, 0, 0)]
        assert render_ann.skeleton == [(0, 1), (0, 2)]

    def test_convert_keypoints_without_meta(self, sample_coco_pose, temp_dir):
        # Strip skeleton from categories — keypoints drawn without lines
        data = json.loads(sample_coco_pose.read_text(encoding="utf-8"))
        del data["categories"][0]["skeleton"]
        stripped = temp_dir / "stripped.json"
        stripped.write_text(json.dumps(data))
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        visualizer = COCOVisualizer(
            annotation_file=str(stripped),
            image_dir=str(image_dir),
            is_show=False,
            is_save=False,
        )
        handler = visualizer._create_handler()
        visualizer._handler = handler
        image_ann = next(handler.iter_images())
        render_data = visualizer._convert_to_render_data(image_ann)
        assert render_data.annotations[0].skeleton is None


class TestLabelMeVisualizerPose:
    """LabelMe pose visualizer (spec_visualize.md §5.3)."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp(prefix="test_pose_viz_labelme_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_pose_data(self, temp_dir):
        label_dir = temp_dir / "labels"
        label_dir.mkdir()
        data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [
                {
                    "label": "person",
                    "points": [[50, 20], [110, 100]],
                    "group_id": 1,
                    "shape_type": "rectangle",
                    "flags": {},
                },
                {
                    "label": "nose",
                    "points": [[96, 45]],
                    "group_id": 1,
                    "shape_type": "point",
                    "flags": {},
                },
                {
                    "label": "left_eye",
                    "points": [[104, 45]],
                    "group_id": 1,
                    "shape_type": "point",
                    "flags": {},
                },
            ],
            "imagePath": "test.jpg",
            "imageData": None,
            "imageHeight": 100,
            "imageWidth": 200,
        }
        (label_dir / "test.json").write_text(json.dumps(data))
        return {"label_dir": label_dir}

    def test_convert_keypoints_with_skeleton(self, sample_pose_data, temp_dir):
        visualizer = LabelMeVisualizer(
            label_dir=str(sample_pose_data["label_dir"]),
            image_dir=str(temp_dir),
            kpts_config=str(_write_kpts_config(temp_dir)),
            is_show=False,
            is_save=False,
        )
        handler = visualizer._create_handler()
        image_ann = next(handler.iter_images())
        render_data = visualizer._convert_to_render_data(image_ann)

        render_ann = render_data.annotations[0]
        # Config order: nose, left_eye, right_eye (missing → v=0 at (0,0))
        assert render_ann.keypoints == [(96, 45, 2), (104, 45, 2), (0, 0, 0)]
        assert render_ann.skeleton == [(0, 1), (0, 2)]


class TestDrawKeypoints:
    """Keypoint drawing behavior (spec_visualize.md §4.3)."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp(prefix="test_pose_draw_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def _make_visualizer(self, temp_dir):
        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\n")
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        label_dir = temp_dir / "labels"
        label_dir.mkdir()
        return YOLOVisualizer(
            label_dir=str(label_dir),
            image_dir=str(image_dir),
            class_file=str(class_file),
            pose=True,
            is_show=False,
            is_save=False,
        )

    def test_draw_keypoints_visibility_colors(self, temp_dir):
        visualizer = self._make_visualizer(temp_dir)
        image = np.zeros((60, 60, 3), dtype=np.uint8)

        # Points only (no skeleton) — deterministic visibility assertions
        keypoints = [(10, 10, 2), (30, 10, 1), (50, 10, 0)]
        visualizer._draw_keypoints(image, keypoints, None, (0, 0, 0))

        # v=2: filled circle at center → visible color (green)
        assert tuple(image[10, 10]) == visualizer.config["kpt_visible_color"]
        # v=1: hollow circle — center stays background, ring takes the
        # occluded color somewhere in the ring neighborhood
        # (keypoint at (x=30, y=10) → row 10, col 30)
        assert tuple(image[10, 30]) == (0, 0, 0)
        ring = image[10 - 4 : 10 + 5, 30 - 4 : 30 + 5].reshape(-1, 3)
        assert any(tuple(px) == visualizer.config["kpt_occluded_color"] for px in ring)
        # v=0: skipped entirely (keypoint at (x=50, y=10) → row 10, col 50)
        assert np.all(image[10 - 3 : 10 + 4, 50 - 3 : 50 + 4] == 0)

    def test_draw_keypoints_skeleton_line(self, temp_dir):
        visualizer = self._make_visualizer(temp_dir)
        image = np.zeros((60, 60, 3), dtype=np.uint8)

        class_color = visualizer.color_manager.get_color(0)
        # Skeleton line between (10,10) and (30,10) in the class color
        visualizer._draw_keypoints(image, [(10, 10, 2), (30, 10, 1)], [(0, 1)], class_color)
        # Mid-line pixel (x=20, y=10) carries the class color — clear of the
        # v=2 circle (x ≤ 13) and the v=1 ring (x ≥ 27)
        assert tuple(image[10, 20]) == class_color

    def test_draw_keypoints_out_of_range_edges_skipped(self, temp_dir):
        visualizer = self._make_visualizer(temp_dir)
        image = np.zeros((60, 60, 3), dtype=np.uint8)
        keypoints = [(10, 10, 2)]
        skeleton = [(0, 1), (5, 6)]  # both out of range for k=1
        visualizer._draw_keypoints(
            image, keypoints, skeleton, visualizer.color_manager.get_color(0)
        )
        # No crash; only the v=2 point drawn
        assert tuple(image[10, 10]) == visualizer.config["kpt_visible_color"]
        assert np.count_nonzero(image) > 0


class TestVisualizeCLIPoseFlags:
    """--pose / --kpts-config CLI options (spec_cli.md §3.2)."""

    def test_yolo_help_shows_pose_options(self):
        from click.testing import CliRunner

        from dataflow.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "yolo", "--help"])
        assert result.exit_code == 0
        assert "--pose" in result.output
        assert "--kpts-config" in result.output

    def test_labelme_help_shows_kpts_config(self):
        from click.testing import CliRunner

        from dataflow.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "labelme", "--help"])
        assert result.exit_code == 0
        assert "--kpts-config" in result.output
