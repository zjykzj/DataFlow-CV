"""
Unit tests for yolo_and_coco.py
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from dataflow.convert.yolo_and_coco import YoloAndCocoConverter
from dataflow.label.base import AnnotationResult
from dataflow.label.models import (
    AnnotationFormat,
    BoundingBox,
    DatasetAnnotations,
    ImageAnnotation,
    ObjectAnnotation,
)


class TestYoloAndCocoConverter:
    """Test suite for YoloAndCocoConverter class."""

    def test_init_yolo_to_coco(self):
        """Test initialization for YOLO→COCO direction."""
        converter = YoloAndCocoConverter(source_to_target=True)
        assert converter.source_format == "yolo"
        assert converter.target_format == "coco"
        assert converter.source_to_target is True

    def test_init_coco_to_yolo(self):
        """Test initialization for COCO→YOLO direction."""
        converter = YoloAndCocoConverter(source_to_target=False)
        assert converter.source_format == "coco"
        assert converter.target_format == "yolo"
        assert converter.source_to_target is False

    def test_validate_inputs_yolo_to_coco_missing_class_file(self):
        """Test validation for YOLO→COCO with missing class file."""
        converter = YoloAndCocoConverter(source_to_target=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            # Missing class_file parameter
            kwargs = {"image_dir": str(Path(tmpdir) / "images")}
            assert not converter.validate_inputs(str(source_path), str(target_path), kwargs)

    def test_validate_inputs_yolo_to_coco_invalid_class_file(self):
        """Test validation for YOLO→COCO with non-existent class file."""
        converter = YoloAndCocoConverter(source_to_target=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            kwargs = {
                "class_file": "/nonexistent/classes.txt",
                "image_dir": str(Path(tmpdir) / "images"),
            }
            assert not converter.validate_inputs(str(source_path), str(target_path), kwargs)

    def test_validate_inputs_yolo_to_coco_missing_image_dir(self):
        """Test validation for YOLO→COCO with missing image_dir."""
        converter = YoloAndCocoConverter(source_to_target=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            # Create a dummy class file
            class_file = Path(tmpdir) / "classes.txt"
            class_file.write_text("cat\ndog\n")

            kwargs = {"class_file": str(class_file)}
            # Missing image_dir parameter
            assert not converter.validate_inputs(str(source_path), str(target_path), kwargs)

    def test_validate_inputs_yolo_to_coco_invalid_image_dir(self):
        """Test validation for YOLO→COCO with non-existent image_dir."""
        converter = YoloAndCocoConverter(source_to_target=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            # Create a dummy class file
            class_file = Path(tmpdir) / "classes.txt"
            class_file.write_text("cat\ndog\n")

            kwargs = {"class_file": str(class_file), "image_dir": "/nonexistent/images"}
            assert not converter.validate_inputs(str(source_path), str(target_path), kwargs)

    def test_validate_inputs_yolo_to_coco_valid(self):
        """Test validation for valid YOLO→COCO inputs."""
        converter = YoloAndCocoConverter(source_to_target=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "target"

            # Create dummy files
            class_file = Path(tmpdir) / "classes.txt"
            class_file.write_text("cat\ndog\n")

            image_dir = Path(tmpdir) / "images"
            image_dir.mkdir()

            kwargs = {"class_file": str(class_file), "image_dir": str(image_dir)}
            assert converter.validate_inputs(str(source_path), str(target_path), kwargs)

    def test_validate_inputs_coco_to_yolo_valid(self):
        """Test validation for valid COCO→YOLO inputs."""
        converter = YoloAndCocoConverter(source_to_target=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "coco.json"
            source_path.write_text("{}")  # Empty JSON
            target_path = Path(tmpdir) / "target"

            # class_file and image_dir are optional for COCO→YOLO
            kwargs = {}
            assert converter.validate_inputs(str(source_path), str(target_path), kwargs)

    @patch("dataflow.convert.yolo_and_coco.YoloAnnotationHandler")
    def test_create_source_handler_yolo_to_coco(self, mock_handler_class):
        """Test creating source handler for YOLO→COCO."""
        converter = YoloAndCocoConverter(source_to_target=True)

        mock_handler = Mock()
        mock_handler_class.return_value = mock_handler

        source_path = "/path/to/source"
        kwargs = {"class_file": "/path/to/classes.txt", "image_dir": "/path/to/images"}

        handler = converter.create_source_handler(source_path, kwargs)

        # Verify handler was created with correct parameters
        mock_handler_class.assert_called_once_with(
            label_dir=source_path,
            class_file=kwargs["class_file"],
            image_dir=kwargs["image_dir"],
            prediction=False,
            strict_mode=True,
            logger=converter.logger,
        )
        assert handler == mock_handler

    @patch("dataflow.convert.yolo_and_coco.CocoAnnotationHandler")
    def test_create_source_handler_coco_to_yolo(self, mock_handler_class):
        """Test creating source handler for COCO→YOLO."""
        converter = YoloAndCocoConverter(source_to_target=False)

        mock_handler = Mock()
        mock_handler_class.return_value = mock_handler

        source_path = "/path/to/coco.json"
        kwargs = {}

        handler = converter.create_source_handler(source_path, kwargs)

        # Verify handler was created with correct parameters
        mock_handler_class.assert_called_once_with(
            annotation_file=source_path,
            strict_mode=True,
            logger=converter.logger,
        )
        assert handler == mock_handler

    @patch("dataflow.convert.yolo_and_coco.CocoAnnotationHandler")
    def test_create_target_handler_yolo_to_coco(self, mock_handler_class):
        """Test creating target handler for YOLO→COCO."""
        converter = YoloAndCocoConverter(source_to_target=True)

        mock_handler = Mock()
        mock_handler_class.return_value = mock_handler

        target_path = "/path/to/coco.json"
        kwargs = {"do_rle": False}

        with patch.object(Path, "mkdir") as mock_mkdir:
            handler = converter.create_target_handler(target_path, kwargs)

            # Verify handler was created
            mock_handler_class.assert_called_once_with(
                annotation_file=target_path,
                logger=converter.logger,
                strict_mode=True,
                do_rle=False,
                prediction=False,
            )
            assert handler == mock_handler

            # Verify mkdir was called
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("dataflow.convert.yolo_and_coco.YoloAnnotationHandler")
    def test_create_target_handler_coco_to_yolo(self, mock_handler_class):
        """Test creating target handler for COCO→YOLO."""
        converter = YoloAndCocoConverter(source_to_target=False)

        mock_handler = Mock()
        mock_handler_class.return_value = mock_handler

        target_path = "/path/to/target"
        kwargs = {}

        with patch.object(Path, "mkdir") as mock_mkdir:
            handler = converter.create_target_handler(target_path, kwargs)

            # Verify handler was created
            assert handler == mock_handler

            # Verify directories were created
            assert mock_mkdir.call_count >= 3  # base, labels, images directories

    def test_convert_annotations_default(self):
        """Test convert_annotations returns correctly transformed data."""
        converter = YoloAndCocoConverter(source_to_target=True)

        # Create dummy annotations in YOLO format
        annotations = DatasetAnnotations(
            format=AnnotationFormat.YOLO,
            images=[
                ImageAnnotation(
                    image_id="test1",
                    image_path="/path/to/image.jpg",
                    width=200,
                    height=100,
                    objects=[
                        ObjectAnnotation(
                            class_id=0,
                            class_name="cat",
                            bbox=BoundingBox(x=0.5, y=0.5, width=0.2, height=0.2),
                        )
                    ],
                )
            ],
            categories={0: "cat", 1: "dog"},
        )

        # YOLO→COCO: convert normalized to absolute pixels
        result = converter.convert_annotations(annotations, {})
        assert result.format == AnnotationFormat.COCO
        assert len(result.images) == 1
        assert len(result.images[0].objects) == 1
        obj = result.images[0].objects[0]
        # Check converted bbox: cx=100, cy=50, w=40, h=20 → x_tl=80, y_tl=40
        assert obj.bbox.x == 80
        assert obj.bbox.y == 40
        assert obj.bbox.width == 40
        assert obj.bbox.height == 20

        # COCO→YOLO: convert absolute pixels to normalized
        converter_reverse = YoloAndCocoConverter(source_to_target=False)
        coco_anns = DatasetAnnotations(
            format=AnnotationFormat.COCO,
            images=[
                ImageAnnotation(
                    image_id="test1",
                    image_path="/path/to/image.jpg",
                    width=200,
                    height=100,
                    objects=[
                        ObjectAnnotation(
                            class_id=0,
                            class_name="cat",
                            bbox=BoundingBox(x=80, y=40, width=40, height=20),
                        )
                    ],
                )
            ],
            categories={0: "cat"},
        )
        result_rev = converter_reverse.convert_annotations(coco_anns, {})
        assert result_rev.format == AnnotationFormat.YOLO
        obj_rev = result_rev.images[0].objects[0]
        assert obj_rev.bbox.x == pytest.approx(0.5)
        assert obj_rev.bbox.y == pytest.approx(0.5)

    @patch("dataflow.convert.yolo_and_coco.YoloAnnotationHandler")
    @patch("dataflow.convert.yolo_and_coco.CocoAnnotationHandler")
    def test_convert_yolo_to_coco_mocked(self, mock_coco_handler_class, mock_yolo_handler_class):
        """Test YOLO→COCO conversion with mocked handlers."""
        converter = YoloAndCocoConverter(source_to_target=True)

        # Mock handlers
        mock_source_handler = Mock()
        mock_target_handler = Mock()

        mock_yolo_handler_class.return_value = mock_source_handler
        mock_coco_handler_class.return_value = mock_target_handler

        # Mock read result
        mock_annotations = Mock(spec=DatasetAnnotations)
        mock_annotations.images = []
        mock_annotations.categories = {}
        mock_read_result = Mock(spec=AnnotationResult)
        mock_read_result.success = True
        mock_read_result.data = mock_annotations
        mock_read_result.errors = []
        mock_source_handler.read.return_value = mock_read_result

        # Mock write result
        mock_write_result = Mock(spec=AnnotationResult)
        mock_write_result.success = True
        mock_write_result.errors = []
        mock_target_handler.write.return_value = mock_write_result

        # Run conversion
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source"
            source_path.mkdir()
            target_path = Path(tmpdir) / "coco.json"

            # Create dummy files
            class_file = Path(tmpdir) / "classes.txt"
            class_file.write_text("cat\ndog\n")

            image_dir = Path(tmpdir) / "images"
            image_dir.mkdir()

            kwargs = {
                "class_file": str(class_file),
                "image_dir": str(image_dir),
                "do_rle": False,
            }
            result = converter.convert(str(source_path), str(target_path), **kwargs)

        # Verify result
        assert result.success is True
        assert result.source_format == "yolo"
        assert result.target_format == "coco"

        # Verify handlers were called
        mock_source_handler.read.assert_called_once()
        # convert_annotations creates a new DatasetAnnotations object,
        # so we check that write was called once with any args
        assert mock_target_handler.write.call_count == 1
        args, _ = mock_target_handler.write.call_args
        assert args[1] == str(target_path)  # target path

    @patch("dataflow.convert.yolo_and_coco.CocoAnnotationHandler")
    @patch("dataflow.convert.yolo_and_coco.YoloAnnotationHandler")
    def test_convert_coco_to_yolo_mocked(self, mock_yolo_handler_class, mock_coco_handler_class):
        """Test COCO→YOLO conversion with mocked handlers (streaming)."""
        converter = YoloAndCocoConverter(source_to_target=False)

        # Mock handlers
        mock_source_handler = Mock()
        mock_target_handler = Mock()
        mock_target_handler.label_dir = "mock_labels_dir"

        mock_coco_handler_class.return_value = mock_source_handler
        mock_yolo_handler_class.return_value = mock_target_handler

        # Mock streaming source: empty iterator
        mock_source_handler.iter_images.return_value = iter([])
        # Categories extracted from COCO JSON by _ensure_categories_for_streaming
        # — source_path is "{}" (empty JSON), so no categories are extracted.
        # This is fine; the test verifies the streaming pipeline runs successfully
        # with an empty source.

        # Mock streaming target
        mock_write_result = Mock(spec=AnnotationResult)
        mock_write_result.success = True
        mock_write_result.errors = []
        mock_target_handler.write_one.return_value = mock_write_result

        # Run conversion
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "coco.json"
            source_path.write_text('{"images":[],"annotations":[],"categories":[]}')
            target_path = Path(tmpdir) / "target"

            kwargs = {}
            result = converter.convert(str(source_path), str(target_path), **kwargs)

        assert result.success is True
        assert result.source_format == "coco"
        assert result.target_format == "yolo"

        mock_source_handler.iter_images.assert_called_once()

    def test_converter_verbose_param(self):
        """Test converter verbose parameter."""
        from dataflow.util.logging import LogConfig

        # Test verbose=False (default)
        converter_no_verbose = YoloAndCocoConverter(source_to_target=True)
        assert converter_no_verbose._log_manager.log_path is None

        # Test verbose=True
        log_config = LogConfig(name="test_v", verbose=True)
        converter_verbose = YoloAndCocoConverter(source_to_target=True, log_config=log_config)
        assert hasattr(converter_verbose, "_log_manager")
        assert converter_verbose._log_manager.log_path is not None

        # Test verbose parameter in COCO→YOLO direction
        log_config2 = LogConfig(name="test_v2", verbose=True)
        converter_reverse = YoloAndCocoConverter(source_to_target=False, log_config=log_config2)
        assert converter_reverse._log_manager.log_path is not None

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test output."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_yolo_to_coco_prediction_output_is_list(self, temp_dir):
        """prediction=True should produce a plain JSON list."""
        pred_data_dir = (
            Path(__file__).parent.parent.parent / "assets" / "test_data" / "det" / "yolo_pred"
        )
        label_dir = str(pred_data_dir / "labels")
        class_file = str(pred_data_dir / "classes.txt")
        image_dir = str(
            Path(__file__).parent.parent.parent / "assets" / "test_data" / "det" / "yolo" / "images"
        )
        output_file = str(temp_dir / "pred.json")

        converter = YoloAndCocoConverter(source_to_target=True, prediction=True, strict_mode=False)
        result = converter.convert(
            label_dir, output_file, image_dir=image_dir, class_file=class_file
        )

        assert result.success is True, f"Conversion failed: {result.errors}"

        import json

        with open(output_file, "r") as f:
            data = json.load(f)
        assert isinstance(data, list), f"Expected list, got {type(data)}"
        assert len(data) > 0
        for ann in data:
            assert "score" in ann, f"Annotation missing 'score': {ann}"
            assert "image_id" in ann
            assert "category_id" in ann
            assert "bbox" in ann

    def test_yolo_to_coco_annotation_still_dict(self, temp_dir):
        """prediction=False should still produce a full COCO dict."""
        output_file = str(temp_dir / "anno.json")

        # Re-read labels WITHOUT prediction mode (will fail with 6-token lines)
        # So use standard 5-token test data for annotation mode
        std_data_dir = Path(__file__).parent.parent.parent / "assets" / "test_data" / "det" / "yolo"
        converter = YoloAndCocoConverter(source_to_target=True, prediction=False, strict_mode=False)
        result = converter.convert(
            str(std_data_dir / "labels"),
            output_file,
            image_dir=str(std_data_dir / "images"),
            class_file=str(std_data_dir / "classes.txt"),
        )

        assert result.success is True, f"Conversion failed: {result.errors}"

        import json

        with open(output_file, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data


class TestCocoToYoloOutputNaming:
    """Regression tests: COCO→YOLO output names derive from file_name stem."""

    def _coco_ann(self, image_id="1", image_path="000001.jpg"):
        """Build a single-image COCO DatasetAnnotations."""
        return DatasetAnnotations(
            format=AnnotationFormat.COCO,
            images=[
                ImageAnnotation(
                    image_id=image_id,
                    image_path=image_path,
                    width=200,
                    height=100,
                    objects=[
                        ObjectAnnotation(
                            class_id=0,
                            class_name="cat",
                            bbox=BoundingBox(x=80, y=40, width=40, height=20),
                        )
                    ],
                )
            ],
            categories={0: "cat"},
        )

    @pytest.mark.parametrize(
        "image_id, image_path, expected",
        [
            ("1", "000001.jpg", "000001"),  # leading zeros preserved
            ("2", "sub/000002.jpg", "000002"),  # subdirectory flattened
            ("3", "win\\000003.jpg", "000003"),  # backslash normalized
            ("4", "", "4"),  # empty file_name falls back to numeric id
        ],
    )
    def test_image_id_rewritten_to_file_stem(self, image_id, image_path, expected):
        """Test _coco_to_yolo_one rewrites image_id to the file_name stem."""
        converter = YoloAndCocoConverter(source_to_target=False)
        annotations = self._coco_ann(image_id=image_id, image_path=image_path)
        result = converter.convert_annotations(annotations, {})
        assert result.format == AnnotationFormat.YOLO
        assert len(result.images) == 1
        assert result.images[0].image_id == expected
        # image_path is preserved (relative path, not flattened)
        assert result.images[0].image_path == image_path

    def _write_coco_json(self, tmp_path, images):
        """Write a minimal COCO JSON with the given image entries."""
        import json

        coco = {
            "images": [
                {"id": i + 1, "file_name": fn, "width": 200, "height": 100}
                for i, fn in enumerate(images)
            ],
            "annotations": [
                {
                    "id": i + 1,
                    "image_id": i + 1,
                    "category_id": 1,
                    "bbox": [80, 40, 40, 20],
                    "area": 800,
                    "iscrowd": 0,
                }
                for i in range(len(images))
            ],
            "categories": [{"id": 1, "name": "cat"}],
        }
        src = tmp_path / "ann.json"
        src.write_text(json.dumps(coco), encoding="utf-8")
        return src

    def test_streaming_output_uses_file_name_stem(self, tmp_path):
        """End-to-end: zero-padded file_name → labels/000001.txt."""
        src = self._write_coco_json(tmp_path, ["000001.jpg"])
        out = tmp_path / "out"

        converter = YoloAndCocoConverter(source_to_target=False)
        result = converter.convert(str(src), str(out))

        assert result.success is True, f"Conversion failed: {result.errors}"
        label_files = sorted(p.name for p in (out / "labels").glob("*.txt"))
        assert label_files == ["000001.txt"]
        # Rename notice lands in result.warnings
        assert any("Output '000001': renamed 1 → 000001" in w for w in result.warnings)

    def test_subdir_flatten_warns_even_when_stem_unchanged(self, tmp_path):
        """file_name with subdirectory warns even if the stem equals the id."""
        src = self._write_coco_json(tmp_path, ["train2017/1.jpg"])
        out = tmp_path / "out"

        converter = YoloAndCocoConverter(source_to_target=False)
        result = converter.convert(str(src), str(out))

        assert result.success is True, f"Conversion failed: {result.errors}"
        label_files = sorted(p.name for p in (out / "labels").glob("*.txt"))
        assert label_files == ["1.txt"]  # stem == id, no rename
        # Structure information dropped — warning must still fire
        assert any("subdirectory flattened" in w for w in result.warnings)

    def test_duplicate_stem_warning(self, tmp_path):
        """Two images sharing a basename → one file + duplicate warning."""
        src = self._write_coco_json(tmp_path, ["a/000001.jpg", "b/000001.jpg"])
        out = tmp_path / "out"

        converter = YoloAndCocoConverter(source_to_target=False)
        result = converter.convert(str(src), str(out))

        assert result.success is True, f"Conversion failed: {result.errors}"
        label_files = sorted(p.name for p in (out / "labels").glob("*.txt"))
        assert label_files == ["000001.txt"]
        assert any("Duplicate output stem '000001'" in w for w in result.warnings)

    def test_roundtrip_preserves_stem(self, tmp_path):
        """YOLO→COCO→YOLO roundtrip keeps the original file stem."""
        # 1. Build a YOLO source: labels/000001.txt + images/000001.jpg (dummy)
        import cv2
        import numpy as np

        yolo_dir = tmp_path / "yolo"
        (yolo_dir / "labels").mkdir(parents=True)
        (yolo_dir / "images").mkdir(parents=True)
        (yolo_dir / "classes.txt").write_text("cat\n", encoding="utf-8")
        (yolo_dir / "labels" / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        cv2.imwrite(
            str(yolo_dir / "images" / "000001.jpg"),
            np.zeros((100, 200, 3), dtype=np.uint8),
        )

        coco_out = tmp_path / "roundtrip.json"
        to_coco = YoloAndCocoConverter(source_to_target=True, strict_mode=False)
        r1 = to_coco.convert(
            str(yolo_dir / "labels"),
            str(coco_out),
            image_dir=str(yolo_dir / "images"),
            class_file=str(yolo_dir / "classes.txt"),
        )
        assert r1.success is True, f"YOLO→COCO failed: {r1.errors}"

        # 2. Convert back: COCO file_name must be the original stem
        out = tmp_path / "out"
        to_yolo = YoloAndCocoConverter(source_to_target=False)
        r2 = to_yolo.convert(str(coco_out), str(out))
        assert r2.success is True, f"COCO→YOLO failed: {r2.errors}"
        label_files = sorted(p.name for p in (out / "labels").glob("*.txt"))
        assert label_files == ["000001.txt"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
class TestCocoToYoloMultiSegment:
    """Regression tests: multi-segment polygons merge into one outline (#47)."""

    def _write_coco(self, tmp_path, segmentation):
        import json

        coco = {
            "images": [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": segmentation,
                    "bbox": [10, 10, 440, 340],
                    "area": 2500.0,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "obj"}],
        }
        src = tmp_path / "ann.json"
        src.write_text(json.dumps(coco), encoding="utf-8")
        return src

    def _row_points(self, out):
        row = (out / "labels" / "img1.txt").read_text().strip()
        vals = [float(v) for v in row.split()[1:]]
        return [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]

    def test_two_disjoint_rects_merge_to_union(self, tmp_path):
        """Rasterizing the merged outline must reproduce the source union."""
        import cv2
        import numpy as np

        two_rects = [
            [10, 10, 60, 10, 60, 60, 10, 60],
            [400, 300, 450, 300, 450, 350, 400, 350],
        ]
        src = self._write_coco(tmp_path, two_rects)
        out = tmp_path / "out"

        r = YoloAndCocoConverter(source_to_target=False).convert(str(src), str(out))
        assert r.success is True, f"conversion failed: {r.errors}"

        pts = self._row_points(out)
        h, w = 480, 640
        src_mask = np.zeros((h, w), np.uint8)
        for seg in two_rects:
            cv2.fillPoly(src_mask, [np.array(seg, np.int32).reshape(-1, 1, 2)], 1)
        out_mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(
            out_mask,
            [np.array([(p[0] * w, p[1] * h) for p in pts], np.int32).reshape(-1, 1, 2)],
            1,
        )
        # No fabricated mask region and no lost source area beyond the
        # ~1-px bridge band: old concatenated output fabricated ~8750 px
        # and lost ~1430 px.
        assert int((out_mask & ~src_mask).sum()) < 600  # bridge band only
        assert int((src_mask & ~out_mask).sum()) == 0

    def test_single_segment_stays_bit_identical(self, tmp_path):
        """Single-segment annotations must be unchanged (no #1 regression)."""
        single = [[10, 10, 60, 10, 60, 60, 10, 60]]
        src = self._write_coco(tmp_path, single)
        out = tmp_path / "out"

        r = YoloAndCocoConverter(source_to_target=False).convert(str(src), str(out))
        assert r.success is True
        row = (out / "labels" / "img1.txt").read_text().strip()
        assert row == ("0 0.015625 0.020833 0.093750 0.020833 0.093750 0.125000 0.015625 0.125000")


