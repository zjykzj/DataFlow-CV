"""Tests for evaluate utilities."""

import json
from pathlib import Path

import pytest

from dataflow.evaluate.utils import (
    HAS_COCO,
    _create_coco_from_dict,
    _dataset_to_coco_dict,
    _extract_stats,
    _load_coco,
    _load_dt,
    _validate_coco_available,
    _validate_coco_dict,
    _validate_dt_scores,
    format_metric_table,
    format_per_class_table,
    format_prf1_output,
)
from dataflow.evaluate.result import (
    EvaluationMetrics,
    PerClassMetrics,
    PRF1Result,
    PRF1Values,
)

TEST_DATA = Path(__file__).parent.parent.parent / "assets" / "test_data" / "evaluate"


@pytest.fixture
def gt_path():
    return TEST_DATA / "gt_coco.json"


@pytest.fixture
def dt_path():
    return TEST_DATA / "dt_coco.json"


@pytest.fixture
def gt_dict():
    with open(TEST_DATA / "gt_coco.json") as f:
        return json.load(f)


@pytest.fixture
def dt_dict():
    with open(TEST_DATA / "dt_coco.json") as f:
        return json.load(f)


@pytest.fixture
def dt_list_path():
    return TEST_DATA / "dt_list.json"


@pytest.fixture
def dt_list():
    with open(TEST_DATA / "dt_list.json") as f:
        return json.load(f)


class TestCocoAvailability:
    """Test pycocotools guard."""

    def test_has_coco(self):
        assert HAS_COCO is True

    def test_validate_available(self):
        _validate_coco_available()  # Should not raise


class TestValidateCocoDict:
    """Test COCO dict validation."""

    def test_valid(self, gt_dict):
        _validate_coco_dict(gt_dict)  # Should not raise

    def test_missing_key(self):
        with pytest.raises(ValueError, match="missing required keys"):
            _validate_coco_dict({"images": [], "categories": []})

    def test_missing_images(self):
        with pytest.raises(ValueError, match="images"):
            _validate_coco_dict({"annotations": [], "categories": []})


class TestLoadCoco:
    """Test _load_coco input normalization."""

    def test_load_from_path(self, gt_path):
        coco = _load_coco(gt_path)
        assert len(coco.getImgIds()) == 2
        assert len(coco.getCatIds()) == 2

    def test_load_from_str(self, gt_path):
        coco = _load_coco(str(gt_path))
        assert len(coco.getImgIds()) == 2

    def test_load_from_dict(self, gt_dict):
        coco = _load_coco(gt_dict)
        assert len(coco.getImgIds()) == 2

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            _load_coco("/nonexistent/path.json")

    def test_load_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported"):
            _load_coco([1, 2, 3])


class TestLoadDt:
    """Test _load_dt input normalization (list-format DT support)."""

    def test_load_from_list_path(self, gt_path, dt_list_path):
        coco_gt = _load_coco(gt_path)
        coco_dt = _load_dt(dt_list_path, coco_gt)
        assert len(coco_dt.getAnnIds()) == 5
        # DT inherits images and categories from GT
        assert len(coco_dt.getImgIds()) == 2
        assert len(coco_dt.getCatIds()) == 2

    def test_load_from_list_in_memory(self, gt_path, dt_list):
        coco_gt = _load_coco(gt_path)
        coco_dt = _load_dt(dt_list, coco_gt)
        assert len(coco_dt.getAnnIds()) == 5
        assert len(coco_dt.getImgIds()) == 2

    def test_load_from_dict_path(self, gt_path, dt_path):
        """Dict-format DT should still work via _load_dt."""
        coco_gt = _load_coco(gt_path)
        coco_dt = _load_dt(dt_path, coco_gt)
        assert len(coco_dt.getAnnIds()) == 5
        assert len(coco_dt.getImgIds()) == 2

    def test_load_from_dict_in_memory(self, gt_path, dt_dict):
        coco_gt = _load_coco(gt_path)
        coco_dt = _load_dt(dt_dict, coco_gt)
        assert len(coco_dt.getAnnIds()) == 5

    def test_nonexistent_file(self, gt_path):
        coco_gt = _load_coco(gt_path)
        with pytest.raises(FileNotFoundError, match="DT file not found"):
            _load_dt("/nonexistent/dt.json", coco_gt)

    def test_unsupported_type(self, gt_path):
        coco_gt = _load_coco(gt_path)
        with pytest.raises(ValueError, match="Unsupported DT source"):
            _load_dt(42, coco_gt)

    def test_list_copies_gt_categories(self, gt_path, dt_list_path):
        coco_gt = _load_coco(gt_path)
        coco_dt = _load_dt(dt_list_path, coco_gt)
        gt_cats = {c["id"]: c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
        dt_cats = {c["id"]: c["name"] for c in coco_dt.loadCats(coco_dt.getCatIds())}
        assert dt_cats == gt_cats

    def test_list_copies_gt_images(self, gt_path, dt_list_path):
        coco_gt = _load_coco(gt_path)
        coco_dt = _load_dt(dt_list_path, coco_gt)
        assert coco_dt.getImgIds() == coco_gt.getImgIds()

    def test_list_annotations_have_scores(self, gt_path, dt_list_path):
        coco_gt = _load_coco(gt_path)
        coco_dt = _load_dt(dt_list_path, coco_gt)
        for ann in coco_dt.loadAnns(coco_dt.getAnnIds()):
            assert "score" in ann


class TestCreateCocoFromDict:
    """Test in-memory COCO construction."""

    def test_valid(self, gt_dict):
        coco = _create_coco_from_dict(gt_dict)
        assert len(coco.getImgIds()) == 2
        assert len(coco.getAnnIds()) == 4

    def test_preserves_data(self, gt_dict):
        coco = _create_coco_from_dict(gt_dict)
        cats = coco.loadCats(coco.getCatIds())
        names = {c["name"] for c in cats}
        assert "cat" in names
        assert "dog" in names


class TestExtractStats:
    """Test statistics extraction."""

    def test_stats(self, gt_path, dt_path):
        coco_gt = _load_coco(gt_path)
        coco_dt = _load_coco(dt_path)
        gt_stats, dt_stats = _extract_stats(coco_gt, coco_dt)
        assert gt_stats["images"] == 2
        assert gt_stats["annotations"] == 4
        assert gt_stats["categories"] == 2
        assert dt_stats["images"] == 2
        assert dt_stats["annotations"] == 5


class TestValidateDtScores:
    """Test DT score validation."""

    def test_all_have_scores(self, dt_path):
        coco_dt = _load_coco(dt_path)
        # Returns None on success (raises ValueError on failure)
        _validate_dt_scores(coco_dt)

    def test_missing_scores(self):
        data = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}],
            "categories": [{"id": 1, "name": "cat"}],
        }
        coco = _load_coco(data)
        with pytest.raises(ValueError, match="missing 'score'"):
            _validate_dt_scores(coco)


class TestFormatMetricTable:
    """Test metric table formatting."""

    def test_format(self):
        metrics = EvaluationMetrics(
            ap=0.352,
            ap50=0.568,
            ap75=0.371,
            ap_small=0.152,
            ap_medium=0.389,
            ap_large=0.524,
            ar_max_1=0.289,
            ar_max_10=0.452,
            ar_max_100=0.467,
            ar_small=0.213,
            ar_medium=0.501,
            ar_large=0.689,
        )
        output = format_metric_table(metrics)
        assert "0.352" in output
        assert "0.568" in output
        assert "AP" in output
        assert "AR" in output
        # Should have 12 lines
        lines = output.strip().split("\n")
        assert len(lines) == 12


class TestFormatPerClassTable:
    """Test per-class table formatting."""

    def test_empty(self):
        assert format_per_class_table({}) == ""

    def test_with_data(self):
        per_class = {
            1: PerClassMetrics(
                class_id=1,
                class_name="cat",
                gt_count=10,
                dt_count=12,
                tp=8,
                fp=4,
                fn=2,
                ap=0.45,
                ap50=0.68,
                ap75=0.42,
                precision=0.667,
                recall=0.8,
                f1_score=0.727,
            ),
        }
        output = format_per_class_table(per_class)
        assert "cat" in output
        assert "0.450" in output
        assert "0.680" in output


class TestFormatPRF1Output:
    """Test PRF1 output formatting."""

    def test_success(self):
        overall = PRF1Values(precision=0.8, recall=0.9, f1_score=0.847, tp=8, fp=2, fn=1)
        result = PRF1Result(
            success=True, iou_threshold=0.5, confidence_threshold=0.0, overall=overall
        )
        output = format_prf1_output(result)
        assert "P=0.800" in output
        assert "R=0.900" in output
        assert "F1=0.847" in output
        assert "TP=8" in output

    def test_failed(self):
        result = PRF1Result(success=False)
        assert "failed" in format_prf1_output(result).lower()


class TestDatasetToCocoDictImageIds:
    """Tests for _dataset_to_coco_dict image id resolution."""

    def _dataset(self, image_ids):
        from dataflow.label.models import (
            AnnotationFormat,
            BoundingBox,
            DatasetAnnotations,
            ImageAnnotation,
            ObjectAnnotation,
        )

        return DatasetAnnotations(
            format=AnnotationFormat.COCO,
            images=[
                ImageAnnotation(
                    image_id=image_id,
                    image_path=f"{image_id}.jpg",
                    width=10,
                    height=10,
                    objects=[
                        ObjectAnnotation(
                            class_id=1,
                            class_name="cat",
                            bbox=BoundingBox(x=0, y=0, width=5, height=5),
                        )
                    ],
                )
                for image_id in image_ids
            ],
            categories={1: "cat"},
        )

    def test_non_digit_ids_get_counter(self):
        """Non-digit image_ids resolve to int counter ids (never strings)."""
        result = _dataset_to_coco_dict(self._dataset(["frame_a", "frame_b"]))
        ids = [img["id"] for img in result["images"]]
        assert ids == [1, 2]
        assert result["annotations"][0]["image_id"] == 1
        assert result["annotations"][1]["image_id"] == 2

    def test_colliding_ids_stay_unique(self):
        """Colliding digit ids ('01' vs '1') resolve to unique int ids."""
        result = _dataset_to_coco_dict(self._dataset(["01", "1"]))
        ids = [img["id"] for img in result["images"]]
        assert ids[0] == 1
        assert len(set(ids)) == 2
        assert all(isinstance(i, int) for i in ids)

    def test_non_coco_format_raises(self):
        """Non-COCO DatasetAnnotations raise ValueError."""
        from dataflow.label.models import AnnotationFormat, DatasetAnnotations

        with pytest.raises(ValueError, match="format must be COCO"):
            _dataset_to_coco_dict(DatasetAnnotations(format=AnnotationFormat.YOLO))
