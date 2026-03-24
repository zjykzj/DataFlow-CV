"""
Unit tests for coco_handler.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dataflow.label.coco_handler import CocoAnnotationHandler
from dataflow.label.models import (BoundingBox, DatasetAnnotations,
                                   ImageAnnotation, ObjectAnnotation,
                                   Segmentation)


class TestCocoAnnotationHandler:
    """Test suite for CocoAnnotationHandler class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_coco_data(self, temp_dir):
        """Create sample COCO test data (standard polygon format)."""
        annotation_file = temp_dir / "annotations.json"

        # Create minimal COCO dataset
        coco_data = {
            "info": {
                "description": "Test dataset",
                "version": "1.0",
                "year": 2026,
                "contributor": "Test",
                "date_created": "2026-03-21",
            },
            "images": [
                {
                    "id": 1,
                    "width": 640,
                    "height": 480,
                    "file_name": "image1.jpg",
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": "",
                },
                {
                    "id": 2,
                    "width": 800,
                    "height": 600,
                    "file_name": "image2.jpg",
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": "",
                },
            ],
            "categories": [
                {"id": 1, "name": "person", "supercategory": "human"},
                {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [
                        [100, 100, 200, 100, 200, 200, 100, 200]
                    ],  # Polygon
                    "area": 10000.0,
                    "bbox": [100, 100, 100, 100],
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "segmentation": [
                        [300, 300, 350, 300, 350, 350, 300, 350]
                    ],  # Polygon
                    "area": 2500.0,
                    "bbox": [300, 300, 50, 50],
                    "iscrowd": 0,
                },
                {
                    "id": 3,
                    "image_id": 2,
                    "category_id": 1,
                    "segmentation": [],  # No segmentation (bbox only)
                    "area": 6000.0,
                    "bbox": [200, 200, 100, 60],
                    "iscrowd": 0,
                },
            ],
        }

        annotation_file.write_text(json.dumps(coco_data), encoding="utf-8")
        return str(annotation_file)

    @pytest.fixture
    def sample_coco_rle_data(self, temp_dir):
        """Create sample COCO test data with RLE format."""
        annotation_file = temp_dir / "annotations_rle.json"

        # Create COCO dataset with RLE segmentation
        # Note: We'll create a dummy RLE structure for testing
        # Actual RLE encoding requires pycocotools
        coco_data = {
            "info": {
                "description": "Test dataset with RLE",
                "version": "1.0",
                "year": 2026,
                "contributor": "Test",
                "date_created": "2026-03-21",
            },
            "images": [
                {
                    "id": 1,
                    "width": 640,
                    "height": 480,
                    "file_name": "image1.jpg",
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": "",
                }
            ],
            "categories": [{"id": 1, "name": "person", "supercategory": "human"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": {
                        "size": [480, 640],
                        "counts": "eNqztbW1BQ==",  # Dummy RLE data
                    },
                    "area": 10000.0,
                    "bbox": [100, 100, 100, 100],
                    "iscrowd": 1,  # RLE annotations are typically crowd annotations
                }
            ],
        }

        annotation_file.write_text(json.dumps(coco_data), encoding="utf-8")
        return str(annotation_file)

    def test_init(self):
        """Test handler initialization."""
        handler = CocoAnnotationHandler(
            annotation_file="/path/to/annotations.json", strict_mode=True
        )
        assert handler.annotation_file == Path("/path/to/annotations.json")
        assert handler.strict_mode is True
        assert handler.output_rle is False

    def test_read_success_polygon(self, sample_coco_data):
        """Test successful reading of COCO polygon format."""
        handler = CocoAnnotationHandler(
            annotation_file=sample_coco_data, strict_mode=True
        )

        result = handler.read()

        assert result.success is True
        assert result.data is not None
        assert len(result.data.images) == 2
        assert len(result.data.categories) == 2
        assert result.data.num_objects == 3

        # Check annotation flags
        assert handler.is_det is True  # Has bbox
        assert handler.is_seg is True  # Has segmentation
        assert handler.is_rle is False  # No RLE format

        # Check first image
        img1 = result.data.images[0]
        assert img1.image_id == "1"
        assert img1.width == 640
        assert img1.height == 480
        assert len(img1.objects) == 2

        # Check first object
        obj1 = img1.objects[0]
        assert obj1.class_id == 1
        assert obj1.class_name == "person"
        assert obj1.bbox is not None
        assert obj1.segmentation is not None
        assert len(obj1.segmentation.points) == 4  # Polygon with 4 points

        # Check normalized coordinates
        assert 0 <= obj1.bbox.x <= 1
        assert 0 <= obj1.bbox.y <= 1
        assert 0 <= obj1.bbox.width <= 1
        assert 0 <= obj1.bbox.height <= 1

    def test_read_success_rle(self, sample_coco_rle_data):
        """Test reading COCO RLE format (requires pycocotools)."""
        # Skip if pycocotools not available
        try:
            from pycocotools import mask as coco_mask
        except ImportError:
            pytest.skip("pycocotools not installed, skipping RLE tests")

        handler = CocoAnnotationHandler(
            annotation_file=sample_coco_rle_data,
            strict_mode=False,  # Use non-strict mode for dummy RLE data
        )

        result = handler.read()

        # Reading may fail due to invalid dummy RLE data
        # We just test that the handler detects RLE format
        assert handler.is_rle is True

    def test_rle_preservation(self, temp_dir):
        """Test that RLE data is preserved through read/write cycle."""
        try:
            import numpy as np
            from pycocotools import mask as coco_mask
        except ImportError:
            pytest.skip("pycocotools not installed, skipping RLE preservation test")

        # Create a simple binary mask (10x10) with a rectangle
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 3:7] = 1  # Rectangle from (3,2) to (6,7)
        # Encode to RLE
        rle = coco_mask.encode(np.asfortranarray(mask))
        # Convert counts to string for JSON serialization (as COCO does)
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("utf-8")

        # Create COCO data with two annotations: one crowd, one non-crowd
        coco_data = {
            "info": {"description": "RLE preservation test"},
            "images": [{"id": 1, "width": 10, "height": 10, "file_name": "test.png"}],
            "categories": [{"id": 1, "name": "object"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": rle,
                    "area": float(np.sum(mask)),
                    "bbox": [3, 2, 4, 6],  # x, y, w, h
                    "iscrowd": 1,  # Crowd annotation (should stay RLE)
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": rle,  # Same RLE but non-crowd
                    "area": float(np.sum(mask)),
                    "bbox": [0, 0, 2, 2],
                    "iscrowd": 0,  # Non-crowd, can be converted to polygon
                },
            ],
        }

        annotation_file = temp_dir / "rle_preservation.json"
        annotation_file.write_text(json.dumps(coco_data), encoding="utf-8")

        # Read with handler
        handler = CocoAnnotationHandler(
            annotation_file=str(annotation_file), strict_mode=False
        )
        result = handler.read()
        assert result.success is True
        assert handler.is_rle is True

        # Check that segmentation has RLE data for both objects
        img = result.data.images[0]
        assert len(img.objects) == 2

        # Sort by is_crowd to ensure consistent order
        crowd_obj = next(obj for obj in img.objects if obj.is_crowd)
        noncrowd_obj = next(obj for obj in img.objects if not obj.is_crowd)

        for obj in [crowd_obj, noncrowd_obj]:
            assert obj.segmentation is not None
            assert obj.segmentation.has_rle() is True
            assert obj.segmentation.rle is not None
            # Ensure counts match (original RLE counts)
            assert obj.segmentation.rle["counts"] == rle["counts"]
            assert obj.segmentation.rle["size"] == rle["size"]

        # Write back with RLE output (should use preserved RLE for both)
        output_file = temp_dir / "rle_preservation_output.json"
        write_result = handler.write(result.data, str(output_file), output_rle=True)
        assert write_result.success is True

        # Load written file and verify RLE counts unchanged
        with open(output_file, "r", encoding="utf-8") as f:
            written_data = json.load(f)

        assert len(written_data["annotations"]) == 2
        for ann in written_data["annotations"]:
            written_rle = ann["segmentation"]
            assert isinstance(written_rle, dict)
            assert "counts" in written_rle
            assert written_rle["counts"] == rle["counts"]
            assert written_rle["size"] == rle["size"]
            # iscrowd should match original
            if ann["id"] == 1:
                assert ann["iscrowd"] == 1
            else:
                assert ann["iscrowd"] == 0

        # Test writing with output_rle=False (crowd stays RLE, non-crowd becomes polygon)
        output_file2 = temp_dir / "rle_preservation_mixed.json"
        write_result2 = handler.write(result.data, str(output_file2), output_rle=False)
        assert write_result2.success is True
        with open(output_file2, "r", encoding="utf-8") as f:
            written_data2 = json.load(f)

        assert len(written_data2["annotations"]) == 2
        for ann in written_data2["annotations"]:
            if ann["id"] == 1:
                # Crowd annotation should remain RLE
                assert isinstance(ann["segmentation"], dict)
                assert "counts" in ann["segmentation"]
                assert ann["iscrowd"] == 1
            else:
                # Non-crowd annotation should be polygon list
                assert isinstance(ann["segmentation"], list)
                assert ann["iscrowd"] == 0
                # Polygon should have points (decoded from RLE)
                assert len(ann["segmentation"]) > 0

    def test_read_file_not_found(self):
        """Test reading non-existent file."""
        handler = CocoAnnotationHandler(
            annotation_file="/non/existent/file.json", strict_mode=True
        )

        result = handler.read()

        assert result.success is False
        assert "does not exist" in result.message or any(
            "does not exist" in e for e in result.errors
        )

    def test_read_invalid_json(self, temp_dir):
        """Test reading invalid JSON file."""
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("{invalid json}")

        handler = CocoAnnotationHandler(
            annotation_file=str(invalid_file), strict_mode=True
        )

        result = handler.read()

        assert result.success is False
        assert "Invalid JSON" in result.message or any(
            "Invalid JSON" in e for e in result.errors
        )

    def test_read_missing_required_fields(self, temp_dir):
        """Test reading COCO file missing required fields."""
        incomplete_file = temp_dir / "incomplete.json"
        incomplete_data = {
            "info": {"description": "Test"},
            # Missing images, annotations, categories
        }
        incomplete_file.write_text(json.dumps(incomplete_data))

        handler = CocoAnnotationHandler(
            annotation_file=str(incomplete_file), strict_mode=True
        )

        result = handler.read()

        assert result.success is False
        assert "Missing required field" in result.message or any(
            "Missing required field" in e for e in result.errors
        )

    def test_write_success_polygon(self, sample_coco_data, temp_dir):
        """Test successful writing of COCO polygon format."""
        # First read the sample data
        handler = CocoAnnotationHandler(
            annotation_file=sample_coco_data, strict_mode=True
        )
        read_result = handler.read()
        assert read_result.success is True

        # Write to new file
        output_file = temp_dir / "output.json"
        write_result = handler.write(read_result.data, str(output_file))

        assert write_result.success is True
        assert output_file.exists()

        # Verify the written file can be read back
        verify_handler = CocoAnnotationHandler(
            annotation_file=str(output_file), strict_mode=False
        )
        verify_result = verify_handler.read()

        assert verify_result.success is True
        assert len(verify_result.data.images) == len(read_result.data.images)
        assert len(verify_result.data.categories) == len(read_result.data.categories)
        assert verify_result.data.num_objects == read_result.data.num_objects

    def test_write_success_rle(self, sample_coco_data, temp_dir):
        """Test writing COCO RLE format (requires pycocotools)."""
        # Skip if pycocotools not available
        try:
            from pycocotools import mask as coco_mask
        except ImportError:
            pytest.skip("pycocotools not installed, skipping RLE write tests")

        # First read the sample data
        handler = CocoAnnotationHandler(
            annotation_file=sample_coco_data, strict_mode=True
        )
        read_result = handler.read()
        assert read_result.success is True

        # Write with RLE output
        output_file = temp_dir / "output_rle.json"
        write_result = handler.write(
            read_result.data, str(output_file), output_rle=True
        )

        assert write_result.success is True
        assert output_file.exists()

        # Load and check if RLE format is present
        with open(output_file, "r", encoding="utf-8") as f:
            written_data = json.load(f)

        # Check if any annotation has RLE segmentation
        has_rle = any(
            isinstance(ann.get("segmentation"), dict)
            and "counts" in ann.get("segmentation", {})
            for ann in written_data["annotations"]
        )
        assert has_rle, "RLE format not found in output"

    def test_write_with_crowd_annotation(self, temp_dir):
        """Test writing annotations with is_crowd flag."""
        # Create a simple dataset with crowd annotation
        dataset = DatasetAnnotations()
        dataset.add_category(1, "person")

        bbox = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.2)
        segmentation = Segmentation(
            points=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]
        )
        obj = ObjectAnnotation(
            class_id=1,
            class_name="person",
            bbox=bbox,
            segmentation=segmentation,
            is_crowd=True,
        )

        image = ImageAnnotation(
            image_id="1", image_path="image1.jpg", width=640, height=480, objects=[obj]
        )
        dataset.add_image(image)

        # Write with RLE output (crowd annotations should use RLE)
        handler = CocoAnnotationHandler(
            annotation_file=str(temp_dir / "dummy.json"), strict_mode=True
        )
        output_file = temp_dir / "output_crowd.json"
        write_result = handler.write(dataset, str(output_file), output_rle=True)

        assert write_result.success is True

        # Load and check iscrowd flag
        with open(output_file, "r", encoding="utf-8") as f:
            written_data = json.load(f)

        assert written_data["annotations"][0]["iscrowd"] == 1

    def test_validate_success(self, sample_coco_data):
        """Test successful validation of COCO file."""
        handler = CocoAnnotationHandler(
            annotation_file=sample_coco_data, strict_mode=True
        )

        assert handler.validate() is True

    def test_validate_failure(self, temp_dir):
        """Test validation failure for invalid COCO file."""
        invalid_file = temp_dir / "invalid.json"
        invalid_data = {"images": [], "annotations": [], "categories": []}
        invalid_file.write_text(json.dumps(invalid_data))

        handler = CocoAnnotationHandler(
            annotation_file=str(invalid_file), strict_mode=True
        )

        # Should pass basic structure validation
        assert handler.validate() is True

        # Test with missing required fields
        invalid_data2 = {"info": "test"}
        invalid_file2 = temp_dir / "invalid2.json"
        invalid_file2.write_text(json.dumps(invalid_data2))

        handler2 = CocoAnnotationHandler(
            annotation_file=str(invalid_file2), strict_mode=True
        )

        assert handler2.validate() is False

    def test_without_pycocotools(self, sample_coco_data, temp_dir):
        """Test behavior when pycocotools is not available."""
        # Mock HAS_COCO_MASK to False
        with patch("dataflow.label.coco_handler.HAS_COCO_MASK", False):
            # Test reading RLE data (should skip RLE annotations)
            handler = CocoAnnotationHandler(
                annotation_file=sample_coco_data, strict_mode=False
            )
            result = handler.read()
            assert result.success is True

            # Test writing with output_rle=True (should fall back to polygon)
            dataset = result.data
            output_file = temp_dir / "output_no_rle.json"
            write_result = handler.write(dataset, str(output_file), output_rle=True)
            assert write_result.success is True

    def test_object_to_coco_annotation_no_annotation(self, temp_dir):
        """Test conversion of object with neither bbox nor segmentation."""
        handler = CocoAnnotationHandler(
            annotation_file=str(temp_dir / "dummy.json"), strict_mode=False
        )

        # Create object without bbox or segmentation (should be invalid)
        # This should raise ValueError in ObjectAnnotation.__post_init__
        with pytest.raises(
            ValueError, match="At least one of bbox or segmentation must be provided"
        ):
            obj = ObjectAnnotation(
                class_id=1, class_name="person", bbox=None, segmentation=None
            )

    def test_detect_rle_format(self):
        """Test RLE format detection."""
        handler = CocoAnnotationHandler(annotation_file="/dummy.json", strict_mode=True)

        # Test with polygon annotations
        polygon_anns = [
            {"segmentation": [[100, 100, 200, 100, 200, 200]]},
            {"segmentation": []},
        ]
        assert handler._detect_rle_format(polygon_anns) is False

        # Test with RLE annotation
        rle_anns = [
            {"segmentation": {"size": [100, 100], "counts": "abc"}},
            {"segmentation": [[100, 100, 200, 100]]},
        ]
        assert handler._detect_rle_format(rle_anns) is True

        # Test with no segmentation
        no_seg_anns = [{"bbox": [0, 0, 10, 10]}]
        assert handler._detect_rle_format(no_seg_anns) is False

    def test_parse_polygon_segmentation(self):
        """Test polygon segmentation parsing."""
        handler = CocoAnnotationHandler(annotation_file="/dummy.json", strict_mode=True)

        # Valid polygon
        seg_data = [[100, 100, 200, 100, 200, 200, 100, 200]]
        points = handler._parse_polygon_segmentation(seg_data, 640, 480)
        assert len(points) == 4
        for x, y in points:
            assert 0 <= x <= 1
            assert 0 <= y <= 1

        # Multiple polygons
        seg_data_multi = [
            [100, 100, 200, 100, 200, 200],
            [300, 300, 350, 300, 350, 350],
        ]
        points_multi = handler._parse_polygon_segmentation(seg_data_multi, 640, 480)
        assert len(points_multi) == 6

        # Odd number of coordinates (should be skipped with warning)
        seg_data_odd = [[100, 100, 200, 100, 200]]
        points_odd = handler._parse_polygon_segmentation(seg_data_odd, 640, 480)
        assert len(points_odd) == 0

    def test_output_rle_flag(self, sample_coco_data, temp_dir):
        """Test output_rle flag behavior."""
        handler = CocoAnnotationHandler(
            annotation_file=sample_coco_data, strict_mode=True
        )

        # Read data (sets output_rle to is_rle which is False)
        result = handler.read()
        assert result.success is True
        assert handler.output_rle is False

        # Change output_rle
        handler.output_rle = True
        assert handler.output_rle is True

        # Write with explicit output_rle parameter
        output_file = temp_dir / "output_test.json"
        write_result = handler.write(result.data, str(output_file), output_rle=False)
        assert write_result.success is True
        # Should restore original output_rle after write
        assert (
            handler.output_rle is True
        )  # Actually write method saves and restores, so remains True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
