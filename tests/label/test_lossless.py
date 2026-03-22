"""
Unit tests for lossless read/write functionality.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
import filecmp
import hashlib

from dataflow.label.labelme_handler import LabelMeAnnotationHandler
from dataflow.label.yolo_handler import YoloAnnotationHandler
from dataflow.label.coco_handler import CocoAnnotationHandler
from dataflow.label.models import DatasetAnnotations, OriginalData, AnnotationFormat


class TestLosslessFunctionality:
    """Test suite for lossless read/write functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_labelme_dir(self, temp_dir):
        """Create sample LabelMe directory with JSON files."""
        # Create image file (empty)
        image_file = temp_dir / "test.jpg"
        image_file.write_bytes(b"")  # Empty file for testing

        # Create LabelMe JSON file
        json_file = temp_dir / "test.json"
        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [
                {
                    "label": "cat",
                    "points": [[10.5, 20.5], [30.5, 40.5]],  # Float coordinates
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                },
                {
                    "label": "dog",
                    "points": [[50.5, 60.5], [70.5, 80.5], [80.5, 90.5], [60.5, 70.5]],
                    "group_id": 1,
                    "shape_type": "polygon",
                    "flags": {"occluded": True}
                }
            ],
            "imagePath": "test.jpg",
            "imageData": None,
            "imageHeight": 100,
            "imageWidth": 200
        }
        json_file.write_text(json.dumps(labelme_data, indent=2))
        return temp_dir

    @pytest.fixture
    def sample_yolo_dir(self, temp_dir):
        """Create sample YOLO directory with label files."""
        # Create class file
        class_file = temp_dir / "classes.txt"
        class_file.write_text("cat\ndog\nbird\n")

        # Create label file for detection
        label_file = temp_dir / "image1.txt"
        label_file.write_text("0 0.25 0.375 0.1 0.15\n1 0.55 0.66 0.12 0.18\n")  # cat and dog

        # Create label file for segmentation
        label_file2 = temp_dir / "image2.txt"
        label_file2.write_text("2 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8\n")  # bird with polygon

        # Create image files (small dummy images)
        try:
            from PIL import Image
            # Create 100x100 black JPEG images
            img = Image.new('RGB', (100, 100), color='black')
            img.save(temp_dir / "image1.jpg", format='JPEG')
            img.save(temp_dir / "image2.jpg", format='JPEG')
        except ImportError:
            # Fallback: create minimal valid JPEG headers (not real images)
            # This may still fail but better than empty files
            jpeg_header = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01'
            (temp_dir / "image1.jpg").write_bytes(jpeg_header)
            (temp_dir / "image2.jpg").write_bytes(jpeg_header)

        return temp_dir

    @pytest.fixture
    def sample_coco_file(self, temp_dir):
        """Create sample COCO JSON file."""
        coco_file = temp_dir / "annotations.json"
        coco_data = {
            "info": {
                "description": "Lossless test dataset",
                "version": "1.0",
                "year": 2026,
                "contributor": "Test",
                "date_created": "2026-03-22"
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
                    "date_captured": ""
                }
            ],
            "categories": [
                {"id": 1, "name": "person", "supercategory": "human"},
                {"id": 2, "name": "bicycle", "supercategory": "vehicle"}
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[100.5, 100.5, 200.5, 100.5, 200.5, 200.5, 100.5, 200.5]],
                    "area": 10000.0,
                    "bbox": [100.5, 100.5, 100.0, 100.0],
                    "iscrowd": 0
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "segmentation": [],
                    "area": 5000.0,
                    "bbox": [300.5, 200.5, 50.0, 100.0],
                    "iscrowd": 0
                }
            ]
        }
        coco_file.write_text(json.dumps(coco_data, indent=2))
        return str(coco_file)

    def test_labelme_lossless_roundtrip(self, sample_labelme_dir, temp_dir):
        """Test lossless round-trip for LabelMe format."""
        # Read original data
        handler = LabelMeAnnotationHandler(label_dir=str(sample_labelme_dir))
        read_result = handler.read()
        assert read_result.success is True

        # Check that original data is preserved
        dataset = read_result.data
        assert isinstance(dataset, DatasetAnnotations)
        assert len(dataset.images) == 1

        image_ann = dataset.images[0]
        assert image_ann.has_original_data() is True
        assert image_ann.original_data.format == AnnotationFormat.LABELME.value

        # Check object original data
        for obj in image_ann.objects:
            assert obj.has_original_data() is True
            assert obj.original_data.format == AnnotationFormat.LABELME.value
            # Check that original shape data is preserved
            assert "label" in obj.original_data.raw_data
            assert "points" in obj.original_data.raw_data
            assert "shape_type" in obj.original_data.raw_data

        # Write to new location
        output_dir = temp_dir / "output_labelme"
        write_result = handler.write(dataset, str(output_dir))
        assert write_result.success is True

        # Compare input and output files
        input_files = sorted(sample_labelme_dir.glob("*.json"))
        output_files = sorted(output_dir.glob("*.json"))
        assert len(input_files) == len(output_files) == 1

        # Load and compare JSON (ignore imageData which is None)
        with open(input_files[0], 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        with open(output_files[0], 'r', encoding='utf-8') as f:
            output_data = json.load(f)

        # Remove imageData for comparison (may be null)
        input_data.pop("imageData", None)
        output_data.pop("imageData", None)

        assert input_data == output_data, "LabelMe files are not identical"

    def test_yolo_lossless_roundtrip(self, sample_yolo_dir, temp_dir):
        """Test lossless round-trip for YOLO format."""
        # Read original data
        handler = YoloAnnotationHandler(
            image_dir=str(sample_yolo_dir),
            label_dir=str(sample_yolo_dir),
            class_file=str(sample_yolo_dir / "classes.txt"),
            strict_mode=False
        )
        read_result = handler.read()
        assert read_result.success is True

        # Check that original data is preserved
        dataset = read_result.data
        assert isinstance(dataset, DatasetAnnotations)

        for image_ann in dataset.images:
            for obj in image_ann.objects:
                assert obj.has_original_data() is True
                assert obj.original_data.format == AnnotationFormat.YOLO.value
                # Check original line data
                assert "line" in obj.original_data.raw_data
                assert "items" in obj.original_data.raw_data

        # Write to new location
        output_dir = temp_dir / "output_yolo"
        write_result = handler.write(dataset, str(output_dir))
        assert write_result.success is True

        # Compare input and output files
        input_files = sorted(sample_yolo_dir.glob("*.txt"))
        output_files = sorted(output_dir.glob("*.txt"))
        # Filter out class file from input
        label_input_files = [f for f in input_files if f.name != "classes.txt"]
        assert len(label_input_files) == len(output_files)

        for in_file, out_file in zip(label_input_files, output_files):
            in_lines = in_file.read_text().strip().splitlines()
            out_lines = out_file.read_text().strip().splitlines()
            assert in_lines == out_lines, f"YOLO file {in_file.name} not identical"

    def test_coco_lossless_roundtrip(self, sample_coco_file, temp_dir):
        """Test lossless round-trip for COCO format."""
        # Read original data
        handler = CocoAnnotationHandler(annotation_file=sample_coco_file)
        read_result = handler.read()
        assert read_result.success is True

        # Check that original data is preserved
        dataset = read_result.data
        assert isinstance(dataset, DatasetAnnotations)

        # Check image original data
        for image_ann in dataset.images:
            assert image_ann.has_original_data() is True
            assert image_ann.original_data.format == AnnotationFormat.COCO.value

        # Check object original data
        for image_ann in dataset.images:
            for obj in image_ann.objects:
                assert obj.has_original_data() is True
                assert obj.original_data.format == AnnotationFormat.COCO.value
                # Check original annotation data
                assert "category_id" in obj.original_data.raw_data
                assert "image_id" in obj.original_data.raw_data

        # Write to new location (preserve polygon format)
        output_file = temp_dir / "output_coco.json"
        write_result = handler.write(dataset, str(output_file), output_rle=False)
        assert write_result.success is True

        # Compare input and output files
        with open(sample_coco_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        with open(output_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)

        # Remove auto-generated fields that may differ
        for data in [input_data, output_data]:
            data.pop("__coco_original_data__", None)
            # Remove generated description fields
            for field in ["description", "url", "version", "year", "contributor", "date_created"]:
                data.pop(field, None)

        # Compare annotations (IDs may be regenerated, so compare content)
        assert len(input_data["annotations"]) == len(output_data["annotations"])
        for i, (in_ann, out_ann) in enumerate(zip(input_data["annotations"], output_data["annotations"])):
            # Remove ID for comparison
            in_copy = {k: v for k, v in in_ann.items() if k != "id"}
            out_copy = {k: v for k, v in out_ann.items() if k != "id"}
            assert in_copy == out_copy, f"Annotation {i} content differs"

        # Compare images and categories
        assert input_data["images"] == output_data["images"]
        assert input_data["categories"] == output_data["categories"]

    def test_original_data_manager(self):
        """Test OriginalDataManager functionality."""
        from dataflow.label.models import OriginalDataManager, OriginalData, ObjectAnnotation, BoundingBox

        # Create test objects
        original_data1 = OriginalData(
            format=AnnotationFormat.LABELME.value,
            raw_data={"label": "cat", "points": [[10, 20], [30, 40]]}
        )

        original_data2 = OriginalData(
            format=AnnotationFormat.YOLO.value,
            raw_data={"line": "0 0.5 0.5 0.1 0.1"}
        )

        # Test should_use_original
        obj = ObjectAnnotation(
            class_id=0,
            class_name="cat",
            bbox=BoundingBox(x=0.5, y=0.5, width=0.1, height=0.1),
            original_data=original_data1
        )

        # Should use original if format matches
        assert OriginalDataManager.should_use_original(obj, AnnotationFormat.LABELME.value) is True
        assert OriginalDataManager.should_use_original(obj, AnnotationFormat.YOLO.value) is False

        # Test merge_original_data
        merged = OriginalDataManager.merge_original_data(original_data1, original_data2)
        assert merged.format == AnnotationFormat.YOLO.value  # Different formats, keep newer one

        merged2 = OriginalDataManager.merge_original_data(None, original_data2)
        assert merged2.format == AnnotationFormat.YOLO.value

    def test_lossless_with_modified_data(self, sample_labelme_dir, temp_dir):
        """Test lossless behavior when data is modified."""
        # Read original data
        handler = LabelMeAnnotationHandler(label_dir=str(sample_labelme_dir))
        read_result = handler.read()
        assert read_result.success is True

        dataset = read_result.data
        image_ann = dataset.images[0]

        # Modify an object (change class name)
        original_obj = image_ann.objects[0]
        original_class_name = original_obj.class_name
        original_obj.class_name = "modified_cat"

        # Write to new location
        output_dir = temp_dir / "output_modified"
        write_result = handler.write(dataset, str(output_dir))
        assert write_result.success is True

        # Load written file
        output_file = output_dir / "test.json"
        with open(output_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)

        # Check that label was updated but coordinates preserved
        shapes = output_data["shapes"]
        for shape in shapes:
            if shape["label"] == "modified_cat":
                # Should have original points
                assert shape["points"] == [[10.5, 20.5], [30.5, 40.5]]
                break
        else:
            pytest.fail("Modified object not found in output")