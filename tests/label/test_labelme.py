"""
Unit tests for labelme_handler.py
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from dataflow.label.labelme_handler import LabelMeAnnotationHandler
from dataflow.label.models import (AnnotationFormat, BoundingBox,
                                   DatasetAnnotations, ImageAnnotation,
                                   ObjectAnnotation, Segmentation)


class TestLabelMeAnnotationHandler:
    """Test suite for LabelMeAnnotationHandler class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_labelme_data(self):
        """Create sample LabelMe JSON data."""
        return {
            "version": "5.0.1",
            "flags": {},
            "shapes": [
                {
                    "label": "cat",
                    "points": [[10, 20], [30, 40]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                },
                {
                    "label": "dog",
                    "points": [[50, 60], [70, 80], [80, 90]],
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                },
            ],
            "imagePath": "test.jpg",
            "imageData": None,
            "imageHeight": 100,
            "imageWidth": 200,
        }

    @pytest.fixture
    def labelme_dir_with_data(self, temp_dir, sample_labelme_data):
        """Create a directory with LabelMe JSON files for testing."""
        image_file = temp_dir / "test.jpg"
        image_file.touch()

        json_file = temp_dir / "test.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_labelme_data, f, indent=2)

        class_file = temp_dir / "classes.txt"
        class_file.write_text("cat\ndog\nbird\n")

        return temp_dir

    def test_init_with_class_file(self, labelme_dir_with_data):
        """Test initialization with class file."""
        class_file = labelme_dir_with_data / "classes.txt"
        handler = LabelMeAnnotationHandler(
            label_dir=str(labelme_dir_with_data), class_file=str(class_file)
        )
        assert len(handler.categories) == 3
        assert handler.categories[0] == "cat"
        assert handler.categories[1] == "dog"
        assert handler.categories[2] == "bird"

    def test_init_without_class_file(self, labelme_dir_with_data):
        """Test initialization without class file."""
        handler = LabelMeAnnotationHandler(label_dir=str(labelme_dir_with_data))
        assert handler.categories == {}
        assert handler.class_file is None

    def test_read_success(self, labelme_dir_with_data):
        """Test successful reading of LabelMe annotations."""
        handler = LabelMeAnnotationHandler(
            label_dir=str(labelme_dir_with_data), strict_mode=True
        )

        result = handler.read()
        assert result.success is True
        assert isinstance(result.data, DatasetAnnotations)
        assert result.data.format == AnnotationFormat.LABELME
        assert len(result.data.images) == 1

        image_ann = result.data.images[0]
        assert image_ann.image_id == "test"
        assert image_ann.width == 200
        assert image_ann.height == 100
        assert len(image_ann.objects) == 2

        # Check first object (rectangle -> bbox, absolute pixels, top-left)
        obj1 = image_ann.objects[0]
        assert obj1.class_name == "cat"
        assert obj1.bbox is not None
        assert obj1.segmentation is None
        # points: [[10,20],[30,40]] -> x=10, y=20, w=20, h=20
        assert obj1.bbox.x == 10
        assert obj1.bbox.y == 20
        assert obj1.bbox.width == 20
        assert obj1.bbox.height == 20

        # Check second object (polygon -> segmentation, absolute pixels)
        obj2 = image_ann.objects[1]
        assert obj2.class_name == "dog"
        assert obj2.bbox is None
        assert obj2.segmentation is not None
        assert len(obj2.segmentation.points) == 3
        # Points stored as absolute pixels
        assert obj2.segmentation.points[0] == (50, 60)
        assert obj2.segmentation.points[1] == (70, 80)
        assert obj2.segmentation.points[2] == (80, 90)

    def test_read_no_json_files(self, temp_dir):
        """Test reading from directory with no JSON files."""
        handler = LabelMeAnnotationHandler(label_dir=str(temp_dir))
        result = handler.read()
        assert result.success is False
        assert "No JSON files found" in result.message

    def test_read_invalid_json(self, temp_dir):
        """Test reading invalid JSON file."""
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("{invalid json}")

        handler = LabelMeAnnotationHandler(label_dir=str(temp_dir), strict_mode=True)
        result = handler.read()
        assert result.success is False
        assert "Invalid JSON" in result.message

    def test_read_missing_required_field(self, temp_dir, sample_labelme_data):
        """Test reading JSON with missing required field."""
        del sample_labelme_data["imagePath"]

        json_file = temp_dir / "test.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_labelme_data, f)

        handler = LabelMeAnnotationHandler(label_dir=str(temp_dir), strict_mode=True)
        result = handler.read()
        assert result.success is False
        assert "Missing required field" in result.message

    def test_read_non_strict_mode(self, temp_dir, sample_labelme_data):
        """Test reading with strict_mode=False skips invalid files."""
        valid_file = temp_dir / "valid.json"
        with open(valid_file, "w", encoding="utf-8") as f:
            json.dump(sample_labelme_data, f)

        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("{invalid json}")

        handler = LabelMeAnnotationHandler(label_dir=str(temp_dir), strict_mode=False)
        result = handler.read()
        assert result.success is True
        assert len(result.data.images) == 1

    def test_parse_shape_rectangle(self):
        """Test parsing rectangle shape."""
        handler = LabelMeAnnotationHandler(label_dir=".")
        shape = {
            "label": "car",
            "points": [[10, 20], [30, 40]],
            "shape_type": "rectangle",
        }

        handler.categories = {0: "car"}

        result = handler._parse_shape(shape, img_width=100, img_height=100)
        assert result.success is True
        obj = result.data
        assert obj.class_name == "car"
        assert obj.bbox is not None
        assert obj.segmentation is None

        # Check absolute pixel coordinates (LabelMe native format)
        assert obj.bbox.x == 10  # min(10, 30)
        assert obj.bbox.y == 20  # min(20, 40)
        assert obj.bbox.width == 20  # |30-10|
        assert obj.bbox.height == 20  # |40-20|

    def test_parse_shape_polygon(self):
        """Test parsing polygon shape."""
        handler = LabelMeAnnotationHandler(label_dir=".")
        shape = {
            "label": "person",
            "points": [[10, 20], [30, 40], [50, 60]],
            "shape_type": "polygon",
        }

        handler.categories = {0: "person"}

        result = handler._parse_shape(shape, img_width=100, img_height=100)
        assert result.success is True
        obj = result.data
        assert obj.class_name == "person"
        assert obj.bbox is None
        assert obj.segmentation is not None
        assert len(obj.segmentation.points) == 3

        # Check absolute pixel coordinates (LabelMe native format)
        points = obj.segmentation.points
        assert points[0] == (10, 20)
        assert points[1] == (30, 40)
        assert points[2] == (50, 60)

    def test_parse_shape_invalid_type(self):
        """Test parsing unsupported shape type."""
        handler = LabelMeAnnotationHandler(label_dir=".")
        shape = {
            "label": "test",
            "points": [[10, 20]],
            "shape_type": "circle",
        }

        result = handler._parse_shape(shape, img_width=100, img_height=100)
        assert result.success is False
        assert "Unsupported shape type" in result.message

    def test_parse_shape_missing_label(self):
        """Test parsing shape with missing label."""
        handler = LabelMeAnnotationHandler(label_dir=".")
        shape = {"label": "", "points": [[10, 20], [30, 40]], "shape_type": "rectangle"}

        result = handler._parse_shape(shape, img_width=100, img_height=100)
        assert result.success is False
        assert "Shape missing label" in result.message

    def test_write_success(self, temp_dir, sample_labelme_data):
        """Test successful writing of LabelMe annotations."""
        json_file = temp_dir / "test.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_labelme_data, f)

        handler = LabelMeAnnotationHandler(label_dir=str(temp_dir))
        read_result = handler.read()
        assert read_result.success is True

        output_dir = temp_dir / "output"
        write_result = handler.write(read_result.data, str(output_dir))
        assert write_result.success is True
        assert write_result.data["written_count"] == 1

        output_file = output_dir / "test.json"
        assert output_file.exists()

        with open(output_file, "r", encoding="utf-8") as f:
            output_data = json.load(f)

        assert output_data["version"] == "5.0.1"
        assert output_data["imageHeight"] == 100
        assert output_data["imageWidth"] == 200
        assert len(output_data["shapes"]) == 2

    def test_write_empty_annotations(self, temp_dir):
        """Test writing empty annotations."""
        handler = LabelMeAnnotationHandler(label_dir=str(temp_dir))
        dataset = DatasetAnnotations()

        output_dir = temp_dir / "output"
        result = handler.write(dataset, str(output_dir))
        assert result.success is True
        assert result.data["written_count"] == 0

    def test_write_with_bbox_and_segmentation(self, temp_dir):
        """Test writing annotations with both bbox and segmentation."""
        dataset = DatasetAnnotations()
        dataset.add_category(0, "cat")
        dataset.add_category(1, "dog")

        image = ImageAnnotation(
            image_id="mixed",
            image_path=str(temp_dir / "mixed.jpg"),
            width=200,
            height=100,
            objects=[
                ObjectAnnotation(
                    class_id=0,
                    class_name="cat",
                    bbox=BoundingBox(x=50, y=25, width=20, height=20),
                ),
                ObjectAnnotation(
                    class_id=1,
                    class_name="dog",
                    segmentation=Segmentation(
                        points=[(10, 10), (20, 10), (20, 20)]
                    ),
                ),
            ],
        )
        dataset.add_image(image)

        handler = LabelMeAnnotationHandler(label_dir=str(temp_dir))
        output_dir = temp_dir / "output"
        result = handler.write(dataset, str(output_dir))
        assert result.success is True

        output_file = output_dir / "mixed.json"
        assert output_file.exists()

        with open(output_file, "r", encoding="utf-8") as f:
            output_data = json.load(f)

        assert len(output_data["shapes"]) == 2
        shape_types = [s["shape_type"] for s in output_data["shapes"]]
        assert "rectangle" in shape_types
        assert "polygon" in shape_types

    def test_validate_valid_file(self, temp_dir, sample_labelme_data):
        """Test validation of valid LabelMe JSON file."""
        json_file = temp_dir / "valid.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_labelme_data, f)

        handler = LabelMeAnnotationHandler(label_dir=str(temp_dir))
        assert handler.validate(str(json_file)) is True

    def test_validate_invalid_json(self, temp_dir):
        """Test validation of invalid JSON file."""
        json_file = temp_dir / "invalid.json"
        json_file.write_text("{invalid}")

        handler = LabelMeAnnotationHandler(label_dir=str(temp_dir), strict_mode=False)
        assert handler.validate(str(json_file)) is False

    def test_validate_missing_field(self, temp_dir, sample_labelme_data):
        """Test validation of JSON with missing required field."""
        del sample_labelme_data["shapes"]
        json_file = temp_dir / "invalid.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(sample_labelme_data, f)

        handler = LabelMeAnnotationHandler(label_dir=str(temp_dir), strict_mode=False)
        assert handler.validate(str(json_file)) is False

    def test_object_to_shape_bbox(self):
        """Test converting object with bbox to LabelMe shape.

        Bbox is now stored in absolute pixel coordinates (LabelMe native),
        so no img_width/img_height needed.
        """
        handler = LabelMeAnnotationHandler(label_dir=".")
        # Bbox in absolute pixels: x=40, y=40, w=20, h=20
        obj = ObjectAnnotation(
            class_id=0,
            class_name="car",
            bbox=BoundingBox(x=40, y=40, width=20, height=20),
        )

        shape = handler._object_to_shape(obj)
        assert shape is not None
        assert shape["label"] == "car"
        assert shape["shape_type"] == "rectangle"
        assert len(shape["points"]) == 2
        assert shape["points"][0] == [40, 40]  # x1, y1
        assert shape["points"][1] == [60, 60]  # x2, y2

    def test_object_to_shape_segmentation(self):
        """Test converting object with segmentation to LabelMe shape.

        Points are already in absolute pixels (LabelMe native).
        """
        handler = LabelMeAnnotationHandler(label_dir=".")
        obj = ObjectAnnotation(
            class_id=0,
            class_name="person",
            segmentation=Segmentation(points=[(10, 20), (30, 40), (50, 60)]),
        )

        shape = handler._object_to_shape(obj)
        assert shape is not None
        assert shape["label"] == "person"
        assert shape["shape_type"] == "polygon"
        assert len(shape["points"]) == 3
        assert shape["points"][0] == [10, 20]
        assert shape["points"][1] == [30, 40]
        assert shape["points"][2] == [50, 60]

    def test_object_to_shape_no_annotation(self):
        """Test converting object with neither bbox nor segmentation."""
        handler = LabelMeAnnotationHandler(label_dir=".")
        # Create an ObjectAnnotation with bbox then remove it to bypass
        # __post_init__ validation (which requires at least one)
        obj = ObjectAnnotation(
            class_id=0,
            class_name="ghost",
            bbox=BoundingBox(x=40, y=40, width=20, height=20),
        )
        # Remove bbox and segmentation to test edge case
        obj.bbox = None

        shape = handler._object_to_shape(obj)
        assert shape is None
