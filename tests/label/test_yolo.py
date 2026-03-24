"""
Unit tests for yolo_handler.py
"""

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from dataflow.label.models import (BoundingBox, DatasetAnnotations,
                                   ImageAnnotation, ObjectAnnotation,
                                   Segmentation)
from dataflow.label.yolo_handler import YoloAnnotationHandler


class TestYoloAnnotationHandler:
    """Test suite for YoloAnnotationHandler class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_detection_data(self, temp_dir):
        """Create sample YOLO detection test data."""
        # Create class file
        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\nbicycle\ncar\n")

        # Create image directory and a dummy image
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        label_dir = temp_dir / "labels"
        label_dir.mkdir()

        # Create a dummy image
        img_path = image_dir / "test.jpg"
        img = np.zeros((100, 200, 3), dtype=np.uint8)  # 100x200 image
        cv2.imwrite(str(img_path), img)

        # Create label file with detection annotations
        label_file = label_dir / "test.txt"
        label_content = """0 0.5 0.5 0.2 0.2
1 0.3 0.3 0.1 0.1
2 0.7 0.7 0.15 0.15"""
        label_file.write_text(label_content)

        return {
            "class_file": class_file,
            "image_dir": image_dir,
            "label_dir": label_dir,
            "img_path": img_path,
            "label_file": label_file,
        }

    @pytest.fixture
    def sample_segmentation_data(self, temp_dir):
        """Create sample YOLO segmentation test data."""
        # Create class file
        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\nbicycle\ncar\n")

        # Create image directory and a dummy image
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        label_dir = temp_dir / "labels"
        label_dir.mkdir()

        # Create a dummy image
        img_path = image_dir / "test_seg.jpg"
        img = np.zeros((100, 200, 3), dtype=np.uint8)  # 100x200 image
        cv2.imwrite(str(img_path), img)

        # Create label file with segmentation annotations (polygon)
        label_file = label_dir / "test_seg.txt"
        # Simple triangle polygon: class_id x1 y1 x2 y2 x3 y3
        label_content = """0 0.1 0.1 0.2 0.1 0.15 0.2
1 0.5 0.5 0.6 0.5 0.55 0.6 0.52 0.58"""
        label_file.write_text(label_content)

        return {
            "class_file": class_file,
            "image_dir": image_dir,
            "label_dir": label_dir,
            "img_path": img_path,
            "label_file": label_file,
        }

    @pytest.fixture
    def sample_mixed_data(self, temp_dir):
        """Create sample YOLO mixed test data (both detection and segmentation)."""
        # Create class file
        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\nbicycle\ncar\n")

        # Create image directory and a dummy image
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        label_dir = temp_dir / "labels"
        label_dir.mkdir()

        # Create a dummy image
        img_path = image_dir / "test_mixed.jpg"
        img = np.zeros((100, 200, 3), dtype=np.uint8)  # 100x200 image
        cv2.imwrite(str(img_path), img)

        # Create label file with mixed annotations
        label_file = label_dir / "test_mixed.txt"
        label_content = """0 0.5 0.5 0.2 0.2
1 0.1 0.1 0.2 0.1 0.15 0.2
2 0.7 0.7 0.15 0.15"""
        label_file.write_text(label_content)

        return {
            "class_file": class_file,
            "image_dir": image_dir,
            "label_dir": label_dir,
            "img_path": img_path,
            "label_file": label_file,
        }

    def test_init_success(self, sample_detection_data):
        """Test successful initialization."""
        handler = YoloAnnotationHandler(
            label_dir=str(sample_detection_data["label_dir"]),
            class_file=str(sample_detection_data["class_file"]),
            image_dir=str(sample_detection_data["image_dir"]),
            strict_mode=True,
        )
        assert handler is not None
        assert handler.label_dir == sample_detection_data["label_dir"]
        assert handler.class_file == sample_detection_data["class_file"]
        assert handler.image_dir == sample_detection_data["image_dir"]
        assert handler.strict_mode is True
        assert len(handler.categories) == 3
        assert handler.categories[0] == "person"
        assert handler.categories[1] == "bicycle"
        assert handler.categories[2] == "car"

    def test_init_missing_class_file(self, temp_dir):
        """Test initialization with missing class file."""
        with pytest.raises(ValueError):
            handler = YoloAnnotationHandler(
                label_dir=str(temp_dir),
                class_file=str(temp_dir / "nonexistent.txt"),
                image_dir=str(temp_dir),
                strict_mode=True,
            )
            # The error should be raised when trying to load categories
            _ = handler.categories

    def test_load_categories_success(self, sample_detection_data):
        """Test successful category loading."""
        handler = YoloAnnotationHandler(
            label_dir=str(sample_detection_data["label_dir"]),
            class_file=str(sample_detection_data["class_file"]),
            image_dir=str(sample_detection_data["image_dir"]),
            strict_mode=False,
        )
        categories = handler.categories
        assert len(categories) == 3
        assert categories[0] == "person"
        assert categories[1] == "bicycle"
        assert categories[2] == "car"

    def test_load_categories_empty_file(self, temp_dir):
        """Test category loading from empty file."""
        class_file = temp_dir / "empty_classes.txt"
        class_file.write_text("")

        # Create minimal other directories
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        label_dir = temp_dir / "labels"
        label_dir.mkdir()

        handler = YoloAnnotationHandler(
            label_dir=str(label_dir),
            class_file=str(class_file),
            image_dir=str(image_dir),
            strict_mode=False,
        )
        categories = handler.categories
        assert len(categories) == 0

    def test_detect_annotation_type_detection(self, sample_detection_data):
        """Test annotation type detection for object detection."""
        handler = YoloAnnotationHandler(
            label_dir=str(sample_detection_data["label_dir"]),
            class_file=str(sample_detection_data["class_file"]),
            image_dir=str(sample_detection_data["image_dir"]),
            strict_mode=True,
        )

        # Test detection format (5 items)
        line_items = ["0", "0.5", "0.5", "0.2", "0.2"]
        is_det, is_seg = handler._detect_annotation_type(line_items)
        assert is_det is True
        assert is_seg is False

    def test_detect_annotation_type_segmentation(self, sample_segmentation_data):
        """Test annotation type detection for instance segmentation."""
        handler = YoloAnnotationHandler(
            label_dir=str(sample_segmentation_data["label_dir"]),
            class_file=str(sample_segmentation_data["class_file"]),
            image_dir=str(sample_segmentation_data["image_dir"]),
            strict_mode=True,
        )

        # Test segmentation format (7 items: class_id + 3 points = 6 coordinates)
        line_items = ["0", "0.1", "0.1", "0.2", "0.1", "0.15", "0.2"]
        is_det, is_seg = handler._detect_annotation_type(line_items)
        assert is_det is False
        assert is_seg is True

    def test_detect_annotation_type_invalid(self, sample_detection_data):
        """Test annotation type detection for invalid format."""
        handler = YoloAnnotationHandler(
            label_dir=str(sample_detection_data["label_dir"]),
            class_file=str(sample_detection_data["class_file"]),
            image_dir=str(sample_detection_data["image_dir"]),
            strict_mode=True,
        )

        # Test invalid format (4 items)
        line_items = ["0", "0.5", "0.5", "0.2"]
        with pytest.raises(ValueError, match="Invalid YOLO format"):
            handler._detect_annotation_type(line_items)

        # Test invalid format (6 items, even number)
        line_items = ["0", "0.1", "0.1", "0.2", "0.1", "0.15"]
        with pytest.raises(ValueError, match="Invalid YOLO format"):
            handler._detect_annotation_type(line_items)

    def test_read_detection_success(self, sample_detection_data):
        """Test successful reading of detection annotations."""
        handler = YoloAnnotationHandler(
            label_dir=str(sample_detection_data["label_dir"]),
            class_file=str(sample_detection_data["class_file"]),
            image_dir=str(sample_detection_data["image_dir"]),
            strict_mode=True,
        )

        result = handler.read()
        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, DatasetAnnotations)
        assert len(result.data.images) == 1
        assert len(result.data.categories) == 3

        image_ann = result.data.images[0]
        assert image_ann.image_id == "test"
        assert image_ann.width == 200
        assert image_ann.height == 100
        assert len(image_ann.objects) == 3

        # Check first object
        obj = image_ann.objects[0]
        assert obj.class_id == 0
        assert obj.class_name == "person"
        assert obj.bbox is not None
        assert obj.segmentation is None
        assert obj.bbox.x == 0.5
        assert obj.bbox.y == 0.5
        assert obj.bbox.width == 0.2
        assert obj.bbox.height == 0.2

        # Check handler flags
        assert handler.is_det is True
        assert handler.is_seg is False

    def test_read_segmentation_success(self, sample_segmentation_data):
        """Test successful reading of segmentation annotations."""
        handler = YoloAnnotationHandler(
            label_dir=str(sample_segmentation_data["label_dir"]),
            class_file=str(sample_segmentation_data["class_file"]),
            image_dir=str(sample_segmentation_data["image_dir"]),
            strict_mode=True,
        )

        result = handler.read()
        assert result.success is True
        assert result.data is not None
        assert isinstance(result.data, DatasetAnnotations)
        assert len(result.data.images) == 1

        image_ann = result.data.images[0]
        assert image_ann.image_id == "test_seg"
        assert len(image_ann.objects) == 2

        # Check first object (triangle)
        obj = image_ann.objects[0]
        assert obj.class_id == 0
        assert obj.class_name == "person"
        assert obj.bbox is None
        assert obj.segmentation is not None
        assert len(obj.segmentation.points) == 3

        # Check second object (quadrilateral)
        obj = image_ann.objects[1]
        assert obj.class_id == 1
        assert obj.class_name == "bicycle"
        assert obj.bbox is None
        assert obj.segmentation is not None
        assert len(obj.segmentation.points) == 4

        # Check handler flags
        assert handler.is_det is False
        assert handler.is_seg is True

    def test_read_mixed_success(self, sample_mixed_data):
        """Test successful reading of mixed annotations."""
        handler = YoloAnnotationHandler(
            label_dir=str(sample_mixed_data["label_dir"]),
            class_file=str(sample_mixed_data["class_file"]),
            image_dir=str(sample_mixed_data["image_dir"]),
            strict_mode=True,
        )

        result = handler.read()
        assert result.success is True
        assert result.data is not None
        assert len(result.data.images) == 1

        image_ann = result.data.images[0]
        assert len(image_ann.objects) == 3

        # Check object types
        obj1 = image_ann.objects[0]  # Detection
        assert obj1.bbox is not None
        assert obj1.segmentation is None

        obj2 = image_ann.objects[1]  # Segmentation
        assert obj2.bbox is None
        assert obj2.segmentation is not None

        obj3 = image_ann.objects[2]  # Detection
        assert obj3.bbox is not None
        assert obj3.segmentation is None

        # Check handler flags
        assert handler.is_det is True
        assert handler.is_seg is True

    def test_read_no_label_files(self, temp_dir):
        """Test reading when no label files exist."""
        # Create minimal directories
        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\n")
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        label_dir = temp_dir / "labels"
        label_dir.mkdir()

        handler = YoloAnnotationHandler(
            label_dir=str(label_dir),
            class_file=str(class_file),
            image_dir=str(image_dir),
            strict_mode=True,
        )

        result = handler.read()
        assert result.success is False
        assert "No TXT files found" in result.message

    def test_read_invalid_class_id(self, temp_dir):
        """Test reading with invalid class ID."""
        # Create test data
        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\nbicycle\n")

        image_dir = temp_dir / "images"
        image_dir.mkdir()
        label_dir = temp_dir / "labels"
        label_dir.mkdir()

        # Create dummy image
        img_path = image_dir / "test.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        # Create label with invalid class ID (2, but only 0-1 are valid)
        label_file = label_dir / "test.txt"
        label_file.write_text("2 0.5 0.5 0.2 0.2")

        handler = YoloAnnotationHandler(
            label_dir=str(label_dir),
            class_file=str(class_file),
            image_dir=str(image_dir),
            strict_mode=True,
        )

        result = handler.read()
        assert result.success is False
        assert "Invalid class ID" in result.message

    def test_read_invalid_coordinate(self, temp_dir):
        """Test reading with invalid coordinate (out of range)."""
        # Create test data
        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\n")

        image_dir = temp_dir / "images"
        image_dir.mkdir()
        label_dir = temp_dir / "labels"
        label_dir.mkdir()

        # Create dummy image
        img_path = image_dir / "test.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        # Create label with invalid coordinate (> 1.0)
        label_file = label_dir / "test.txt"
        label_file.write_text("0 1.5 0.5 0.2 0.2")  # x_center = 1.5

        handler = YoloAnnotationHandler(
            label_dir=str(label_dir),
            class_file=str(class_file),
            image_dir=str(image_dir),
            strict_mode=True,
        )

        result = handler.read()
        assert result.success is False
        assert "out of range" in result.message.lower()

    def test_read_missing_image(self, temp_dir):
        """Test reading when corresponding image is missing."""
        # Create test data
        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\n")

        image_dir = temp_dir / "images"
        image_dir.mkdir()
        label_dir = temp_dir / "labels"
        label_dir.mkdir()

        # Create label file but no image
        label_file = label_dir / "test.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.2")

        handler = YoloAnnotationHandler(
            label_dir=str(label_dir),
            class_file=str(class_file),
            image_dir=str(image_dir),
            strict_mode=True,
        )

        result = handler.read()
        assert result.success is False
        assert "No corresponding image" in result.message

    def test_write_detection_success(self, sample_detection_data):
        """Test successful writing of detection annotations."""
        # First read the data
        handler = YoloAnnotationHandler(
            label_dir=str(sample_detection_data["label_dir"]),
            class_file=str(sample_detection_data["class_file"]),
            image_dir=str(sample_detection_data["image_dir"]),
            strict_mode=True,
        )

        read_result = handler.read()
        assert read_result.success is True

        # Write to new directory
        output_dir = sample_detection_data["label_dir"].parent / "output"
        write_result = handler.write(read_result.data, str(output_dir))

        assert write_result.success is True
        assert output_dir.exists()

        # Check output file
        output_file = output_dir / "test.txt"
        assert output_file.exists()

        # Read and verify content
        content = output_file.read_text().strip()
        lines = content.split("\n")
        assert len(lines) == 3

        # Verify first line
        parts = lines[0].split()
        assert len(parts) == 5
        assert parts[0] == "0"  # class_id
        assert float(parts[1]) == pytest.approx(0.5, abs=1e-6)  # x_center
        assert float(parts[2]) == pytest.approx(0.5, abs=1e-6)  # y_center
        assert float(parts[3]) == pytest.approx(0.2, abs=1e-6)  # width
        assert float(parts[4]) == pytest.approx(0.2, abs=1e-6)  # height

    def test_write_segmentation_success(self, sample_segmentation_data):
        """Test successful writing of segmentation annotations."""
        # First read the data
        handler = YoloAnnotationHandler(
            label_dir=str(sample_segmentation_data["label_dir"]),
            class_file=str(sample_segmentation_data["class_file"]),
            image_dir=str(sample_segmentation_data["image_dir"]),
            strict_mode=True,
        )

        read_result = handler.read()
        assert read_result.success is True

        # Write to new directory
        output_dir = sample_segmentation_data["label_dir"].parent / "output"
        write_result = handler.write(read_result.data, str(output_dir))

        assert write_result.success is True
        assert output_dir.exists()

        # Check output file
        output_file = output_dir / "test_seg.txt"
        assert output_file.exists()

        # Read and verify content
        content = output_file.read_text().strip()
        lines = content.split("\n")
        assert len(lines) == 2

        # Verify first line (triangle)
        parts = lines[0].split()
        assert len(parts) == 7  # class_id + 3 points * 2 coordinates
        assert parts[0] == "0"  # class_id
        assert float(parts[1]) == pytest.approx(0.1, abs=1e-6)  # x1
        assert float(parts[2]) == pytest.approx(0.1, abs=1e-6)  # y1

    def test_write_empty_dataset(self, temp_dir):
        """Test writing empty dataset."""
        # Create minimal handler
        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\n")
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        label_dir = temp_dir / "labels"
        label_dir.mkdir()

        handler = YoloAnnotationHandler(
            label_dir=str(label_dir),
            class_file=str(class_file),
            image_dir=str(image_dir),
            strict_mode=True,
        )

        # Create empty dataset
        dataset = DatasetAnnotations()
        dataset.categories = {0: "person"}

        # Write to output directory
        output_dir = temp_dir / "output"
        write_result = handler.write(dataset, str(output_dir))

        # Should succeed but write no files (no images in dataset)
        assert write_result.success is True
        assert write_result.message == "Successfully wrote 0/0 images"
        assert output_dir.exists()

    def test_validate_detection_success(self, sample_detection_data):
        """Test successful validation of detection file."""
        handler = YoloAnnotationHandler(
            label_dir=str(sample_detection_data["label_dir"]),
            class_file=str(sample_detection_data["class_file"]),
            image_dir=str(sample_detection_data["image_dir"]),
            strict_mode=True,
        )

        is_valid = handler.validate(str(sample_detection_data["label_file"]))
        assert is_valid is True

    def test_validate_segmentation_success(self, sample_segmentation_data):
        """Test successful validation of segmentation file."""
        handler = YoloAnnotationHandler(
            label_dir=str(sample_segmentation_data["label_dir"]),
            class_file=str(sample_segmentation_data["class_file"]),
            image_dir=str(sample_segmentation_data["image_dir"]),
            strict_mode=True,
        )

        is_valid = handler.validate(str(sample_segmentation_data["label_file"]))
        assert is_valid is True

    def test_validate_invalid_format(self, temp_dir):
        """Test validation of invalid format file."""
        # Create test data
        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\n")

        image_dir = temp_dir / "images"
        image_dir.mkdir()
        label_dir = temp_dir / "labels"
        label_dir.mkdir()

        # Create invalid label file (wrong number of items)
        label_file = label_dir / "test.txt"
        label_file.write_text("0 0.5 0.5 0.2")  # Only 4 items

        handler = YoloAnnotationHandler(
            label_dir=str(label_dir),
            class_file=str(class_file),
            image_dir=str(image_dir),
            strict_mode=True,
        )

        is_valid = handler.validate(str(label_file))
        assert is_valid is False

    def test_validate_out_of_range_coordinate(self, temp_dir):
        """Test validation of file with out-of-range coordinate."""
        # Create test data
        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\n")

        image_dir = temp_dir / "images"
        image_dir.mkdir()
        label_dir = temp_dir / "labels"
        label_dir.mkdir()

        # Create label with invalid coordinate
        label_file = label_dir / "test.txt"
        label_file.write_text("0 1.5 0.5 0.2 0.2")  # x_center = 1.5

        handler = YoloAnnotationHandler(
            label_dir=str(label_dir),
            class_file=str(class_file),
            image_dir=str(image_dir),
            strict_mode=True,
        )

        is_valid = handler.validate(str(label_file))
        assert is_valid is False

    def test_validate_invalid_class_id(self, temp_dir):
        """Test validation of file with invalid class ID."""
        # Create test data
        class_file = temp_dir / "classes.txt"
        class_file.write_text("person\n")  # Only class 0

        image_dir = temp_dir / "images"
        image_dir.mkdir()
        label_dir = temp_dir / "labels"
        label_dir.mkdir()

        # Create label with invalid class ID
        label_file = label_dir / "test.txt"
        label_file.write_text("1 0.5 0.5 0.2 0.2")  # Class ID 1 doesn't exist

        handler = YoloAnnotationHandler(
            label_dir=str(label_dir),
            class_file=str(class_file),
            image_dir=str(image_dir),
            strict_mode=True,
        )

        is_valid = handler.validate(str(label_file))
        assert is_valid is False

    def test_validate_missing_file(self, sample_detection_data):
        """Test validation of non-existent file."""
        handler = YoloAnnotationHandler(
            label_dir=str(sample_detection_data["label_dir"]),
            class_file=str(sample_detection_data["class_file"]),
            image_dir=str(sample_detection_data["image_dir"]),
            strict_mode=True,
        )

        is_valid = handler.validate("/nonexistent/path/test.txt")
        assert is_valid is False

    def test_object_to_yolo_line_detection(self, sample_detection_data):
        """Test conversion of object annotation to YOLO line (detection)."""
        handler = YoloAnnotationHandler(
            label_dir=str(sample_detection_data["label_dir"]),
            class_file=str(sample_detection_data["class_file"]),
            image_dir=str(sample_detection_data["image_dir"]),
            strict_mode=True,
        )

        # Create object with bbox
        bbox = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.2)
        obj = ObjectAnnotation(
            class_id=0,
            class_name="person",
            bbox=bbox,
            segmentation=None,
            confidence=1.0,
        )

        line = handler._object_to_yolo_line(obj, 100, 100)
        assert line is not None

        parts = line.split()
        assert len(parts) == 5
        assert parts[0] == "0"
        assert float(parts[1]) == pytest.approx(0.5, abs=1e-6)
        assert float(parts[2]) == pytest.approx(0.5, abs=1e-6)
        assert float(parts[3]) == pytest.approx(0.2, abs=1e-6)
        assert float(parts[4]) == pytest.approx(0.2, abs=1e-6)

    def test_object_to_yolo_line_segmentation(self, sample_segmentation_data):
        """Test conversion of object annotation to YOLO line (segmentation)."""
        handler = YoloAnnotationHandler(
            label_dir=str(sample_segmentation_data["label_dir"]),
            class_file=str(sample_segmentation_data["class_file"]),
            image_dir=str(sample_segmentation_data["image_dir"]),
            strict_mode=True,
        )

        # Create object with segmentation
        points = [(0.1, 0.1), (0.2, 0.1), (0.15, 0.2)]
        segmentation = Segmentation(points=points)
        obj = ObjectAnnotation(
            class_id=1,
            class_name="bicycle",
            bbox=None,
            segmentation=segmentation,
            confidence=1.0,
        )

        line = handler._object_to_yolo_line(obj, 100, 100)
        assert line is not None

        parts = line.split()
        assert len(parts) == 7  # class_id + 3 points * 2
        assert parts[0] == "1"
        assert float(parts[1]) == pytest.approx(0.1, abs=1e-6)
        assert float(parts[2]) == pytest.approx(0.1, abs=1e-6)

    def test_object_to_yolo_line_no_annotation(self, sample_detection_data):
        """Test conversion of object with class name not in categories."""
        handler = YoloAnnotationHandler(
            label_dir=str(sample_detection_data["label_dir"]),
            class_file=str(sample_detection_data["class_file"]),
            image_dir=str(sample_detection_data["image_dir"]),
            strict_mode=True,
        )

        # Create object with bbox but class_name not in handler's categories
        bbox = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.2)
        obj = ObjectAnnotation(
            class_id=99,  # Invalid class ID
            class_name="dog",  # Not in categories
            bbox=bbox,
            segmentation=None,
            confidence=1.0,
        )

        line = handler._object_to_yolo_line(obj, 100, 100)
        assert line is None  # Should return None because class_name not found
