"""
YOLO annotation format handler.

Handles reading and writing of YOLO format annotation files.
Supports both object detection and instance segmentation formats.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from dataflow.util.file_util import FileOperations

from .base import AnnotationResult, BaseAnnotationHandler
from .models import (AnnotationFormat, BoundingBox, DatasetAnnotations,
                     ImageAnnotation, ObjectAnnotation, OriginalData,
                     Segmentation)


class YoloAnnotationHandler(BaseAnnotationHandler):
    """Handler for YOLO annotation format."""

    def __init__(self, label_dir: str, class_file: str, image_dir: str, **kwargs):
        """
        Initialize YOLO handler.

        Args:
            label_dir: Directory containing YOLO TXT label files
            class_file: File containing class names (one per line, required)
            image_dir: Directory containing image files (for getting image dimensions)
            **kwargs: Additional arguments for BaseAnnotationHandler
        """
        super().__init__(**kwargs)
        self.label_dir = Path(label_dir)
        self.class_file = Path(class_file)
        self.image_dir = Path(image_dir)
        self.file_ops = FileOperations(logger=self.logger)
        self.categories = self._load_categories()

    def _load_categories(self) -> Dict[int, str]:
        """Load category mapping from class file."""
        categories: Dict[int, str] = {}

        if not self.class_file.exists():
            self._log_error(f"Class file does not exist: {self.class_file}")
            return categories

        try:
            lines = self.file_ops.read_lines(self.class_file)
            for i, line in enumerate(lines):
                if line.strip():  # Skip empty lines
                    categories[i] = line.strip()
            self._log_info(
                f"Loaded {len(categories)} categories from {self.class_file}"
            )
        except Exception as e:
            self._log_error(f"Failed to load class file {self.class_file}: {e}")

        return categories

    def _detect_annotation_type(self, line_items: List[str]) -> Tuple[bool, bool]:
        """
        Detect annotation type from line data.

        Args:
            line_items: Split line data items

        Returns:
            Tuple[is_detection, is_segmentation]

        Raises:
            ValueError: If format is invalid
        """
        if len(line_items) == 5:
            # Object detection format: class_id x_center y_center width height
            return True, False
        elif len(line_items) > 5 and len(line_items) % 2 == 1:
            # Instance segmentation format: class_id x1 y1 x2 y2 ... xn yn
            # First is class_id, followed by pairs of x,y coordinates
            return False, True
        else:
            raise ValueError(f"Invalid YOLO format: {len(line_items)} items")

    def _get_image_size(self, image_path: Path) -> Tuple[int, int]:
        """Get image dimensions from image file."""
        try:
            if not image_path.exists():
                self._log_error(f"Image file not found: {image_path}")
                return 0, 0

            # Read image using OpenCV
            img = cv2.imread(str(image_path))
            if img is None:
                self._log_error(f"Failed to read image: {image_path}")
                return 0, 0

            height, width = img.shape[:2]
            return width, height
        except Exception as e:
            self._log_error(f"Error getting image size for {image_path}: {e}")
            return 0, 0

    def read(self) -> AnnotationResult:
        """Read all YOLO TXT files in the directory."""
        result = AnnotationResult(success=False)

        # Validate directories
        if not self.label_dir.exists():
            result.add_error(f"Label directory does not exist: {self.label_dir}")
            return result

        if not self.image_dir.exists():
            result.add_error(f"Image directory does not exist: {self.image_dir}")
            return result

        if not self.categories:
            result.add_error(f"No categories loaded from {self.class_file}")
            return result

        try:
            # Find all TXT files in label directory
            txt_files = self.file_ops.find_files(
                self.label_dir, "*.txt", recursive=False
            )
            if not txt_files:
                result.add_error(f"No TXT files found in {self.label_dir}")
                return result

            dataset = DatasetAnnotations()
            dataset.categories = self.categories.copy()

            for txt_file in txt_files:
                image_result = self._read_single_file(txt_file)
                if not image_result.success:
                    if self.strict_mode:
                        result.add_error(
                            f"Failed to read {txt_file}: {image_result.message}"
                        )
                        return result
                    else:
                        self._log_warning(
                            f"Skipping {txt_file}: {image_result.message}"
                        )
                        continue

                image_ann = image_result.data
                if image_ann is None:
                    result.add_error(
                        f"Internal error: image annotation data is None for {txt_file}"
                    )
                    return result
                if not isinstance(image_ann, ImageAnnotation):
                    result.add_error(
                        f"Internal error: invalid image annotation type for {txt_file}"
                    )
                    return result

                dataset.add_image(image_ann)

            # Set annotation flags
            self._set_annotation_flags(dataset)

            result.success = True
            result.data = dataset
            result.message = f"Successfully read {len(dataset.images)} images"

        except Exception as e:
            result.add_error(f"Unexpected error reading YOLO annotations: {e}")

        return result

    def _read_single_file(self, txt_file: Path) -> AnnotationResult:
        """Read a single YOLO TXT file."""
        result = AnnotationResult(success=False)

        try:
            # Get corresponding image file
            image_stem = txt_file.stem
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
            image_path = None

            for ext in image_extensions:
                potential_path = self.image_dir / f"{image_stem}{ext}"
                if potential_path.exists():
                    image_path = potential_path
                    break

            if image_path is None:
                # Try with any extension
                image_files = list(self.image_dir.glob(f"{image_stem}.*"))
                if image_files:
                    image_path = image_files[0]
                else:
                    result.add_error(f"No corresponding image found for {txt_file}")
                    return result

            # Get image dimensions
            img_width, img_height = self._get_image_size(image_path)
            if img_width <= 0 or img_height <= 0:
                result.add_error(
                    f"Invalid image dimensions for {image_path}: {img_width}x{img_height}"
                )
                return result

            # Read label file
            lines = self.file_ops.read_lines(txt_file)
            objects: List[ObjectAnnotation] = []

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    items = line.split()
                    if not items:
                        continue

                    is_detection, is_segmentation = self._detect_annotation_type(items)
                    class_id = int(items[0])

                    # Validate class ID
                    if class_id not in self.categories:
                        result.add_error(
                            f"Invalid class ID {class_id} in {txt_file}, line {line_num}"
                        )
                        if self.strict_mode:
                            return result
                        else:
                            self._log_warning(
                                f"Skipping line {line_num}: invalid class ID {class_id}"
                            )
                            continue

                    class_name = self.categories[class_id]
                    bbox = None
                    segmentation = None

                    if is_detection:
                        # Parse detection: class_id x_center y_center width height
                        x_center = float(items[1])
                        y_center = float(items[2])
                        width = float(items[3])
                        height = float(items[4])

                        # Validate normalized coordinates
                        coords_to_check = [
                            ("x_center", x_center),
                            ("y_center", y_center),
                            ("width", width),
                            ("height", height),
                        ]
                        for name, value in coords_to_check:
                            if not self._validate_normalized_coordinate(value, name):
                                result.add_error(
                                    f"Invalid coordinate in {txt_file}, line {line_num}"
                                )
                                if self.strict_mode:
                                    return result
                                else:
                                    self._log_warning(
                                        f"Skipping line {line_num}: invalid coordinate"
                                    )
                                    continue

                        bbox = BoundingBox(
                            x=x_center, y=y_center, width=width, height=height
                        )
                        if not self._validate_bbox(bbox):
                            result.add_error(
                                f"Invalid bbox in {txt_file}, line {line_num}"
                            )
                            if self.strict_mode:
                                return result
                            else:
                                self._log_warning(
                                    f"Skipping line {line_num}: invalid bbox"
                                )
                                continue

                    elif is_segmentation:
                        # Parse segmentation: class_id x1 y1 x2 y2 ... xn yn
                        coords = [float(x) for x in items[1:]]
                        if len(coords) % 2 != 0:
                            result.add_error(
                                f"Odd number of coordinates in {txt_file}, line {line_num}"
                            )
                            if self.strict_mode:
                                return result
                            else:
                                self._log_warning(
                                    f"Skipping line {line_num}: odd number of coordinates"
                                )
                                continue

                        # Create list of (x, y) points
                        points = [
                            (coords[i], coords[i + 1]) for i in range(0, len(coords), 2)
                        ]

                        # Validate points
                        for i, (x, y) in enumerate(points):
                            if not self._validate_normalized_coordinate(
                                x, f"point[{i}].x"
                            ):
                                result.add_error(
                                    f"Invalid x coordinate in {txt_file}, line {line_num}"
                                )
                                if self.strict_mode:
                                    return result
                                else:
                                    self._log_warning(
                                        f"Skipping line {line_num}: invalid x coordinate"
                                    )
                                    continue
                            if not self._validate_normalized_coordinate(
                                y, f"point[{i}].y"
                            ):
                                result.add_error(
                                    f"Invalid y coordinate in {txt_file}, line {line_num}"
                                )
                                if self.strict_mode:
                                    return result
                                else:
                                    self._log_warning(
                                        f"Skipping line {line_num}: invalid y coordinate"
                                    )
                                    continue

                        # Check polygon has at least 3 points
                        if len(points) < 3:
                            result.add_error(
                                f"Polygon needs at least 3 points in {txt_file}, line {line_num}"
                            )
                            if self.strict_mode:
                                return result
                            else:
                                self._log_warning(
                                    f"Skipping line {line_num}: polygon has {len(points)} points"
                                )
                                continue

                        segmentation = Segmentation(points=points)
                        if not self._validate_segmentation_points(points):
                            result.add_error(
                                f"Invalid segmentation points in {txt_file}, line {line_num}"
                            )
                            if self.strict_mode:
                                return result
                            else:
                                self._log_warning(
                                    f"Skipping line {line_num}: invalid segmentation"
                                )
                                continue

                    # Create original data for lossless round-trip
                    original_data = OriginalData(
                        format=AnnotationFormat.YOLO.value,
                        raw_data={
                            "line": line,
                            "line_number": line_num,
                            "items": items.copy(),
                            "is_detection": is_detection,
                            "is_segmentation": is_segmentation,
                        },
                    )

                    # Attach original data to components
                    if bbox is not None:
                        bbox.original_data = original_data
                    if segmentation is not None:
                        segmentation.original_data = original_data

                    # Create object annotation
                    obj = ObjectAnnotation(
                        class_id=class_id,
                        class_name=class_name,
                        bbox=bbox,
                        segmentation=segmentation,
                        confidence=1.0,
                        original_data=original_data,
                    )
                    objects.append(obj)

                except (ValueError, IndexError) as e:
                    error_msg = f"Error parsing line {line_num} in {txt_file}: {e}"
                    if self.strict_mode:
                        result.add_error(error_msg)
                        return result
                    else:
                        self._log_warning(f"Skipping line {line_num}: {error_msg}")
                        continue

            # Create image annotation
            image_ann = ImageAnnotation(
                image_id=image_stem,
                image_path=str(image_path),
                width=img_width,
                height=img_height,
                objects=objects,
            )

            result.success = True
            result.data = image_ann

        except Exception as e:
            result.add_error(f"Error reading {txt_file}: {e}")

        return result

    def write(
        self, annotations: DatasetAnnotations, output_dir: str
    ) -> AnnotationResult:
        """Write annotations to YOLO TXT format."""
        result = AnnotationResult(success=False)
        output_path = Path(output_dir)

        try:
            self.file_ops.ensure_dir(output_path)

            written_count = 0
            for image_ann in annotations.images:
                file_result = self._write_single_image(image_ann, output_path)
                if file_result.success:
                    written_count += 1
                elif self.strict_mode:
                    result.add_error(
                        f"Failed to write {image_ann.image_id}: {file_result.message}"
                    )
                    return result
                else:
                    self._log_warning(
                        f"Skipping {image_ann.image_id}: {file_result.message}"
                    )

            result.success = True
            result.message = (
                f"Successfully wrote {written_count}/{len(annotations.images)} images"
            )
            result.data = {
                "output_dir": str(output_path),
                "written_count": written_count,
            }

        except Exception as e:
            result.add_error(f"Unexpected error writing YOLO annotations: {e}")

        return result

    def _write_single_image(
        self, image_ann: ImageAnnotation, output_dir: Path
    ) -> AnnotationResult:
        """Write annotations for a single image to YOLO TXT file."""
        result = AnnotationResult(success=False)

        try:
            output_file = output_dir / f"{image_ann.image_id}.txt"
            lines = []

            for obj in image_ann.objects:
                line = self._object_to_yolo_line(obj, image_ann.width, image_ann.height)
                if line is not None:
                    lines.append(line)
                elif self.strict_mode:
                    result.add_error(
                        f"Failed to convert object {obj.class_name} to YOLO format"
                    )
                    return result
                else:
                    self._log_warning(f"Skipping object {obj.class_name}")

            if lines:
                success = self.file_ops.write_lines(output_file, lines)
                if success:
                    result.success = True
                    result.message = f"Written {output_file} with {len(lines)} objects"
                else:
                    result.add_error(f"Failed to write to {output_file}")
            else:
                # Write empty file if no objects
                success = self.file_ops.write_lines(output_file, [])
                if success:
                    result.success = True
                    result.message = f"Written empty file {output_file}"
                else:
                    result.add_error(f"Failed to write empty file {output_file}")

        except Exception as e:
            result.add_error(f"Error writing image {image_ann.image_id}: {e}")

        return result

    def _object_to_yolo_line(
        self, obj: ObjectAnnotation, img_width: int, img_height: int
    ) -> Optional[str]:
        """Convert ObjectAnnotation to YOLO format line."""
        try:
            # Get class ID (try to find by name if not matching)
            class_id = obj.class_id
            if (
                class_id not in self.categories
                or self.categories[class_id] != obj.class_name
            ):
                # Try to find by name
                found_id = None
                for cat_id, cat_name in self.categories.items():
                    if cat_name == obj.class_name:
                        found_id = cat_id
                        break

                if found_id is None:
                    self._log_warning(
                        f"Class name '{obj.class_name}' not found in categories"
                    )
                    return None
                class_id = found_id

            # Priority 1: Use original data if available and format matches
            if (
                obj.has_original_data()
                and obj.original_data.format == AnnotationFormat.YOLO.value
            ):
                raw_data = obj.original_data.raw_data
                if "line" in raw_data and "items" in raw_data:
                    # Reconstruct line with updated class ID
                    original_items = raw_data["items"]
                    if len(original_items) > 0:
                        # Replace class ID in first position
                        original_items[0] = str(class_id)
                        # Join with single space (preserving original formatting may not be needed)
                        return " ".join(original_items)
                    else:
                        self._log_warning(
                            "Original items empty, falling back to conversion"
                        )
                else:
                    self._log_warning(
                        "Original data missing line or items, falling back to conversion"
                    )

            if obj.bbox:
                # Object detection format: class_id x_center y_center width height
                x_center = obj.bbox.x
                y_center = obj.bbox.y
                width = obj.bbox.width
                height = obj.bbox.height

                # Format: class_id x_center y_center width height
                return (
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

            elif obj.segmentation:
                # Instance segmentation format: class_id x1 y1 x2 y2 ... xn yn
                points = obj.segmentation.points
                coords = []
                for x, y in points:
                    coords.extend([f"{x:.6f}", f"{y:.6f}"])

                # Format: class_id x1 y1 x2 y2 ... xn yn
                return f"{class_id} " + " ".join(coords)

            else:
                self._log_warning(
                    f"Object {obj.class_name} has neither bbox nor segmentation"
                )
                return None

        except Exception as e:
            self._log_error(
                f"Error converting object {obj.class_name} to YOLO format: {e}"
            )
            return None

    def validate(self, annotation_file: str) -> bool:
        """Validate a single YOLO TXT file."""
        try:
            file_path = Path(annotation_file)
            if not file_path.exists():
                self.logger.error(f"Annotation file does not exist: {annotation_file}")
                return False

            # Read file
            lines = self.file_ops.read_lines(file_path)
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                items = line.split()
                if not items:
                    continue

                # Check format
                if len(items) == 5:
                    # Detection format
                    try:
                        class_id = int(items[0])
                        x_center = float(items[1])
                        y_center = float(items[2])
                        width = float(items[3])
                        height = float(items[4])

                        # Check class ID
                        if class_id not in self.categories:
                            self.logger.error(
                                f"Invalid class ID {class_id} in line {line_num}"
                            )
                            return False

                        # Check normalized coordinates
                        for value, name in [
                            (x_center, "x_center"),
                            (y_center, "y_center"),
                            (width, "width"),
                            (height, "height"),
                        ]:
                            if value < 0 or value > 1:
                                self.logger.error(
                                    f"{name} out of range [0, 1] in line {line_num}: {value}"
                                )
                                return False

                    except (ValueError, IndexError) as e:
                        self.logger.error(f"Error parsing line {line_num}: {e}")
                        return False

                elif len(items) > 5 and len(items) % 2 == 1:
                    # Segmentation format
                    try:
                        class_id = int(items[0])
                        if class_id not in self.categories:
                            self.logger.error(
                                f"Invalid class ID {class_id} in line {line_num}"
                            )
                            return False

                        # Check coordinates
                        coords = [float(x) for x in items[1:]]
                        if len(coords) % 2 != 0:
                            self.logger.error(
                                f"Odd number of coordinates in line {line_num}"
                            )
                            return False

                        # Check each coordinate
                        for i in range(0, len(coords), 2):
                            x, y = coords[i], coords[i + 1]
                            if x < 0 or x > 1:
                                self.logger.error(
                                    f"x coordinate out of range [0, 1] in line {line_num}: {x}"
                                )
                                return False
                            if y < 0 or y > 1:
                                self.logger.error(
                                    f"y coordinate out of range [0, 1] in line {line_num}: {y}"
                                )
                                return False

                        # Check polygon has at least 3 points
                        if len(coords) // 2 < 3:
                            self.logger.error(
                                f"Polygon needs at least 3 points in line {line_num}"
                            )
                            return False

                    except (ValueError, IndexError) as e:
                        self.logger.error(f"Error parsing line {line_num}: {e}")
                        return False

                else:
                    self.logger.error(
                        f"Invalid number of items in line {line_num}: {len(items)}"
                    )
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating {annotation_file}: {e}")
            return False
