"""
LabelMe annotation format handler.

Handles reading and writing of LabelMe JSON annotation files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sys

from .base import BaseAnnotationHandler, AnnotationResult
from .models import (
    DatasetAnnotations, ImageAnnotation, ObjectAnnotation,
    BoundingBox, Segmentation, OriginalData, AnnotationFormat
)
from dataflow.util.file_util import FileOperations


class LabelMeAnnotationHandler(BaseAnnotationHandler):
    """Handler for LabelMe JSON annotation format."""

    def __init__(self, label_dir: str, class_file: Optional[str] = None, **kwargs):
        """
        Initialize LabelMe handler.

        Args:
            label_dir: Directory containing LabelMe JSON files
            class_file: Optional file containing class names (one per line)
            **kwargs: Additional arguments for BaseAnnotationHandler
        """
        super().__init__(**kwargs)
        self.label_dir = Path(label_dir)
        self.class_file = Path(class_file) if class_file else None
        self.file_ops = FileOperations(logger=self.logger)
        self.categories = self._load_categories()

    def _load_categories(self) -> Dict[int, str]:
        """Load category mapping from class file or extract from annotations."""
        categories = {}

        if self.class_file and self.class_file.exists():
            try:
                lines = self.file_ops.read_lines(self.class_file)
                for i, line in enumerate(lines):
                    if line.strip():  # Skip empty lines
                        categories[i] = line.strip()
                self._log_info(f"Loaded {len(categories)} categories from {self.class_file}")
            except Exception as e:
                self._log_error(f"Failed to load class file {self.class_file}: {e}")
        else:
            self._log_info("No class file provided, will extract categories from annotations")

        return categories

    def read(self) -> AnnotationResult:
        """Read all LabelMe JSON files in the directory."""
        result = AnnotationResult(success=False)

        if not self.label_dir.exists():
            result.add_error(f"Label directory does not exist: {self.label_dir}")
            return result

        try:
            json_files = self.file_ops.find_files(self.label_dir, "*.json", recursive=False)
            if not json_files:
                result.add_error(f"No JSON files found in {self.label_dir}")
                return result

            dataset = DatasetAnnotations()
            categories_from_annotations = {}

            for json_file in json_files:
                image_result = self._read_single_file(json_file)
                if not image_result.success:
                    if self.strict_mode:
                        result.add_error(f"Failed to read {json_file}: {image_result.message}")
                        return result
                    else:
                        self._log_warning(f"Skipping {json_file}: {image_result.message}")
                        continue

                image_ann = image_result.data
                if image_ann is None:
                    # This should not happen if image_result.success is True
                    result.add_error(f"Internal error: image annotation data is None for {json_file}")
                    return result
                # Type check for mypy
                if not isinstance(image_ann, ImageAnnotation):
                    result.add_error(f"Internal error: invalid image annotation type for {json_file}")
                    return result
                dataset.add_image(image_ann)

                # Extract categories from this image
                for obj in image_ann.objects:
                    if obj.class_id not in categories_from_annotations:
                        categories_from_annotations[obj.class_id] = obj.class_name

            # Update categories: use provided class file if available, otherwise use extracted
            if not self.categories:
                self.categories = categories_from_annotations
                self._log_info(f"Extracted {len(self.categories)} categories from annotations")

            dataset.categories = self.categories

            # Set annotation flags
            self._set_annotation_flags(dataset)

            result.success = True
            result.data = dataset
            result.message = f"Successfully read {len(dataset.images)} images"

        except Exception as e:
            result.add_error(f"Unexpected error reading LabelMe annotations: {e}")

        return result

    def _read_single_file(self, json_file: Path) -> AnnotationResult:
        """Read a single LabelMe JSON file."""
        result = AnnotationResult(success=False)

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate required fields
            required_fields = ['version', 'flags', 'shapes', 'imagePath', 'imageData']
            for field in required_fields:
                if field not in data:
                    result.add_error(f"Missing required field '{field}' in {json_file}")
                    return result

            # Get image info
            image_path = Path(data['imagePath'])
            if not image_path.is_absolute():
                image_path = json_file.parent / image_path

            # Try to get image dimensions
            image_height = data.get('imageHeight')
            image_width = data.get('imageWidth')

            if image_height is None or image_width is None:
                self._log_warning(f"Image dimensions not in JSON {json_file}, using defaults")
                image_height = 1
                image_width = 1

            if not self._validate_image_dimensions(image_width, image_height):
                result.add_error(f"Invalid image dimensions in {json_file}")
                return result

            # Process shapes
            objects: List[ObjectAnnotation] = []
            for shape in data['shapes']:
                obj_result = self._parse_shape(shape, image_width, image_height)
                if obj_result.success:
                    obj_data = obj_result.data
                    if obj_data is None or not isinstance(obj_data, ObjectAnnotation):
                        # This should not happen if obj_result.success is True
                        result.add_error(f"Internal error: invalid object data for shape in {json_file}")
                        return result
                    objects.append(obj_data)
                elif self.strict_mode:
                    result.add_error(f"Failed to parse shape in {json_file}: {obj_result.message}")
                    return result
                else:
                    self._log_warning(f"Skipping invalid shape in {json_file}: {obj_result.message}")

            # Create original data for the entire image annotation
            image_original_data = OriginalData(
                format=AnnotationFormat.LABELME.value,
                raw_data=data.copy()
            )

            # Create image annotation
            image_ann = ImageAnnotation(
                image_id=json_file.stem,
                image_path=str(image_path),
                width=image_width,
                height=image_height,
                objects=objects,
                original_data=image_original_data
            )

            result.success = True
            result.data = image_ann

        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON in {json_file}: {e}")
        except Exception as e:
            result.add_error(f"Error reading {json_file}: {e}")

        return result

    def _parse_shape(self, shape: Dict, img_width: int, img_height: int) -> AnnotationResult:
        """Parse a single LabelMe shape into ObjectAnnotation."""
        result = AnnotationResult(success=False)

        try:
            label = shape.get('label', '').strip()
            if not label:
                result.add_error("Shape missing label")
                return result

            shape_type = shape.get('shape_type', '').lower()
            points = shape.get('points', [])

            # Determine category ID
            cat_id: int
            if label in self.categories.values():
                # Find ID for this label
                found_id = next((k for k, v in self.categories.items() if v == label), None)
                if found_id is None:
                    # This should not happen if label is in values
                    result.add_error(f"Internal error: label '{label}' not found in categories")
                    return result
                cat_id = found_id
            else:
                # Assign new ID
                cat_id = len(self.categories)
                self.categories[cat_id] = label

            bbox = None
            segmentation = None

            if shape_type == 'rectangle' and len(points) == 2:
                # Convert rectangle to bounding box
                x1, y1 = points[0]
                x2, y2 = points[1]

                # Normalize coordinates
                x1_norm = x1 / img_width
                y1_norm = y1 / img_height
                x2_norm = x2 / img_width
                y2_norm = y2 / img_height

                # Calculate center, width, height
                x_center = (x1_norm + x2_norm) / 2
                y_center = (y1_norm + y2_norm) / 2
                width = abs(x2_norm - x1_norm)
                height = abs(y2_norm - y1_norm)

                bbox = BoundingBox(x=x_center, y=y_center, width=width, height=height)
                if not self._validate_bbox(bbox):
                    result.add_error(f"Invalid bbox for rectangle: {points}")
                    return result

            elif shape_type == 'polygon' and len(points) >= 3:
                # Normalize polygon points
                normalized_points = [(x / img_width, y / img_height) for x, y in points]
                if not self._validate_segmentation_points(normalized_points):
                    result.add_error(f"Invalid polygon points: {points}")
                    return result
                segmentation = Segmentation(points=normalized_points)

            else:
                result.add_error(f"Unsupported shape type '{shape_type}' with {len(points)} points")
                return result

            # Create original data for lossless round-trip
            original_data = OriginalData(
                format=AnnotationFormat.LABELME.value,
                raw_data=shape.copy(),
                metadata={"image_width": img_width, "image_height": img_height}
            )

            # Attach original data to components
            if bbox is not None:
                bbox.original_data = original_data
            if segmentation is not None:
                segmentation.original_data = original_data

            obj = ObjectAnnotation(
                class_id=cat_id,
                class_name=label,
                bbox=bbox,
                segmentation=segmentation,
                confidence=1.0,
                original_data=original_data
            )

            result.success = True
            result.data = obj

        except Exception as e:
            result.add_error(f"Error parsing shape: {e}")

        return result

    def write(self, annotations: DatasetAnnotations, output_dir: str) -> AnnotationResult:
        """Write annotations to LabelMe JSON format."""
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
                    result.add_error(f"Failed to write {image_ann.image_id}: {file_result.message}")
                    return result
                else:
                    self._log_warning(f"Skipping {image_ann.image_id}: {file_result.message}")

            result.success = True
            result.message = f"Successfully wrote {written_count}/{len(annotations.images)} images"
            result.data = {"output_dir": str(output_path), "written_count": written_count}

        except Exception as e:
            result.add_error(f"Unexpected error writing LabelMe annotations: {e}")

        return result

    def _write_single_image(self, image_ann: ImageAnnotation, output_dir: Path) -> AnnotationResult:
        """Write annotations for a single image to LabelMe JSON."""
        result = AnnotationResult(success=False)

        try:
            # Prepare shapes
            shapes = []
            for obj in image_ann.objects:
                shape = self._object_to_shape(obj, image_ann.width, image_ann.height)
                if shape:
                    shapes.append(shape)
                elif self.strict_mode:
                    result.add_error(f"Failed to convert object {obj.class_name} to LabelMe shape")
                    return result
                else:
                    self._log_warning(f"Skipping object {obj.class_name}")

            # Create LabelMe JSON structure
            # Priority 1: Use original data if available and format matches
            if image_ann.has_original_data() and image_ann.original_data.format == AnnotationFormat.LABELME.value:
                # Start with original data to preserve all fields (mask, lineColor, etc.)
                labelme_data = image_ann.original_data.raw_data.copy()
                # Update shapes with current objects
                labelme_data["shapes"] = shapes
                # Update image path (use original if available, otherwise current)
                if "imagePath" not in labelme_data:
                    labelme_data["imagePath"] = Path(image_ann.image_path).name
                # Update dimensions to match current image annotation
                labelme_data["imageHeight"] = image_ann.height
                labelme_data["imageWidth"] = image_ann.width
                # Ensure imageData is None (LabelMe doesn't store image data)
                labelme_data["imageData"] = None
            else:
                # Fallback to default structure
                labelme_data: Dict[str, Any] = {
                    "version": "5.0.1",
                    "flags": {},
                    "shapes": shapes,
                    "imagePath": Path(image_ann.image_path).name,
                    "imageData": None,  # LabelMe doesn't store image data
                    "imageHeight": image_ann.height,
                    "imageWidth": image_ann.width
                }

            # Write JSON file
            output_file = output_dir / f"{image_ann.image_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)

            result.success = True
            result.message = f"Written {output_file}"

        except Exception as e:
            result.add_error(f"Error writing image {image_ann.image_id}: {e}")

        return result

    def _object_to_shape(self, obj: ObjectAnnotation, img_width: int, img_height: int) -> Optional[Dict]:
        """Convert ObjectAnnotation to LabelMe shape dict."""
        try:
            # Priority 1: Use original data if available and format matches
            if obj.has_original_data() and obj.original_data.format == AnnotationFormat.LABELME.value:
                shape_data = obj.original_data.raw_data.copy()
                # Update label to current class name (class mapping may have changed)
                shape_data["label"] = obj.class_name
                # Ensure points are present (they should be)
                if "points" not in shape_data or not shape_data["points"]:
                    self._log_warning(f"Original shape data missing points, falling back to conversion")
                else:
                    return shape_data

            label = obj.class_name

            if obj.bbox:
                # Convert bbox to rectangle points
                x1, y1, x2, y2 = obj.bbox.xyxy(img_width, img_height)
                points = [[float(x1), float(y1)], [float(x2), float(y2)]]
                shape_type = "rectangle"
            elif obj.segmentation:
                # Convert segmentation to polygon points
                points = [[float(x), float(y)] for x, y in obj.segmentation.points_abs(img_width, img_height)]
                shape_type = "polygon"
            else:
                self._log_warning(f"Object {label} has neither bbox nor segmentation")
                return None

            return {
                "label": label,
                "points": points,
                "group_id": None,
                "shape_type": shape_type,
                "flags": {}
            }

        except Exception as e:
            self._log_error(f"Error converting object {obj.class_name} to shape: {e}")
            return None

    def validate(self, annotation_file: str) -> bool:
        """Validate a single LabelMe JSON file."""
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            required_fields = ['version', 'flags', 'shapes', 'imagePath']
            for field in required_fields:
                if field not in data:
                    self._log_error(f"Missing required field '{field}' in {annotation_file}")
                    return False

            # Validate shapes
            for shape in data['shapes']:
                if 'label' not in shape or not shape['label'].strip():
                    self._log_error(f"Shape missing label in {annotation_file}")
                    return False
                if 'shape_type' not in shape:
                    self._log_error(f"Shape missing shape_type in {annotation_file}")
                    return False
                if 'points' not in shape:
                    self._log_error(f"Shape missing points in {annotation_file}")
                    return False

            return True

        except json.JSONDecodeError as e:
            self._log_error(f"Invalid JSON in {annotation_file}: {e}")
            return False
        except Exception as e:
            self._log_error(f"Error validating {annotation_file}: {e}")
            return False