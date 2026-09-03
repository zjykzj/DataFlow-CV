"""
LabelMe annotation format handler.

Handles reading and writing of LabelMe JSON annotation files.
Coordinates are stored in native LabelMe representation:
- Rectangle bbox: (x, y, w, h) with (x,y) = top-left, all absolute pixels
- Polygon points: (x, y) in absolute pixels
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


from .base import AnnotationResult, BaseAnnotationHandler, ImageError
from .models import (
    AnnotationFormat,
    BoundingBox,
    DatasetAnnotations,
    ImageAnnotation,
    ObjectAnnotation,
    Segmentation,
)


class LabelMeAnnotationHandler(BaseAnnotationHandler):
    """Handler for LabelMe JSON annotation format."""

    def __init__(
        self, label_dir: str, class_file: Optional[str] = None, recursive: bool = False, **kwargs
    ):
        """
        Initialize LabelMe handler.

        Args:
            label_dir: Directory containing LabelMe JSON files
            class_file: Optional file containing class names (one per line)
            recursive: If True, use ``rglob`` for file discovery,
                traversing subdirectories recursively.  Default False.
            **kwargs: Additional arguments for BaseAnnotationHandler
        """
        super().__init__(**kwargs)
        self.label_dir = Path(label_dir)
        self.class_file = Path(class_file) if class_file else None
        self.recursive = recursive
        self.categories = self._load_categories()

    def _list_json_files(self) -> List[Path]:
        """List .json label files, optionally recursively."""
        pattern = self.label_dir.rglob if self.recursive else self.label_dir.glob
        return sorted(f for f in pattern("*.json"))

    def _load_categories(self) -> Dict[int, str]:
        """Load category mapping from class file or extract from annotations."""
        categories = {}

        if self.class_file and self.class_file.exists():
            try:
                lines = Path(self.class_file).read_text(encoding="utf-8").splitlines()
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
            json_files = self._list_json_files()
            if not json_files:
                result.add_error(f"No JSON files found in {self.label_dir}")
                return result

            dataset = DatasetAnnotations(format=AnnotationFormat.LABELME)
            categories_from_annotations = {}

            for json_file in json_files:
                try:
                    image_result = self._read_single_file(json_file)
                except ImageError as e:
                    self._log_warning(f"Skipping {json_file} (image error): {e}")
                    continue

                if not image_result.success:
                    if self.strict_mode:
                        result.add_error(f"Failed to read {json_file}: {image_result.message}")
                        return result
                    else:
                        self._log_warning(f"Skipping {json_file}: {image_result.message}")
                        continue

                image_ann = image_result.data
                if image_ann is None:
                    result.add_error(
                        f"Internal error: image annotation data is None for {json_file}"
                    )
                    return result
                if not isinstance(image_ann, ImageAnnotation):
                    result.add_error(
                        f"Internal error: invalid image annotation type for {json_file}"
                    )
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

    def iter_images(self) -> Iterator[ImageAnnotation]:
        """Yield LabelMe ImageAnnotation objects one at a time (streaming).

        Reuses ``_read_single_file()`` for per-file parsing — same validation
        logic as ``read()`` but yields incrementally instead of accumulating
        into a ``DatasetAnnotations``.

        Categories are auto-extracted from annotations when ``class_file``
        is not provided.

        Yields:
            ImageAnnotation with LabelMe-native absolute-pixel coordinates.

        Raises:
            ValueError: Structural errors (bad directory, no files) raise
                immediately.  Per-file parse errors raise in strict mode.
        """
        from .models import ImageAnnotation as IA

        if not self.label_dir.exists():
            raise ValueError(f"Label directory does not exist: {self.label_dir}")

        json_files = self._list_json_files()
        if not json_files:
            raise ValueError(f"No JSON files found in {self.label_dir}")

        categories_from_annotations: Dict[int, str] = {}

        for json_file in json_files:
            try:
                image_result = self._read_single_file(json_file)
            except ImageError as e:
                self._log_warning(f"Skipping {json_file} (image error): {e}")
                continue

            if not image_result.success:
                if self.strict_mode:
                    raise ValueError(f"Failed to read {json_file}: {image_result.message}")
                else:
                    self._log_warning(f"Skipping {json_file}: {image_result.message}")
                    continue

            image_ann = image_result.data
            if image_ann is None:
                raise ValueError(f"Internal error: image annotation data is None for {json_file}")
            if not isinstance(image_ann, IA):
                raise ValueError(f"Internal error: invalid image annotation type for {json_file}")

            # Extract categories from this image
            for obj in image_ann.objects:
                if obj.class_id not in categories_from_annotations:
                    categories_from_annotations[obj.class_id] = obj.class_name

            yield image_ann

        # Update categories from annotations if no class_file was provided
        if not self.categories:
            self.categories = categories_from_annotations
            self._log_info(f"Extracted {len(self.categories)} categories from annotations")

    def _read_single_file(self, json_file: Path) -> AnnotationResult:
        """Read a single LabelMe JSON file."""
        result = AnnotationResult(success=False)

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            required_fields = ["version", "flags", "shapes", "imagePath"]
            for field in required_fields:
                if field not in data:
                    result.add_error(f"Missing required field '{field}' in {json_file}")
                    return result

            # Get image info
            image_path = Path(data["imagePath"])
            if not image_path.is_absolute():
                image_path = json_file.parent / image_path

            # Try to get image dimensions
            image_height = data.get("imageHeight")
            image_width = data.get("imageWidth")

            if not image_path.exists():
                if image_height is None or image_width is None:
                    raise ImageError(
                        f"Image file not found and no dimensions in JSON: {image_path}"
                    )

            if image_height is None or image_width is None:
                dims_read = False
                if image_path.exists():
                    try:
                        import cv2

                        img = cv2.imread(str(image_path))
                        if img is not None:
                            image_height, image_width = img.shape[:2]
                            dims_read = True
                    except Exception as e:
                        self._log_debug(f"Could not read image dimensions from file: {e}")
                if not dims_read:
                    raise ImageError(
                        f"Image dimensions not in JSON {json_file} and could not read from file"
                    )

            if not self._validate_image_dimensions(image_width, image_height):
                raise ImageError(f"Invalid image dimensions in {json_file}")

            # Process shapes
            objects: List[ObjectAnnotation] = []
            for shape in data["shapes"]:
                obj_result = self._parse_shape(shape, image_width, image_height)
                if obj_result.success:
                    obj_data = obj_result.data
                    if obj_data is None or not isinstance(obj_data, ObjectAnnotation):
                        result.add_error(
                            f"Internal error: invalid object data for shape in {json_file}"
                        )
                        return result
                    objects.append(obj_data)
                elif self.strict_mode:
                    result.add_error(f"Failed to parse shape in {json_file}: {obj_result.message}")
                    return result
                else:
                    self._log_warning(
                        f"Skipping invalid shape in {json_file}: {obj_result.message}"
                    )

            # Create image annotation
            try:
                relative_image_path = image_path.relative_to(self.label_dir)
                image_path_str = str(relative_image_path)
            except ValueError:
                image_path_str = str(image_path)
                self._log_warning(
                    f"Image path {image_path} is not relative to label directory {self.label_dir}"
                )

            image_ann = ImageAnnotation(
                image_id=json_file.stem,
                image_path=image_path_str,
                width=image_width,
                height=image_height,
                objects=objects,
            )

            result.success = True
            result.data = image_ann

        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON in {json_file}: {e}")
        except ImageError:
            raise
        except Exception as e:
            result.add_error(f"Error reading {json_file}: {e}")

        return result

    def _parse_shape(self, shape: Dict, img_width: int, img_height: int) -> AnnotationResult:
        """Parse a single LabelMe shape into ObjectAnnotation."""
        result = AnnotationResult(success=False)

        try:
            label = shape.get("label", "").strip()
            if not label:
                result.add_error("Shape missing label")
                return result

            shape_type = shape.get("shape_type", "").lower()
            points = shape.get("points", [])

            # Determine category ID
            cat_id: int
            if label in self.categories.values():
                found_id = next((k for k, v in self.categories.items() if v == label), None)
                if found_id is None:
                    result.add_error(f"Internal error: label '{label}' not found in categories")
                    return result
                cat_id = found_id
            else:
                cat_id = len(self.categories)
                self.categories[cat_id] = label

            bbox = None
            segmentation = None

            if shape_type == "rectangle" and len(points) == 2:
                # Convert rectangle to bounding box (absolute pixels, top-left)
                x1, y1 = points[0]
                x2, y2 = points[1]

                x_min = min(x1, x2)
                y_min = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)

                # Clamp to image boundaries (tolerates FP imprecision at edges)
                x_min, y_min, w, h = self._clamp_abs_bbox(x_min, y_min, w, h, img_width, img_height)

                bbox = BoundingBox(x=x_min, y=y_min, width=w, height=h)
                if not self._validate_bbox(bbox, format=AnnotationFormat.LABELME):
                    result.add_error(f"Invalid bbox for rectangle: {points}")
                    return result

            elif shape_type == "polygon" and len(points) >= 3:
                # Store polygon points in absolute pixels (native LabelMe format)
                abs_points = [(x, y) for x, y in points]
                # Clamp to image boundaries (tolerates FP imprecision at edges)
                abs_points = self._clamp_abs_points(abs_points, img_width, img_height)
                if not self._validate_segmentation_points(
                    abs_points, format=AnnotationFormat.LABELME
                ):
                    result.add_error(f"Invalid polygon points: {points}")
                    return result
                segmentation = Segmentation(points=abs_points)

            else:
                result.add_error(f"Unsupported shape type '{shape_type}' with {len(points)} points")
                return result

            obj = ObjectAnnotation(
                class_id=cat_id,
                class_name=label,
                bbox=bbox,
                segmentation=segmentation,
                confidence=1.0,
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
            Path(output_path).mkdir(parents=True, exist_ok=True)

            written_count = 0
            for image_ann in annotations.images:
                file_result = self.write_one(image_ann, output_path)
                if file_result.success:
                    written_count += 1
                elif self.strict_mode:
                    result.add_error(f"Failed to write {image_ann.image_id}: {file_result.message}")
                    return result
                else:
                    self._log_warning(f"Skipping {image_ann.image_id}: {file_result.message}")

            result.success = True
            result.message = f"Successfully wrote {written_count}/{len(annotations.images)} images"
            result.data = {
                "output_dir": str(output_path),
                "written_count": written_count,
            }

        except Exception as e:
            result.add_error(f"Unexpected error writing LabelMe annotations: {e}")

        return result

    def write_one(self, image_ann: ImageAnnotation, output_dir: Path) -> AnnotationResult:
        """Write annotations for a single image to LabelMe JSON."""
        result = AnnotationResult(success=False)

        try:
            # Prepare shapes
            shapes = []
            for obj in image_ann.objects:
                shape = self._object_to_shape(obj)
                if shape:
                    shapes.append(shape)
                elif self.strict_mode:
                    result.add_error(f"Failed to convert object {obj.class_name} to LabelMe shape")
                    return result
                else:
                    self._log_warning(f"Skipping object {obj.class_name}")

            # Build LabelMe JSON structure
            labelme_data: Dict[str, Any] = {
                "version": "5.0.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": Path(image_ann.image_path).name,
                "imageData": None,
                "imageHeight": image_ann.height,
                "imageWidth": image_ann.width,
            }

            # Validate image_id is safe for path construction (defense in depth)
            self._validate_image_id_for_path(image_ann.image_id)

            # Write JSON file
            output_file = output_dir / f"{image_ann.image_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)

            result.success = True
            result.message = f"Written {output_file}"

        except Exception as e:
            result.add_error(f"Error writing image {image_ann.image_id}: {e}")

        return result

    def _object_to_shape(self, obj: ObjectAnnotation) -> Optional[Dict]:
        """Convert ObjectAnnotation to LabelMe shape dict.

        Coordinates are already in native LabelMe format (absolute pixels).
        """
        try:
            label = obj.class_name

            if obj.bbox:
                # Bbox is already in absolute pixel, top-left format
                x1 = obj.bbox.x
                y1 = obj.bbox.y
                x2 = obj.bbox.x + obj.bbox.width
                y2 = obj.bbox.y + obj.bbox.height
                points = [[float(x1), float(y1)], [float(x2), float(y2)]]
                shape_type = "rectangle"
            elif obj.segmentation:
                # Points are already in absolute pixels
                points = [[float(x), float(y)] for x, y in obj.segmentation.points]
                shape_type = "polygon"
            else:
                self._log_warning(f"Object {label} has neither bbox nor segmentation")
                return None

            return {
                "label": label,
                "points": points,
                "group_id": None,
                "shape_type": shape_type,
                "flags": {},
            }

        except Exception as e:
            self._log_error(f"Error converting object {obj.class_name} to shape: {e}")
            return None

    def validate(self, annotation_file: str) -> bool:
        """Validate a single LabelMe JSON file."""
        try:
            with open(annotation_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            required_fields = ["version", "flags", "shapes", "imagePath"]
            for field in required_fields:
                if field not in data:
                    self.logger.error(f"Missing required field '{field}' in {annotation_file}")
                    return False

            for shape in data["shapes"]:
                if "label" not in shape or not shape["label"].strip():
                    self.logger.error(f"Shape missing label in {annotation_file}")
                    return False
                if "shape_type" not in shape:
                    self.logger.error(f"Shape missing shape_type in {annotation_file}")
                    return False
                if "points" not in shape:
                    self.logger.error(f"Shape missing points in {annotation_file}")
                    return False

                shape_type = shape["shape_type"]
                points = shape["points"]

                if shape_type not in ("rectangle", "polygon", "circle", "line", "point"):
                    self.logger.error(f"Unsupported shape_type '{shape_type}' in {annotation_file}")
                    return False

                if shape_type == "rectangle":
                    if len(points) != 2:
                        self.logger.error(
                            f"Rectangle shape must have exactly 2 points, got {len(points)} in {annotation_file}"
                        )
                        return False

                if shape_type == "polygon":
                    if len(points) < 3:
                        self.logger.error(
                            f"Polygon shape must have at least 3 points, got {len(points)} in {annotation_file}"
                        )
                        return False

            return True

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {annotation_file}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error validating {annotation_file}: {e}")
            return False
