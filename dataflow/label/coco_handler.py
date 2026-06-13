#!/usr/bin/env python3
"""
COCO annotation format handler.

Handles reading and writing of COCO format annotation files.
Supports both polygon and RLE segmentation formats.
Coordinates are stored in native COCO representation:
- Bbox: (x, y, w, h) with (x,y) = top-left, all absolute pixels
- Polygon points: (x, y) in absolute pixels
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

try:
    from pycocotools import mask as coco_mask

    HAS_COCO_MASK = True
except ImportError:
    HAS_COCO_MASK = False

import cv2
import numpy as np


from .base import AnnotationResult, BaseAnnotationHandler
from .models import (AnnotationFormat, BoundingBox, DatasetAnnotations,
                     ImageAnnotation, ObjectAnnotation, Segmentation)


class CocoAnnotationHandler(BaseAnnotationHandler):
    """Handler for COCO annotation format."""

    def __init__(self, annotation_file: str, do_rle: bool = False, prediction: bool = False, **kwargs):
        """
        Initialize COCO handler.

        Args:
            annotation_file: Path to COCO JSON annotation file
            do_rle: Whether to output RLE format when writing (default False)
            prediction: Whether to output prediction format (plain JSON list
                of annotation dicts) instead of full COCO dict. When True,
                ``score`` is always included in each annotation.
                Default False (annotation format).
            **kwargs: Additional arguments for BaseAnnotationHandler
        """
        super().__init__(**kwargs)
        self.annotation_file = Path(annotation_file)
        self.categories = {}
        self.images = {}
        self.annotations = []
        self.dataset_info = {}
        self.output_rle = do_rle  # Whether to output RLE format when writing
        self.prediction = prediction  # Whether to output prediction (list) format

    def read(self) -> AnnotationResult:
        """
        Read COCO JSON annotation file.

        Returns:
            AnnotationResult: Result containing parsed annotations if successful

        Notes:
            - Supports both polygon and RLE segmentation formats
            - Automatically detects format (polygon vs RLE)
            - Stores coordinates in native COCO format (absolute pixels)
            - Handles optional pycocotools dependency gracefully
        """
        result = AnnotationResult(success=False)

        if not self.annotation_file.exists():
            result.add_error(f"Annotation file does not exist: {self.annotation_file}")
            return result

        try:
            with open(self.annotation_file, "r", encoding="utf-8") as f:
                coco_data = json.load(f)

            # Validate required top-level fields
            required_fields = ["images", "annotations", "categories"]
            for field in required_fields:
                if field not in coco_data:
                    result.add_error(
                        f"Missing required field '{field}' in {self.annotation_file}"
                    )
                    return result

            # Load dataset info (optional metadata like info, licenses)
            self.dataset_info = {
                k: v
                for k, v in coco_data.items()
                if k not in ["images", "annotations", "categories"]
            }

            # Load categories
            self.categories = self._load_categories(coco_data["categories"])

            # Load images
            self.images = self._load_images(coco_data["images"])

            # Load annotations
            self.annotations = coco_data["annotations"]

            # Detect RLE format
            self.is_rle = self._detect_rle_format(self.annotations)
            self.output_rle = self.is_rle  # Default to same format as input

            # Create dataset annotations
            dataset = self._create_dataset()

            # Set annotation flags
            self._set_annotation_flags(dataset)

            result.success = True
            result.data = dataset
            result.message = f"Successfully read COCO annotations: {len(dataset.images)} images, {dataset.num_objects} objects"

        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON in {self.annotation_file}: {e}")
        except Exception as e:
            result.add_error(f"Unexpected error reading COCO annotations: {e}")

        return result

    def iter_images(self) -> Iterator[ImageAnnotation]:
        """Yield COCO ImageAnnotation objects one at a time (streaming).

        The full COCO JSON is loaded upfront (required by the single-file
        format), but images are yielded one at a time so callers can start
        processing before all grouping is complete.

        Yields:
            ImageAnnotation with COCO-native absolute-pixel coordinates.

        Raises:
            ValueError: Structural errors (bad file, missing fields, no
                categories) raise immediately.
        """
        from .models import ImageAnnotation as IA

        if not self.annotation_file.exists():
            raise ValueError(
                f"Annotation file does not exist: {self.annotation_file}"
            )

        try:
            with open(self.annotation_file, "r", encoding="utf-8") as f:
                coco_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in {self.annotation_file}: {e}"
            )

        # Validate required fields
        required_fields = ["images", "annotations", "categories"]
        for field in required_fields:
            if field not in coco_data:
                raise ValueError(
                    f"Missing required field '{field}' in "
                    f"{self.annotation_file}"
                )

        # Load metadata
        self.categories = self._load_categories(coco_data["categories"])
        if not self.categories:
            raise ValueError(
                f"No categories found in {self.annotation_file}"
            )

        images_meta = self._load_images(coco_data["images"])
        if not images_meta:
            raise ValueError(
                f"No images found in {self.annotation_file}"
            )

        annotations_list = coco_data["annotations"]
        self.dataset_info = {
            k: v
            for k, v in coco_data.items()
            if k not in ["images", "annotations", "categories"]
        }
        self.is_rle = self._detect_rle_format(annotations_list)

        # Group annotations by image_id
        anns_by_image: Dict[int, List[Dict]] = {}
        for ann in annotations_list:
            img_id = ann.get("image_id")
            if img_id is not None:
                anns_by_image.setdefault(img_id, []).append(ann)

        # Yield per image
        for img_id, img_info in images_meta.items():
            img_anns = anns_by_image.get(img_id, [])
            objects = self._create_objects(
                img_anns, img_info["width"], img_info["height"]
            )
            image_ann = IA(
                image_id=str(img_id),
                image_path=img_info["file_name"],
                width=img_info["width"],
                height=img_info["height"],
                objects=objects,
            )
            yield image_ann

    def _load_categories(self, coco_categories: List[Dict]) -> Dict[int, str]:
        """Load category mapping from COCO categories list."""
        categories = {}
        for cat in coco_categories:
            cat_id = cat.get("id")
            cat_name = cat.get("name", "")
            if cat_id is not None:
                categories[cat_id] = cat_name
        return categories

    def _load_images(self, coco_images: List[Dict]) -> Dict[int, Dict]:
        """Load image information from COCO images list."""
        images = {}
        for img in coco_images:
            img_id = img.get("id")
            if img_id is not None:
                images[img_id] = {
                    "file_name": img.get("file_name", ""),
                    "width": img.get("width", 0),
                    "height": img.get("height", 0),
                    "coco_url": img.get("coco_url", ""),
                    "flickr_url": img.get("flickr_url", ""),
                }
        return images

    def _detect_rle_format(self, annotations: List[Dict]) -> bool:
        """Detect if annotations contain RLE format segmentation."""
        for ann in annotations:
            if "segmentation" in ann:
                seg = ann["segmentation"]
                if isinstance(seg, dict) and "counts" in seg:
                    return True
        return False

    def _create_dataset(self) -> DatasetAnnotations:
        """Create DatasetAnnotations from loaded COCO data."""
        dataset = DatasetAnnotations(format=AnnotationFormat.COCO)
        dataset.categories = self.categories.copy()
        dataset.dataset_info = self.dataset_info.copy()

        for img_id, img_info in self.images.items():
            # Find annotations for this image
            img_anns = [
                ann for ann in self.annotations if ann.get("image_id") == img_id
            ]

            objects = self._create_objects(
                img_anns, img_info["width"], img_info["height"]
            )

            image_ann = ImageAnnotation(
                image_id=str(img_id),
                image_path=img_info["file_name"],
                width=img_info["width"],
                height=img_info["height"],
                objects=objects,
            )
            dataset.add_image(image_ann)

        return dataset

    def _create_objects(
        self, img_anns: List[Dict], img_width: int, img_height: int
    ) -> List[ObjectAnnotation]:
        """Create ObjectAnnotations from COCO annotations for a single image."""
        objects = []

        for ann in img_anns:
            try:
                class_id = ann.get("category_id")
                if class_id is None or class_id not in self.categories:
                    error_msg = f"Invalid category_id in annotation {ann.get('id')}: {ann.get('category_id')}"
                    if self.strict_mode:
                        self._log_error(error_msg)
                        return []
                    else:
                        self._log_warning(f"Skipping annotation: {error_msg}")
                        continue

                class_name = self.categories[class_id]
                is_crowd = ann.get("iscrowd", 0) == 1
                bbox = None
                segmentation = None

                # Parse bbox if present
                # COCO bbox: [x, y, width, height] in absolute pixels, top-left
                if "bbox" in ann and ann["bbox"]:
                    bbox_data = ann["bbox"]
                    if len(bbox_data) == 4:
                        x, y, w, h = bbox_data
                        bbox = BoundingBox(x=x, y=y, width=w, height=h)
                        if not self._validate_bbox(bbox, format=AnnotationFormat.COCO):
                            self._log_warning(
                                f"Skipping annotation {ann.get('id')}: invalid bbox"
                            )
                            continue

                # Parse segmentation
                if "segmentation" in ann and ann["segmentation"]:
                    seg_data = ann["segmentation"]
                    if isinstance(seg_data, dict) and "counts" in seg_data:
                        # RLE format - preserve original RLE data
                        if "size" not in seg_data:
                            self._log_warning(
                                f"RLE segmentation missing 'size' in annotation {ann.get('id')}, skipping"
                            )
                            continue
                        size = seg_data["size"]
                        if not isinstance(size, list) or len(size) != 2:
                            self._log_warning(
                                f"RLE 'size' must be [height, width], got {size} in annotation {ann.get('id')}, skipping"
                            )
                            continue
                        if not all(isinstance(v, int) and v > 0 for v in size):
                            self._log_warning(
                                f"RLE 'size' values must be positive integers in annotation {ann.get('id')}, skipping"
                            )
                            continue
                        if not isinstance(seg_data["counts"], str) or not seg_data["counts"]:
                            self._log_warning(
                                f"RLE 'counts' must be a non-empty string in annotation {ann.get('id')}, skipping"
                            )
                            continue
                        rle_dict = seg_data
                        points = []
                        if HAS_COCO_MASK:
                            try:
                                points = self._decode_rle_to_polygon(
                                    seg_data, img_width, img_height
                                )
                            except Exception as e:
                                self._log_warning(
                                    f"Failed to decode RLE for annotation {ann.get('id')}: {e}"
                                )
                        else:
                            self._log_warning(
                                f"pycocotools not available, preserving RLE without decoding for annotation {ann.get('id')}"
                            )
                        segmentation = Segmentation(points=points, rle=rle_dict)
                    elif isinstance(seg_data, list) and len(seg_data) > 0:
                        # Polygon format (list of lists)
                        points = self._parse_polygon_segmentation(seg_data)
                        if points:
                            segmentation = Segmentation(points=points, rle=None)
                        else:
                            self._log_warning(
                                f"Invalid polygon segmentation in annotation {ann.get('id')}"
                            )

                # Parse confidence/score (prediction JSONs include "score" field)
                confidence = ann.get("score", 1.0)

                # Create object annotation
                obj = ObjectAnnotation(
                    class_id=class_id,
                    class_name=class_name,
                    bbox=bbox,
                    segmentation=segmentation,
                    confidence=confidence,
                    is_crowd=is_crowd,
                )
                objects.append(obj)

            except Exception as e:
                self._log_warning(f"Error processing annotation {ann.get('id')}: {e}")
                continue

        return objects

    def _decode_rle_to_polygon(
        self, rle: Dict, img_width: int, img_height: int
    ) -> List[Tuple[float, float]]:
        """
        Decode RLE to polygon point list.

        Args:
            rle: RLE dict with 'size' and 'counts' fields
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            List[Tuple[float, float]]: Absolute pixel polygon points

        Raises:
            ImportError: If pycocotools is not available
        """
        if not HAS_COCO_MASK:
            raise ImportError("pycocotools required for RLE decoding")

        try:
            rle_dict = dict(rle)

            # Ensure 'counts' is bytes for coco_mask.decode
            if "counts" in rle_dict and isinstance(rle_dict["counts"], str):
                rle_dict["counts"] = rle_dict["counts"].encode("latin1")

            # Decode RLE to binary mask
            binary_mask = coco_mask.decode(rle_dict)

            # Extract contours from mask
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                self._log_warning("RLE decode produced no contours, returning empty polygon")
                return []

            # Use the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Return contour points as absolute pixel coordinates
            points = []
            for point in largest_contour:
                x, y = point[0]
                points.append((float(x), float(y)))

            return points

        except Exception as e:
            self._log_error(f"Error decoding RLE: {e}")
            return []

    def _encode_polygon_to_rle(
        self, points: List[Tuple[float, float]], img_width: int, img_height: int
    ) -> Dict:
        """
        Encode polygon points to RLE format.

        Args:
            points: List of absolute pixel (x, y) polygon points
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Dict: RLE dict with 'size' and 'counts' fields (JSON-serializable)

        Raises:
            ImportError: If pycocotools is not available
        """
        if not HAS_COCO_MASK:
            raise ImportError("pycocotools required for RLE encoding")

        try:
            # Convert to integer coordinates for mask creation
            abs_points = [(int(x), int(y)) for x, y in points]

            # Create binary mask
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            contour = np.array(abs_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [contour], 1)

            # Encode to RLE
            rle = coco_mask.encode(np.asfortranarray(mask))

            # Convert RLE to JSON-serializable format
            if isinstance(rle, dict):
                rle_dict = dict(rle)
                if "counts" in rle_dict and isinstance(rle_dict["counts"], bytes):
                    # Use latin1 encoding for lossless bytes-to-string conversion
                    rle_dict["counts"] = rle_dict["counts"].decode("latin1")
                return rle_dict
            else:
                self._log_warning(f"Unexpected RLE type: {type(rle)}")
                return rle

        except Exception as e:
            self._log_error(f"Error encoding polygon to RLE: {e}")
            raise

    def _parse_polygon_segmentation(
        self, seg_data: List
    ) -> List[Tuple[float, float]]:
        """Parse polygon segmentation data to absolute pixel point list."""
        points = []

        for polygon in seg_data:
            if len(polygon) % 2 != 0:
                self._log_warning(
                    f"Odd number of coordinates in polygon: {len(polygon)}"
                )
                continue
            if len(polygon) < 6:
                self._log_warning(
                    f"Polygon has fewer than 3 vertices ({len(polygon)} values), skipping"
                )
                continue

            # Convert to (x, y) pairs — coordinates are already absolute pixels
            for i in range(0, len(polygon), 2):
                x = polygon[i]
                y = polygon[i + 1]
                points.append((x, y))

        return points

    def write(
        self,
        annotations: DatasetAnnotations,
        output_file: str,
        output_rle: Optional[bool] = None,
    ) -> AnnotationResult:
        """
        Write annotations to COCO JSON format.

        Args:
            annotations: DatasetAnnotations to write
            output_file: Path to output JSON file
            output_rle: Whether to output RLE format. If None, uses self.output_rle

        Returns:
            AnnotationResult with success status

        Notes:
            - Supports both polygon and RLE output formats
            - Expects coordinates in native COCO format (absolute pixels, top-left)
            - Handles crowd annotations (iscrowd flag)
        """
        result = AnnotationResult(success=False)
        output_path = Path(output_file)

        try:
            original_output_rle = self.output_rle
            if output_rle is not None:
                self.output_rle = output_rle

            # Prepare COCO data structure
            coco_data = self._prepare_coco_data(annotations)

            # Write JSON file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(coco_data, f, indent=2, ensure_ascii=False)

            result.success = True
            result.message = f"Successfully wrote COCO annotations to {output_path}"
            result.data = {"output_file": str(output_path)}

            self.output_rle = original_output_rle

        except Exception as e:
            result.add_error(f"Error writing COCO annotations: {e}")
            if "original_output_rle" in locals():
                self.output_rle = original_output_rle

        return result

    def write_one(
        self, image_ann: ImageAnnotation, output_dir: Path
    ) -> AnnotationResult:
        """COCO does not support per-image write.

        COCO is always written as a single JSON file via ``write()``.
        Use ``write()`` for COCO output.
        """
        raise NotImplementedError(
            "COCO format does not support per-image write. "
            "Use write() to output a single JSON file."
        )

    def _prepare_coco_data(self, annotations: DatasetAnnotations) -> Dict[str, Any]:
        """Prepare COCO JSON data structure from DatasetAnnotations."""
        # Info section from dataset_info or defaults
        info = annotations.dataset_info.get("info", {})
        if not info:
            info = {
                "description": "COCO dataset",
                "url": "",
                "version": "1.0",
                "year": 2026,
                "contributor": "",
                "date_created": "2026-03-22",
            }

        # Prepare categories from annotations.categories
        categories = []
        # Check if dataset_info has original category data with extra fields
        original_cats = annotations.dataset_info.get("categories")
        if original_cats:
            categories = original_cats.copy()
            self._log_debug(f"Using dataset_info categories: {len(categories)} categories")
        else:
            for cat_id, cat_name in annotations.categories.items():
                categories.append(
                    {"id": cat_id, "name": cat_name, "supercategory": "none"}
                )

        # Prepare images and annotations
        images = []
        coco_annotations = []
        ann_id = 1
        img_id_counter = 1

        for img in annotations.images:
            images.append(
                {
                    "id": int(img.image_id) if img.image_id.isdigit() else img_id_counter,
                    "width": img.width,
                    "height": img.height,
                    "file_name": img.image_path,
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": "",
                }
            )

            # Add object annotations
            for obj in img.objects:
                coco_ann = self._object_to_coco_annotation(obj, img, ann_id, img_id_counter)
                if coco_ann:
                    coco_annotations.append(coco_ann)
                    ann_id += 1
                elif self.strict_mode:
                    self._log_error(
                        f"Failed to convert object {obj.class_name} "
                        f"(class_id={obj.class_id}) to COCO format"
                    )
                else:
                    self._log_warning(
                        f"Skipping object {obj.class_name}: conversion to COCO format failed"
                    )

            img_id_counter += 1

        # Prediction mode: output plain list of annotation dicts
        if self.prediction:
            return coco_annotations

        # Annotation mode: output full COCO dict
        result = {
            "info": info,
            "images": images,
            "annotations": coco_annotations,
            "categories": categories,
        }

        # Include any additional dataset_info fields (licenses, etc.)
        for k, v in annotations.dataset_info.items():
            if k not in ["info", "images", "annotations", "categories"]:
                result[k] = v

        return result

    def _object_to_coco_annotation(
        self, obj: ObjectAnnotation, img: ImageAnnotation, ann_id: int, img_id: int
    ) -> Optional[Dict]:
        """Convert ObjectAnnotation to COCO annotation dict."""
        try:
            # Determine segmentation format
            segmentation = None
            iscrowd = 1 if obj.is_crowd else 0

            if obj.segmentation:
                seg = obj.segmentation
                use_rle = False
                if obj.is_crowd:
                    use_rle = True
                elif self.output_rle:
                    use_rle = True

                has_rle = seg.has_rle()
                self._log_debug(
                    f"RLE conversion: use_rle={use_rle}, has_rle={has_rle}, "
                    f"HAS_COCO_MASK={HAS_COCO_MASK}, is_crowd={obj.is_crowd}"
                )

                if use_rle and has_rle:
                    # Use preserved RLE data directly
                    segmentation = seg.rle
                    self._log_debug("Using preserved RLE data")
                elif use_rle and HAS_COCO_MASK:
                    try:
                        rle = self._encode_polygon_to_rle(
                            seg.points, img.width, img.height
                        )
                        segmentation = rle
                        self._log_debug("Successfully encoded polygon to RLE")
                    except ImportError:
                        self._log_warning(
                            "pycocotools not available, falling back to polygon format"
                        )
                        use_rle = False
                    except Exception as e:
                        self._log_warning(
                            f"Failed to encode RLE: {e}, falling back to polygon format"
                        )
                        use_rle = False

                if not use_rle:
                    # Convert to COCO polygon format
                    points = seg.points
                    if not points and has_rle and HAS_COCO_MASK:
                        try:
                            points = self._decode_rle_to_polygon(
                                seg.rle, img.width, img.height
                            )
                        except Exception as e:
                            self._log_warning(
                                f"Failed to decode RLE to polygon: {e}, skipping segmentation"
                            )
                            points = []
                    if points:
                        # Points are already in absolute pixels (new architecture)
                        polygon = []
                        for x, y in points:
                            polygon.extend([float(x), float(y)])
                        segmentation = [polygon]
                        iscrowd = 0
                    else:
                        if has_rle:
                            segmentation = seg.rle
                        else:
                            segmentation = []
                            iscrowd = 0

            elif obj.bbox:
                segmentation = []
                iscrowd = 0
            else:
                self._log_warning(
                    f"Object {obj.class_name} has neither bbox nor segmentation"
                )
                return None

            # Convert bbox to COCO format: [x, y, width, height] in absolute pixels
            bbox = []
            area = 0.0
            if obj.bbox:
                # Bbox is already in native COCO format (absolute pixels, top-left)
                bbox = [float(obj.bbox.x), float(obj.bbox.y),
                        float(obj.bbox.width), float(obj.bbox.height)]
                area = float(obj.bbox.width * obj.bbox.height)
            elif (
                obj.segmentation
                and segmentation
                and isinstance(segmentation, list)
                and segmentation
            ):
                # Estimate area from segmentation polygon
                points = obj.segmentation.points
                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)
                    area = float(w * h)

            image_id_val = int(img.image_id) if img.image_id.isdigit() else img_id

            ann_dict = {
                "image_id": image_id_val,
                "category_id": obj.class_id,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": iscrowd,
            }
            if not self.prediction:
                ann_dict["id"] = ann_id

            # Include score for prediction output
            # In prediction mode, always include score (explicit contract).
            # In annotation mode, only include when confidence < 1.0
            # (data-driven, for round-trip compatibility).
            if self.prediction or obj.confidence < 1.0:
                ann_dict["score"] = obj.confidence

            return ann_dict

        except Exception as e:
            self._log_error(f"Error converting object to COCO format: {e}")
            return None

    def validate(self) -> bool:
        """Validate COCO JSON file."""
        try:
            with open(self.annotation_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            required_fields = ["images", "annotations", "categories"]
            for field in required_fields:
                if field not in data:
                    self.logger.error(f"Missing required field '{field}'")
                    return False

            for img in data["images"]:
                if "id" not in img or "file_name" not in img:
                    self.logger.error(f"Image missing required fields: {img}")
                    return False

            for cat in data["categories"]:
                if "id" not in cat or "name" not in cat:
                    self.logger.error(f"Category missing required fields: {cat}")
                    return False

            valid_image_ids = {img["id"] for img in data["images"]}
            valid_category_ids = {cat["id"] for cat in data["categories"]}

            for ann in data["annotations"]:
                if "id" not in ann or "image_id" not in ann or "category_id" not in ann:
                    self.logger.error(f"Annotation missing required fields: {ann}")
                    return False

                if ann["image_id"] not in valid_image_ids:
                    self.logger.error(
                        f"Annotation {ann.get('id')} references non-existent image_id: {ann['image_id']}"
                    )
                    return False

                if ann["category_id"] not in valid_category_ids:
                    self.logger.error(
                        f"Annotation {ann.get('id')} references non-existent category_id: {ann['category_id']}"
                    )
                    return False

                if "segmentation" in ann:
                    seg = ann["segmentation"]
                    if isinstance(seg, dict):
                        if "size" not in seg or "counts" not in seg:
                            self.logger.error(
                                f"RLE segmentation missing 'size' or 'counts' in annotation {ann.get('id')}"
                            )
                            return False
                        size = seg["size"]
                        if not isinstance(size, list) or len(size) != 2:
                            self.logger.error(
                                f"RLE 'size' must be [height, width], got {size} in annotation {ann.get('id')}"
                            )
                            return False
                        if not isinstance(size[0], int) or not isinstance(size[1], int) or size[0] <= 0 or size[1] <= 0:
                            self.logger.error(
                                f"RLE 'size' values must be positive integers in annotation {ann.get('id')}"
                            )
                            return False
                        if not isinstance(seg["counts"], str) or not seg["counts"]:
                            self.logger.error(
                                f"RLE 'counts' must be a non-empty string in annotation {ann.get('id')}"
                            )
                            return False
                    elif isinstance(seg, list):
                        for polygon in seg:
                            if len(polygon) < 6:
                                self.logger.error(
                                    f"Polygon segmentation has fewer than 3 vertices ({len(polygon)} values) in annotation {ann.get('id')}"
                                )
                                return False

            return True

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error validating COCO file: {e}")
            return False
