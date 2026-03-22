#!/usr/bin/env python3
"""
COCO annotation format handler.

Handles reading and writing of COCO format annotation files.
Supports both polygon and RLE segmentation formats.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import sys

try:
    from pycocotools import mask as coco_mask
    HAS_COCO_MASK = True
except ImportError:
    HAS_COCO_MASK = False

import cv2
import numpy as np

from .base import BaseAnnotationHandler, AnnotationResult
from .models import (
    DatasetAnnotations, ImageAnnotation, ObjectAnnotation,
    BoundingBox, Segmentation, OriginalData, AnnotationFormat
)
from dataflow.util.file_util import FileOperations


class CocoAnnotationHandler(BaseAnnotationHandler):
    """Handler for COCO annotation format."""

    def __init__(self, annotation_file: str, **kwargs):
        """
        Initialize COCO handler.

        Args:
            annotation_file: Path to COCO JSON annotation file
            **kwargs: Additional arguments for BaseAnnotationHandler
        """
        super().__init__(**kwargs)
        self.annotation_file = Path(annotation_file)
        self.file_ops = FileOperations(logger=self.logger)
        self.categories = {}
        self.original_categories = []  # Full category data for lossless preservation
        self.original_images = []      # Full image data for lossless preservation
        self.images = {}
        self.annotations = []
        self.dataset_info = {}
        self.output_rle = False  # Whether to output RLE format when writing

    def read(self) -> AnnotationResult:
        """
        Read COCO JSON annotation file.

        Returns:
            AnnotationResult: Result containing parsed annotations if successful

        Notes:
            - Supports both polygon and RLE segmentation formats
            - Automatically detects format (polygon vs RLE)
            - Normalizes coordinates to [0, 1] range
            - Handles optional pycocotools dependency gracefully
            - Validates COCO format structure in strict mode
        """
        result = AnnotationResult(success=False)

        if not self.annotation_file.exists():
            result.add_error(f"Annotation file does not exist: {self.annotation_file}")
            return result

        try:
            with open(self.annotation_file, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)

            # Validate required top-level fields
            required_fields = ['images', 'annotations', 'categories']
            for field in required_fields:
                if field not in coco_data:
                    result.add_error(f"Missing required field '{field}' in {self.annotation_file}")
                    return result

            # Load dataset info (optional)
            self.dataset_info = {k: v for k, v in coco_data.items()
                               if k not in ['images', 'annotations', 'categories']}

            # Load categories
            self.categories = self._load_categories(coco_data['categories'])

            # Load images
            self.images = self._load_images(coco_data['images'])
            self.original_images = coco_data['images'].copy()  # Store for lossless preservation

            # Load annotations
            self.annotations = coco_data['annotations']

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

    def _load_categories(self, coco_categories: List[Dict]) -> Dict[int, str]:
        """Load category mapping from COCO categories list."""
        categories = {}
        # Store full category data for lossless preservation
        self.original_categories = coco_categories.copy()
        for cat in coco_categories:
            cat_id = cat.get('id')
            cat_name = cat.get('name', '')
            if cat_id is not None:
                categories[cat_id] = cat_name
        return categories

    def _load_images(self, coco_images: List[Dict]) -> Dict[int, Dict]:
        """Load image information from COCO images list.

        Returns a dict mapping image_id to a minimal image info dict for internal use.
        The full original image data is stored separately for lossless preservation.
        """
        images = {}
        for img in coco_images:
            img_id = img.get('id')
            if img_id is not None:
                images[img_id] = {
                    'file_name': img.get('file_name', ''),
                    'width': img.get('width', 0),
                    'height': img.get('height', 0),
                    'coco_url': img.get('coco_url', ''),
                    'flickr_url': img.get('flickr_url', '')
                }
        return images

    def _detect_rle_format(self, annotations: List[Dict]) -> bool:
        """Detect if annotations contain RLE format segmentation."""
        for ann in annotations:
            if 'segmentation' in ann:
                seg = ann['segmentation']
                if isinstance(seg, dict) and 'counts' in seg:
                    return True
        return False

    def _create_dataset(self) -> DatasetAnnotations:
        """Create DatasetAnnotations from loaded COCO data."""
        dataset = DatasetAnnotations()
        dataset.categories = self.categories.copy()
        # Save entire COCO data structure in dataset_info for lossless preservation
        dataset.dataset_info = self.dataset_info.copy()
        dataset.dataset_info["__coco_original_data__"] = {
            "images": self.original_images.copy(),  # Store full original image data
            "annotations": self.annotations.copy(),
            "categories": self.original_categories.copy()
        }

        # Create mapping from image_id to full original image data
        original_image_map = {}
        for img_data in self.original_images:
            img_id = img_data.get('id')
            if img_id is not None:
                original_image_map[img_id] = img_data.copy()

        # Create image annotations
        for img_id, img_info in self.images.items():
            # Find annotations for this image
            img_anns = [ann for ann in self.annotations if ann.get('image_id') == img_id]

            objects = self._create_objects(img_anns, img_info['width'], img_info['height'])

            # Get full original image data for lossless preservation
            original_img_data = original_image_map.get(img_id, img_info.copy())

            # Create original data for the image (partial COCO data for this image)
            image_original_data = OriginalData(
                format=AnnotationFormat.COCO.value,
                raw_data={
                    "image_info": original_img_data,  # Store full original image data
                    "image_annotations": img_anns.copy()
                },
                metadata={
                    "image_id": img_id,
                    "total_annotations": len(img_anns)
                }
            )

            image_ann = ImageAnnotation(
                image_id=str(img_id),
                image_path=img_info['file_name'],
                width=img_info['width'],
                height=img_info['height'],
                objects=objects,
                original_data=image_original_data
            )
            dataset.add_image(image_ann)

        return dataset

    def _create_objects(self, img_anns: List[Dict], img_width: int, img_height: int) -> List[ObjectAnnotation]:
        """Create ObjectAnnotations from COCO annotations for a single image."""
        objects = []

        for ann in img_anns:
            try:
                class_id = ann.get('category_id')
                if class_id is None or class_id not in self.categories:
                    self._log_warning(f"Skipping annotation with invalid category_id: {ann.get('category_id')}")
                    continue

                class_name = self.categories[class_id]
                is_crowd = ann.get('iscrowd', 0) == 1
                bbox = None
                segmentation = None

                # Create original data for the entire COCO annotation
                original_data = OriginalData(
                    format=AnnotationFormat.COCO.value,
                    raw_data=ann.copy(),
                    metadata={
                        "image_width": img_width,
                        "image_height": img_height
                    }
                )

                # Parse bbox if present
                if 'bbox' in ann and ann['bbox']:
                    bbox_data = ann['bbox']
                    if len(bbox_data) == 4:
                        # COCO bbox: [x, y, width, height] in absolute pixels
                        x_abs, y_abs, w_abs, h_abs = bbox_data
                        # Convert to normalized center coordinates
                        x_center = (x_abs + w_abs / 2) / img_width
                        y_center = (y_abs + h_abs / 2) / img_height
                        width_norm = w_abs / img_width
                        height_norm = h_abs / img_height

                        bbox = BoundingBox(
                            x=x_center,
                            y=y_center,
                            width=width_norm,
                            height=height_norm,
                            original_data=original_data
                        )
                        if not self._validate_bbox(bbox):
                            self._log_warning(f"Invalid bbox in annotation {ann.get('id')}")
                            bbox = None

                # Parse segmentation
                if 'segmentation' in ann and ann['segmentation']:
                    seg_data = ann['segmentation']
                    if isinstance(seg_data, dict) and 'counts' in seg_data:
                        # RLE format - preserve original RLE data
                        rle_dict = seg_data  # Keep original RLE dict
                        points = []
                        if HAS_COCO_MASK:
                            try:
                                points = self._decode_rle_to_polygon(seg_data, img_width, img_height)
                            except Exception as e:
                                self._log_warning(f"Failed to decode RLE for annotation {ann.get('id')}: {e}")
                                # Continue with empty points, but preserve RLE
                        else:
                            self._log_warning(f"pycocotools not available, preserving RLE without decoding for annotation {ann.get('id')}")
                        segmentation = Segmentation(points=points, rle=rle_dict, original_data=original_data)
                    elif isinstance(seg_data, list) and len(seg_data) > 0:
                        # Polygon format (list of lists)
                        points = self._parse_polygon_segmentation(seg_data, img_width, img_height)
                        if points:
                            segmentation = Segmentation(points=points, rle=None, original_data=original_data)
                        else:
                            self._log_warning(f"Invalid polygon segmentation in annotation {ann.get('id')}")

                # Create object annotation
                obj = ObjectAnnotation(
                    class_id=class_id,
                    class_name=class_name,
                    bbox=bbox,
                    segmentation=segmentation,
                    confidence=1.0,
                    is_crowd=is_crowd,
                    original_data=original_data
                )
                objects.append(obj)

            except Exception as e:
                self._log_warning(f"Error processing annotation {ann.get('id')}: {e}")
                continue

        return objects

    def _decode_rle_to_polygon(self, rle: Dict, img_width: int, img_height: int) -> List[Tuple[float, float]]:
        """
        Decode RLE to polygon point list.

        Args:
            rle: RLE dict with 'size' and 'counts' fields
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            List[Tuple[float, float]]: Normalized polygon points in range [0, 1]

        Raises:
            ImportError: If pycocotools is not available
        """
        if not HAS_COCO_MASK:
            raise ImportError("pycocotools required for RLE decoding")

        try:
            # Make a copy of RLE dict to avoid modifying original
            rle_dict = dict(rle)

            # Ensure 'counts' is bytes for coco_mask.decode
            if 'counts' in rle_dict and isinstance(rle_dict['counts'], str):
                rle_dict['counts'] = rle_dict['counts'].encode('utf-8')

            # Decode RLE to binary mask
            binary_mask = coco_mask.decode(rle_dict)

            # Extract contours from mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return []

            # Use the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Convert contour points to normalized coordinates
            points = []
            for point in largest_contour:
                x, y = point[0]
                x_norm = x / img_width
                y_norm = y / img_height
                points.append((x_norm, y_norm))

            return points

        except Exception as e:
            self._log_error(f"Error decoding RLE: {e}")
            return []

    def _encode_polygon_to_rle(self, points: List[Tuple[float, float]],
                              img_width: int, img_height: int) -> Dict:
        """
        Encode polygon points to RLE format.

        Args:
            points: List of normalized (x, y) polygon points in range [0, 1]
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
            # Convert normalized coordinates to absolute coordinates
            abs_points = [(int(x * img_width), int(y * img_height)) for x, y in points]

            # Create binary mask
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            contour = np.array(abs_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [contour], 1)

            # Encode to RLE
            rle = coco_mask.encode(np.asfortranarray(mask))

            # Convert RLE to JSON-serializable format
            # coco_mask.encode returns dict with 'size' and 'counts'
            # 'counts' is bytes, need to convert to string for JSON serialization
            if isinstance(rle, dict):
                # Make a copy to avoid modifying original
                rle_dict = dict(rle)
                if 'counts' in rle_dict and isinstance(rle_dict['counts'], bytes):
                    # Convert bytes to string (UTF-8 encoding should work for RLE)
                    rle_dict['counts'] = rle_dict['counts'].decode('utf-8')
                return rle_dict
            else:
                # If not a dict, return as-is (shouldn't happen with pycocotools)
                self._log_warning(f"Unexpected RLE type: {type(rle)}")
                return rle

        except Exception as e:
            self._log_error(f"Error encoding polygon to RLE: {e}")
            raise

    def _parse_polygon_segmentation(self, seg_data: List, img_width: int, img_height: int) -> List[Tuple[float, float]]:
        """Parse polygon segmentation data to normalized point list."""
        points = []

        # COCO polygon format: [[x1, y1, x2, y2, ...], ...] for multiple polygons
        # We'll combine all polygons into one (assuming single object)
        for polygon in seg_data:
            if len(polygon) % 2 != 0:
                self._log_warning(f"Odd number of coordinates in polygon: {len(polygon)}")
                continue

            # Convert to (x, y) pairs and normalize
            for i in range(0, len(polygon), 2):
                x = polygon[i] / img_width
                y = polygon[i + 1] / img_height
                points.append((x, y))

        return points

    def write(self, annotations: DatasetAnnotations, output_file: str,
              output_rle: Optional[bool] = None) -> AnnotationResult:
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
            - Converts normalized coordinates to absolute pixels
            - Handles crowd annotations (iscrowd flag)
            - Converts segmentation polygons to RLE when output_rle=True
            - Maintains COCO JSON structure with info, images, annotations, categories
        """
        result = AnnotationResult(success=False)
        output_path = Path(output_file)

        try:
            # Save original output_rle setting
            original_output_rle = self.output_rle
            if output_rle is not None:
                self.output_rle = output_rle

            # Prepare COCO data structure
            coco_data = self._prepare_coco_data(annotations)

            # Write JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, indent=2, ensure_ascii=False)

            result.success = True
            result.message = f"Successfully wrote COCO annotations to {output_path}"
            result.data = {"output_file": str(output_path)}

            # Restore original setting
            self.output_rle = original_output_rle

        except Exception as e:
            result.add_error(f"Error writing COCO annotations: {e}")
            # Ensure original setting is restored even on error
            if 'original_output_rle' in locals():
                self.output_rle = original_output_rle

        return result

    def _prepare_coco_data(self, annotations: DatasetAnnotations) -> Dict[str, Any]:
        """Prepare COCO JSON data structure from DatasetAnnotations."""
        # Prepare info section - use original data if available for lossless preservation
        info = {
            "description": "COCO dataset",
            "url": "",
            "version": "1.0",
            "year": 2026,
            "contributor": "",
            "date_created": "2026-03-22"
        }
        # Check if we have original COCO data
        original_coco_data = annotations.dataset_info.get("__coco_original_data__")
        if original_coco_data and "info" in original_coco_data:
            # Use original info with all fields preserved
            info = original_coco_data["info"].copy()
            self._log_debug("Using original COCO info data")

        # Prepare categories - use original data if available for lossless preservation
        categories = []
        if original_coco_data and "categories" in original_coco_data:
            # Use original categories with all fields preserved
            categories = original_coco_data["categories"].copy()
            self._log_debug(f"Using original COCO categories data: {len(categories)} categories")
        else:
            # Fallback to building categories from annotations.categories
            for cat_id, cat_name in annotations.categories.items():
                categories.append({
                    "id": cat_id,
                    "name": cat_name,
                    "supercategory": "none"
                })

        # Prepare images and annotations
        images = []
        coco_annotations = []
        ann_id = 1

        for img in annotations.images:
            # Priority: Use original data if available and format matches
            if img.has_original_data() and img.original_data.format == AnnotationFormat.COCO.value:
                original_data = img.original_data.raw_data.get("image_info", {})
                image_info = original_data.copy()
                # Update fields that may have changed
                image_info["id"] = int(img.image_id) if img.image_id.isdigit() else ann_id
                image_info["width"] = img.width
                image_info["height"] = img.height
                image_info["file_name"] = img.image_path
                # Ensure required fields exist
                for field in ["license", "flickr_url", "coco_url", "date_captured"]:
                    if field not in image_info:
                        image_info[field] = "" if field.endswith("_url") or field == "date_captured" else 1
                images.append(image_info)
                self._log_debug(f"Using original COCO image data for image {img.image_id}")
            else:
                # Add image info
                images.append({
                    "id": int(img.image_id) if img.image_id.isdigit() else ann_id,
                    "width": img.width,
                    "height": img.height,
                    "file_name": img.image_path,
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": ""
                })

            # Add object annotations
            for obj in img.objects:
                coco_ann = self._object_to_coco_annotation(obj, img, ann_id)
                if coco_ann:
                    coco_annotations.append(coco_ann)
                    ann_id += 1

        # Prepare dataset info - preserve original dataset info for lossless preservation
        dataset_info = annotations.dataset_info.copy() if annotations.dataset_info else {}

        # Remove the internal __coco_original_data__ field as it's for internal use only
        dataset_info.pop("__coco_original_data__", None)

        # Ensure required info fields exist, but preserve original values if available
        if "info" not in dataset_info:
            dataset_info["info"] = {
                "description": "Generated by DataFlow-CV",
                "url": "",
                "version": "1.0",
                "year": 2026,
                "contributor": "",
                "date_created": "2026-03-21"
            }
        else:
            # Ensure info dict has all required fields, but preserve original values
            info = dataset_info["info"]
            info.setdefault("description", "Generated by DataFlow-CV")
            info.setdefault("url", "")
            info.setdefault("version", "1.0")
            info.setdefault("year", 2026)
            info.setdefault("contributor", "")
            info.setdefault("date_created", "2026-03-21")

        return {
            **dataset_info,
            "images": images,
            "annotations": coco_annotations,
            "categories": categories
        }

    def _object_to_coco_annotation(self, obj: ObjectAnnotation, img: ImageAnnotation, ann_id: int) -> Optional[Dict]:
        """Convert ObjectAnnotation to COCO annotation dict."""
        try:
            # Priority 1: Use original data if available and format matches
            use_original_data = False
            original_data_copy = None

            if obj.has_original_data() and obj.original_data.format == AnnotationFormat.COCO.value:
                original_data_copy = obj.original_data.raw_data.copy()

                # Check if we need to convert RLE to polygon based on output_rle and is_crowd
                seg_data = original_data_copy.get("segmentation")
                is_rle_format = isinstance(seg_data, dict) and "counts" in seg_data

                # Determine if we should use original data or convert
                # Rule: Crowd annotations always use RLE, non-crowd depends on output_rle
                if is_rle_format:
                    if obj.is_crowd:
                        # Crowd annotations should remain RLE regardless of output_rle
                        use_original_data = True
                    else:
                        # Non-crowd annotations: use RLE if output_rle=True, convert if False
                        use_original_data = self.output_rle
                else:
                    # Non-RLE format (polygon or empty)
                    # If output_rle=True, we may want to convert polygon to RLE
                    # So skip original data to allow RLE encoding in priority 2
                    use_original_data = not self.output_rle

            if use_original_data and original_data_copy:
                original_data = original_data_copy

                # Update fields that may have changed
                # ID should be new to avoid conflicts
                original_data["id"] = ann_id
                # Update image_id to match current image
                original_data["image_id"] = int(img.image_id) if img.image_id.isdigit() else ann_id
                # Update category_id to current class_id (class mapping may have changed)
                original_data["category_id"] = obj.class_id
                # Update iscrowd flag
                original_data["iscrowd"] = 1 if obj.is_crowd else 0

                # Ensure bbox and segmentation fields exist
                if "bbox" not in original_data:
                    original_data["bbox"] = []
                if "segmentation" not in original_data:
                    original_data["segmentation"] = []
                if "area" not in original_data:
                    original_data["area"] = 0.0

                self._log_debug(f"Using original COCO data for object {obj.class_name}")
                return original_data

            # Priority 2: Use preserved RLE data for segmentation
            # Determine segmentation format
            segmentation = None
            iscrowd = 1 if obj.is_crowd else 0

            if obj.segmentation:
                seg = obj.segmentation
                # Determine whether to output RLE format
                use_rle = False
                if obj.is_crowd:
                    # Crowd annotations should use RLE format
                    use_rle = True
                elif self.output_rle:
                    # Use RLE if output_rle flag is set
                    use_rle = True

                has_rle = seg.has_rle()
                self._log_debug(f"RLE conversion: use_rle={use_rle}, has_rle={has_rle}, HAS_COCO_MASK={HAS_COCO_MASK}, is_crowd={obj.is_crowd}")
                if use_rle and has_rle:
                    # Use preserved RLE data directly
                    segmentation = seg.rle
                    self._log_debug(f"Using preserved RLE data")
                elif use_rle and HAS_COCO_MASK:
                    try:
                        # Encode polygon to RLE
                        self._log_debug(f"Encoding polygon to RLE, points count: {len(seg.points) if seg.points else 0}")
                        rle = self._encode_polygon_to_rle(
                            seg.points, img.width, img.height
                        )
                        segmentation = rle
                        self._log_debug(f"Successfully encoded to RLE")
                    except ImportError:
                        self._log_warning("pycocotools not available, falling back to polygon format")
                        use_rle = False
                    except Exception as e:
                        self._log_warning(f"Failed to encode RLE: {e}, falling back to polygon format")
                        use_rle = False

                if not use_rle:
                    # Convert to COCO polygon format
                    points = seg.points
                    # If points empty but RLE exists, attempt to decode RLE to polygon
                    if not points and has_rle and HAS_COCO_MASK:
                        try:
                            points = self._decode_rle_to_polygon(seg.rle, img.width, img.height)
                        except Exception as e:
                            self._log_warning(f"Failed to decode RLE to polygon: {e}, skipping segmentation")
                            points = []
                    if points:
                        points_abs = [(int(x * img.width), int(y * img.height)) for x, y in points]
                        # Flatten points
                        polygon = []
                        for x, y in points_abs:
                            polygon.extend([float(x), float(y)])
                        segmentation = [polygon]
                        iscrowd = 0
                    else:
                        # No polygon points available, fall back to RLE if exists
                        if has_rle:
                            segmentation = seg.rle
                        else:
                            # No segmentation data
                            segmentation = []
                            iscrowd = 0

            elif obj.bbox:
                # For bbox-only annotations, no segmentation
                # COCO requires segmentation for instance segmentation, but we can leave empty
                segmentation = []
                iscrowd = 0
            else:
                self._log_warning(f"Object {obj.class_name} has neither bbox nor segmentation")
                return None

            # Convert bbox to COCO format: [x, y, width, height] in absolute pixels
            bbox = []
            area = 0.0
            if obj.bbox:
                x_abs, y_abs, w_abs, h_abs = obj.bbox.xywh_abs(img.width, img.height)
                bbox = [float(x_abs), float(y_abs), float(w_abs), float(h_abs)]
                area = float(w_abs * h_abs)
            elif obj.segmentation and segmentation and isinstance(segmentation, list) and segmentation:
                # Estimate area from segmentation polygon (approximate)
                # For simplicity, use bounding box of polygon points
                points_abs = obj.segmentation.points_abs(img.width, img.height)
                if points_abs:
                    xs = [p[0] for p in points_abs]
                    ys = [p[1] for p in points_abs]
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)
                    area = float(w * h)

            return {
                "id": ann_id,
                "image_id": int(img.image_id) if img.image_id.isdigit() else ann_id,
                "category_id": obj.class_id,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": iscrowd
            }

        except Exception as e:
            self._log_error(f"Error converting object to COCO format: {e}")
            return None

    def validate(self) -> bool:
        """Validate COCO JSON file."""
        try:
            with open(self.annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check required fields
            required_fields = ['images', 'annotations', 'categories']
            for field in required_fields:
                if field not in data:
                    self.logger.error(f"Missing required field '{field}'")
                    return False

            # Validate images
            for img in data['images']:
                if 'id' not in img or 'file_name' not in img:
                    self.logger.error(f"Image missing required fields: {img}")
                    return False

            # Validate categories
            for cat in data['categories']:
                if 'id' not in cat or 'name' not in cat:
                    self.logger.error(f"Category missing required fields: {cat}")
                    return False

            # Validate annotations
            for ann in data['annotations']:
                if 'id' not in ann or 'image_id' not in ann or 'category_id' not in ann:
                    self.logger.error(f"Annotation missing required fields: {ann}")
                    return False

            return True

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error validating COCO file: {e}")
            return False