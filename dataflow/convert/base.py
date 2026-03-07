"""
Base converter class for DataFlow.
"""

import json
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import numpy as np


class BaseConverter:
    """Base class for all format converters."""

    @staticmethod
    def validate_paths(*paths: str) -> None:
        """
        Validate that all paths are valid.

        Args:
            *paths: Paths to validate

        Raises:
            FileNotFoundError: If any path doesn't exist
            ValueError: If any path is invalid
        """
        for path in paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Path does not exist: {path}")

    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str) -> None:
        """Save data to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_txt(file_path: str) -> List[str]:
        """Load text file as list of lines."""
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def save_txt(data: List[str], file_path: str) -> None:
        """Save data to text file."""
        with open(file_path, 'w') as f:
            for line in data:
                f.write(line + '\n')

    @staticmethod
    def bbox_x1y1wh_to_xcycwh(bbox: List[float]) -> List[float]:
        """
        Convert bounding box from (x1, y1, w, h) to (xc, yc, w, h).

        Args:
            bbox: [x1, y1, width, height]

        Returns:
            [xc, yc, width, height]
        """
        x1, y1, w, h = bbox
        xc = x1 + w / 2
        yc = y1 + h / 2
        return [xc, yc, w, h]

    @staticmethod
    def bbox_xcycwh_to_x1y1wh(bbox: List[float]) -> List[float]:
        """
        Convert bounding box from (xc, yc, w, h) to (x1, y1, w, h).

        Args:
            bbox: [xc, yc, width, height]

        Returns:
            [x1, y1, width, height]
        """
        xc, yc, w, h = bbox
        x1 = xc - w / 2
        y1 = yc - h / 2
        return [x1, y1, w, h]

    @staticmethod
    def normalize_bbox(bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        Normalize bounding box coordinates to [0, 1].

        Args:
            bbox: [x, y, width, height] in absolute coordinates
            img_width: Image width
            img_height: Image height

        Returns:
            Normalized [x, y, width, height]
        """
        return [
            bbox[0] / img_width,
            bbox[1] / img_height,
            bbox[2] / img_width,
            bbox[3] / img_height
        ]

    @staticmethod
    def denormalize_bbox(bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        Denormalize bounding box coordinates from [0, 1] to absolute.

        Args:
            bbox: [x, y, width, height] in normalized coordinates
            img_width: Image width
            img_height: Image height

        Returns:
            Denormalized [x, y, width, height]
        """
        return [
            bbox[0] * img_width,
            bbox[1] * img_height,
            bbox[2] * img_width,
            bbox[3] * img_height
        ]

    @staticmethod
    def get_image_size(image_path: str) -> Tuple[int, int]:
        """
        Get image dimensions.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (width, height)
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except ImportError:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")
            height, width = img.shape[:2]
            return width, height

    @staticmethod
    def get_image_size_from_source(source_path: str) -> Tuple[int, int]:
        """
        Get image dimensions from various sources (image file, LabelMe JSON, etc.).

        Args:
            source_path: Path to image file or annotation file with image dimensions

        Returns:
            Tuple of (width, height)
        """
        from pathlib import Path
        source_path_obj = Path(source_path)

        # Check if it's an image file
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        if source_path_obj.suffix.lower() in image_extensions:
            return BaseConverter.get_image_size(source_path)

        # Check if it's a LabelMe JSON file
        if source_path_obj.suffix.lower() == '.json':
            try:
                data = BaseConverter.load_json(source_path)
                if 'imageWidth' in data and 'imageHeight' in data:
                    return data['imageWidth'], data['imageHeight']
            except:
                pass

        # Check if it's a COCO JSON file
        if source_path_obj.suffix.lower() == '.json':
            try:
                data = BaseConverter.load_json(source_path)
                if 'images' in data and len(data['images']) > 0:
                    # Return dimensions of first image
                    img = data['images'][0]
                    return img.get('width', 0), img.get('height', 0)
            except:
                pass

        raise ValueError(f"Cannot determine image dimensions from source: {source_path}")

    @staticmethod
    def find_matching_pairs_for_conversion(
        input_dir: str,
        annotation_dir: str,
        annotation_ext: str,
        input_ext: str = None
    ) -> List[Tuple[str, str]]:
        """
        Find matching input-annotation file pairs between directories.
        For conversions that don't need images, input_dir can be None or empty.

        Args:
            input_dir: Directory containing input files (images or annotations)
            annotation_dir: Directory containing annotation files
            annotation_ext: Expected annotation file extension (e.g., '.json', '.txt')
            input_ext: Expected input file extension (e.g., '.jpg', '.png').
                      If None, uses image extensions from config.

        Returns:
            List of (input_path, annotation_path) tuples
        """
        from pathlib import Path
        from ..config import get_config

        config = get_config()
        if input_ext is None:
            # Use image extensions by default
            input_extensions = config["conversion"]["image_extensions"]
        else:
            input_extensions = [input_ext]

        annotation_dir_path = Path(annotation_dir)

        # If no input_dir provided (for labelme2coco, labelme2yolo), just return annotation files
        if not input_dir or input_dir == annotation_dir:
            # Return single annotation files (no pairing)
            pairs = []
            for ann_path in annotation_dir_path.glob(f"*{annotation_ext}"):
                if ann_path.is_file():
                    pairs.append((str(ann_path), str(ann_path)))  # Same file for both
            # Also check uppercase extension
            for ann_path in annotation_dir_path.glob(f"*{annotation_ext.upper()}"):
                if ann_path.is_file():
                    path_str = str(ann_path)
                    if (path_str, path_str) not in pairs:
                        pairs.append((path_str, path_str))
            return sorted(pairs)

        # Build mapping of basename to input path
        input_dir_path = Path(input_dir)
        input_map = {}
        for ext in input_extensions:
            for input_path in input_dir_path.glob(f"*{ext}"):
                if input_path.is_file():
                    input_map[input_path.stem] = str(input_path)
            # Also check uppercase extensions
            for input_path in input_dir_path.glob(f"*{ext.upper()}"):
                if input_path.is_file():
                    input_map[input_path.stem] = str(input_path)

        # Find matching annotations
        pairs = []
        for ann_path in annotation_dir_path.glob(f"*{annotation_ext}"):
            if ann_path.is_file():
                basename = ann_path.stem
                if basename in input_map:
                    pairs.append((input_map[basename], str(ann_path)))

        # Also check uppercase extension for annotations
        for ann_path in annotation_dir_path.glob(f"*{annotation_ext.upper()}"):
            if ann_path.is_file():
                basename = ann_path.stem
                if basename in input_map and (input_map[basename], str(ann_path)) not in pairs:
                    pairs.append((input_map[basename], str(ann_path)))

        return sorted(pairs)  # Sort for consistent ordering

    @staticmethod
    def validate_conversion_directories(input_dir: str, annotation_dir: str, needs_input: bool = True) -> None:
        """
        Validate that directories exist and contain files for conversion.

        Args:
            input_dir: Input directory path (images or annotations)
            annotation_dir: Annotation directory path
            needs_input: Whether input directory is required (True for image-based conversions)

        Raises:
            FileNotFoundError: If directories don't exist
            ValueError: If directories are empty or contain no relevant files
        """
        from pathlib import Path
        from ..config import get_config

        config = get_config()
        image_extensions = config["conversion"]["image_extensions"]

        annotation_dir_path = Path(annotation_dir)

        if not annotation_dir_path.exists():
            raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

        # Check if annotation directory contains any files
        if not any(annotation_dir_path.iterdir()):
            raise ValueError(f"Annotation directory is empty: {annotation_dir}")

        if needs_input:
            if not input_dir:
                raise ValueError("Input directory is required for this conversion")

            input_dir_path = Path(input_dir)
            if not input_dir_path.exists():
                raise FileNotFoundError(f"Input directory not found: {input_dir}")

            # Check if input directory contains any files
            if not any(input_dir_path.iterdir()):
                raise ValueError(f"Input directory is empty: {input_dir}")

            # Check for at least one input file with supported extension
            has_inputs = False
            for ext in image_extensions:
                if any(input_dir_path.glob(f"*{ext}")):
                    has_inputs = True
                    break
                if any(input_dir_path.glob(f"*{ext.upper()}")):
                    has_inputs = True
                    break

            if not has_inputs:
                raise ValueError(f"No supported input files found in {input_dir}. Supported extensions: {image_extensions}")