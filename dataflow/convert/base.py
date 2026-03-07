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