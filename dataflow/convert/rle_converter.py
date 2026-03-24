"""
RLE (Run-Length Encoding) format converter.

Provides tools for converting between polygon points and RLE mask formats,
with optional pycocotools dependency handling.
"""

import logging
import sys
from typing import Dict, List, Optional, Tuple

try:
    from pycocotools import mask as coco_mask

    HAS_COCO_MASK = True
except ImportError:
    HAS_COCO_MASK = False

import numpy as np


class RLEConverter:
    """RLE format conversion utility class."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize RLE converter.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.has_coco_mask = HAS_COCO_MASK
        if not self.has_coco_mask:
            self.logger.warning(
                "pycocotools not installed. RLE conversion will be limited. "
                "Install with: pip install pycocotools"
            )

    def polygon_to_rle(
        self,
        points: List[Tuple[float, float]],
        img_width: int,
        img_height: int,
        require_coco_mask: bool = True,
    ) -> Optional[Dict]:
        """
        Convert polygon points to RLE format.

        Args:
            points: List of normalized (x, y) polygon points in range [0, 1]
            img_width: Image width in pixels
            img_height: Image height in pixels
            require_coco_mask: If True, raise ImportError when pycocotools not available.
                               If False, return None when pycocotools not available.

        Returns:
            RLE dict with 'size' and 'counts' fields (JSON-serializable), or None if failed.

        Raises:
            ImportError: If pycocotools not available and require_coco_mask=True
            ValueError: If points are invalid or image dimensions are invalid
        """
        if not points:
            self.logger.warning("Empty points list provided for RLE conversion")
            return None

        if img_width <= 0 or img_height <= 0:
            raise ValueError(f"Invalid image dimensions: {img_width}x{img_height}")

        if not self.has_coco_mask:
            if require_coco_mask:
                raise ImportError("pycocotools required for RLE encoding")
            else:
                self.logger.warning("pycocotools not available, cannot encode RLE")
                return None

        try:
            # Convert normalized coordinates to absolute coordinates
            abs_points = [(int(x * img_width), int(y * img_height)) for x, y in points]

            # Create binary mask
            import cv2

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
                if "counts" in rle_dict and isinstance(rle_dict["counts"], bytes):
                    # Convert bytes to string (UTF-8 encoding should work for RLE)
                    rle_dict["counts"] = rle_dict["counts"].decode("utf-8")
                return rle_dict
            else:
                # If not a dict, return as-is (shouldn't happen with pycocotools)
                self.logger.warning(f"Unexpected RLE type: {type(rle)}")
                return rle

        except ImportError as e:
            self.logger.error(f"Import error during RLE encoding: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error encoding polygon to RLE: {e}")
            raise

    def rle_to_polygon(
        self, rle: Dict, img_width: int, img_height: int, require_coco_mask: bool = True
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Decode RLE format to polygon points.

        Args:
            rle: RLE dict with 'size' and 'counts' fields
            img_width: Image width in pixels
            img_height: Image height in pixels
            require_coco_mask: If True, raise ImportError when pycocotools not available.
                               If False, return None when pycocotools not available.

        Returns:
            List of normalized (x, y) polygon points in range [0, 1], or None if failed.

        Raises:
            ImportError: If pycocotools not available and require_coco_mask=True
            ValueError: If RLE dict is invalid or image dimensions are invalid
        """
        if not rle or "size" not in rle or "counts" not in rle:
            raise ValueError(f"Invalid RLE dict: missing 'size' or 'counts' fields")

        if img_width <= 0 or img_height <= 0:
            raise ValueError(f"Invalid image dimensions: {img_width}x{img_height}")

        if not self.has_coco_mask:
            if require_coco_mask:
                raise ImportError("pycocotools required for RLE decoding")
            else:
                self.logger.warning("pycocotools not available, cannot decode RLE")
                return None

        try:
            # Make a copy of RLE dict to avoid modifying original
            rle_dict = dict(rle)

            # Ensure 'counts' is bytes for coco_mask.decode
            if "counts" in rle_dict and isinstance(rle_dict["counts"], str):
                rle_dict["counts"] = rle_dict["counts"].encode("utf-8")

            # Decode RLE to binary mask
            binary_mask = coco_mask.decode(rle_dict)

            # Extract contours from mask
            import cv2

            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                self.logger.warning("No contours found in RLE mask")
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

        except ImportError as e:
            self.logger.error(f"Import error during RLE decoding: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error decoding RLE to polygon: {e}")
            raise

    def get_rle_accuracy_warning(self) -> str:
        """
        Get warning message about RLE conversion accuracy loss.

        Returns:
            Warning message about potential accuracy loss in RLE conversion.
        """
        return (
            "RLE conversion involves accuracy loss: "
            "polygon points are approximated by a binary mask. "
            "For lossless conversion, use polygon format (do_rle=False)."
        )

    def check_coco_mask_available(self) -> bool:
        """
        Check if pycocotools is available for RLE operations.

        Returns:
            True if pycocotools is available, False otherwise.
        """
        return self.has_coco_mask

    def validate_rle_dict(self, rle: Dict) -> bool:
        """
        Validate RLE dictionary structure.

        Args:
            rle: RLE dict to validate

        Returns:
            True if RLE dict appears valid, False otherwise.
        """
        if not isinstance(rle, dict):
            return False

        if "size" not in rle or "counts" not in rle:
            return False

        size = rle["size"]
        if not isinstance(size, list) or len(size) != 2:
            return False

        height, width = size
        if (
            not isinstance(height, int)
            or not isinstance(width, int)
            or height <= 0
            or width <= 0
        ):
            return False

        return True
