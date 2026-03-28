"""
Unit tests for rle_converter.py
"""

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from dataflow.convert.rle_converter import RLEConverter


class TestRLEConverter:
    """Test suite for RLEConverter class."""

    def test_init_with_coco_mask(self):
        """Test initialization when pycocotools is available."""
        with patch("dataflow.convert.rle_converter.HAS_COCO_MASK", True):
            converter = RLEConverter()
            assert converter.has_coco_mask is True
            assert converter.logger is not None

    def test_init_without_coco_mask(self):
        """Test initialization when pycocotools is not available."""
        with patch("dataflow.convert.rle_converter.HAS_COCO_MASK", False):
            converter = RLEConverter()
            assert converter.has_coco_mask is False
            assert converter.logger is not None

    def test_get_rle_accuracy_warning(self):
        """Test RLE accuracy warning message."""
        converter = RLEConverter()
        warning_msg = converter.get_rle_accuracy_warning()
        assert isinstance(warning_msg, str)
        assert "accuracy loss" in warning_msg.lower()
        assert "polygon" in warning_msg.lower()

    def test_check_coco_mask_available_true(self):
        """Test check_coco_mask_available when pycocotools is available."""
        with patch("dataflow.convert.rle_converter.HAS_COCO_MASK", True):
            converter = RLEConverter()
            assert converter.check_coco_mask_available() is True

    def test_check_coco_mask_available_false(self):
        """Test check_coco_mask_available when pycocotools is not available."""
        with patch("dataflow.convert.rle_converter.HAS_COCO_MASK", False):
            converter = RLEConverter()
            assert converter.check_coco_mask_available() is False

    def test_validate_rle_dict_valid(self):
        """Test validate_rle_dict with valid RLE dict."""
        converter = RLEConverter()
        valid_rle = {"size": [100, 200], "counts": "some_rle_string"}
        assert converter.validate_rle_dict(valid_rle) is True

    def test_validate_rle_dict_invalid_missing_fields(self):
        """Test validate_rle_dict with missing fields."""
        converter = RLEConverter()
        invalid_rle = {"size": [100, 200]}  # missing 'counts'
        assert converter.validate_rle_dict(invalid_rle) is False

        invalid_rle2 = {"counts": "some_rle_string"}  # missing 'size'
        assert converter.validate_rle_dict(invalid_rle2) is False

    def test_validate_rle_dict_invalid_size_format(self):
        """Test validate_rle_dict with invalid size format."""
        converter = RLEConverter()
        # size not a list
        invalid_rle = {"size": (100, 200), "counts": "some_rle_string"}
        assert converter.validate_rle_dict(invalid_rle) is False

        # size list length not 2
        invalid_rle = {"size": [100, 200, 300], "counts": "some_rle_string"}
        assert converter.validate_rle_dict(invalid_rle) is False

        # size values not positive integers
        invalid_rle = {"size": [0, 200], "counts": "some_rle_string"}
        assert converter.validate_rle_dict(invalid_rle) is False

        invalid_rle = {"size": [100, -200], "counts": "some_rle_string"}
        assert converter.validate_rle_dict(invalid_rle) is False

    def test_validate_rle_dict_not_dict(self):
        """Test validate_rle_dict with non-dict input."""
        converter = RLEConverter()
        assert converter.validate_rle_dict(None) is False
        assert converter.validate_rle_dict("not a dict") is False
        assert converter.validate_rle_dict([]) is False

    @patch("dataflow.convert.rle_converter.HAS_COCO_MASK", True)
    @patch("dataflow.convert.rle_converter.coco_mask")
    @patch("dataflow.convert.rle_converter.np")
    def test_polygon_to_rle_with_coco_mask(self, mock_np, mock_coco_mask):
        """Test polygon_to_rle when pycocotools is available."""
        # Mock cv2 module before importing it in the method
        with patch.dict("sys.modules", {"cv2": MagicMock()}):
            converter = RLEConverter()

            # Setup mocks
            mock_mask = Mock()
            mock_np.zeros.return_value = mock_mask
            mock_np.asfortranarray.return_value = mock_mask
            mock_np.array.return_value = Mock()

            mock_coco_mask.encode.return_value = {
                "size": [100, 200],
                "counts": b"rle_bytes",
            }

            # Get the mocked cv2 module
            import sys

            mock_cv2 = sys.modules["cv2"]
            mock_cv2.fillPoly = Mock()

            # Test data
            points = [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4)]
            img_width = 500
            img_height = 300

            result = converter.polygon_to_rle(points, img_width, img_height)

            # Verify result
            assert result is not None
            assert "size" in result
            assert "counts" in result
            # counts should be string (converted from bytes)
            assert isinstance(result["counts"], str)

            # Verify mocks were called
            mock_np.zeros.assert_called_once()
            mock_cv2.fillPoly.assert_called_once()
            mock_coco_mask.encode.assert_called_once_with(mock_mask)

    @patch("dataflow.convert.rle_converter.HAS_COCO_MASK", False)
    def test_polygon_to_rle_without_coco_mask_require_false(self):
        """Test polygon_to_rle when pycocotools not available and require_coco_mask=False."""
        converter = RLEConverter()

        points = [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4)]
        img_width = 500
        img_height = 300

        # Should return None without raising exception
        result = converter.polygon_to_rle(
            points, img_width, img_height, require_coco_mask=False
        )
        assert result is None

    @patch("dataflow.convert.rle_converter.HAS_COCO_MASK", False)
    def test_polygon_to_rle_without_coco_mask_require_true(self):
        """Test polygon_to_rle when pycocotools not available and require_coco_mask=True."""
        converter = RLEConverter()

        points = [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4)]
        img_width = 500
        img_height = 300

        # Should raise ImportError
        with pytest.raises(ImportError, match="pycocotools required"):
            converter.polygon_to_rle(
                points, img_width, img_height, require_coco_mask=True
            )

    def test_polygon_to_rle_empty_points(self):
        """Test polygon_to_rle with empty points list."""
        converter = RLEConverter()

        result = converter.polygon_to_rle([], 100, 100, require_coco_mask=False)
        assert result is None

    def test_polygon_to_rle_invalid_dimensions(self):
        """Test polygon_to_rle with invalid image dimensions."""
        converter = RLEConverter()

        points = [(0.1, 0.2), (0.3, 0.2)]

        # Zero width
        with pytest.raises(ValueError):
            converter.polygon_to_rle(points, 0, 100)

        # Zero height
        with pytest.raises(ValueError):
            converter.polygon_to_rle(points, 100, 0)

        # Negative width
        with pytest.raises(ValueError):
            converter.polygon_to_rle(points, -100, 100)

    @patch("dataflow.convert.rle_converter.HAS_COCO_MASK", True)
    @patch("dataflow.convert.rle_converter.coco_mask")
    def test_rle_to_polygon_with_coco_mask(self, mock_coco_mask):
        """Test rle_to_polygon when pycocotools is available."""
        # Mock cv2 module before importing it in the method
        with patch.dict("sys.modules", {"cv2": MagicMock()}):
            converter = RLEConverter()

            # Setup mocks
            mock_binary_mask = Mock()
            mock_coco_mask.decode.return_value = mock_binary_mask

            # Get the mocked cv2 module
            import sys

            mock_cv2 = sys.modules["cv2"]
            mock_contours = [Mock(), Mock()]
            mock_cv2.findContours = Mock(return_value=(mock_contours, None))
            mock_cv2.contourArea = Mock(side_effect=[100, 50])  # First contour larger

            # Mock contour points
            mock_contour = [[[10, 20]], [[30, 40]], [[50, 60]]]
            mock_contours[0] = mock_contour

            # Test data
            rle_dict = {"size": [100, 200], "counts": "rle_string"}
            img_width = 200
            img_height = 100

            result = converter.rle_to_polygon(rle_dict, img_width, img_height)

            # Verify result
            assert isinstance(result, list)
            # Should have 3 points (from mock contour)
            # Actually we need to check the conversion logic

            # Verify mocks were called
            mock_coco_mask.decode.assert_called_once()
            mock_cv2.findContours.assert_called_once()

    @patch("dataflow.convert.rle_converter.HAS_COCO_MASK", False)
    def test_rle_to_polygon_without_coco_mask_require_false(self):
        """Test rle_to_polygon when pycocotools not available and require_coco_mask=False."""
        converter = RLEConverter()

        rle_dict = {"size": [100, 200], "counts": "rle_string"}
        img_width = 200
        img_height = 100

        # Should return None without raising exception
        result = converter.rle_to_polygon(
            rle_dict, img_width, img_height, require_coco_mask=False
        )
        assert result is None

    @patch("dataflow.convert.rle_converter.HAS_COCO_MASK", False)
    def test_rle_to_polygon_without_coco_mask_require_true(self):
        """Test rle_to_polygon when pycocotools not available and require_coco_mask=True."""
        converter = RLEConverter()

        rle_dict = {"size": [100, 200], "counts": "rle_string"}
        img_width = 200
        img_height = 100

        # Should raise ImportError
        with pytest.raises(ImportError, match="pycocotools required"):
            converter.rle_to_polygon(
                rle_dict, img_width, img_height, require_coco_mask=True
            )

    def test_rle_to_polygon_invalid_rle_dict(self):
        """Test rle_to_polygon with invalid RLE dict."""
        converter = RLEConverter()

        # Missing 'size' field
        with pytest.raises(ValueError, match="Invalid RLE dict"):
            converter.rle_to_polygon({"counts": "rle_string"}, 100, 100)

        # Missing 'counts' field
        with pytest.raises(ValueError, match="Invalid RLE dict"):
            converter.rle_to_polygon({"size": [100, 200]}, 100, 100)

        # Empty dict
        with pytest.raises(ValueError, match="Invalid RLE dict"):
            converter.rle_to_polygon({}, 100, 100)

    def test_rle_to_polygon_invalid_dimensions(self):
        """Test rle_to_polygon with invalid image dimensions."""
        converter = RLEConverter()

        rle_dict = {"size": [100, 200], "counts": "rle_string"}

        # Zero width
        with pytest.raises(ValueError):
            converter.rle_to_polygon(rle_dict, 0, 100)

        # Zero height
        with pytest.raises(ValueError):
            converter.rle_to_polygon(rle_dict, 100, 0)

        # Negative width
        with pytest.raises(ValueError):
            converter.rle_to_polygon(rle_dict, -100, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
