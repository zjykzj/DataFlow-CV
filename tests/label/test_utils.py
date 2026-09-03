"""
Unit tests for label utility functions.
"""

from dataflow.label.utils import parse_yolo_class_id


class TestParseYoloClassId:
    """Tests for parse_yolo_class_id — float-tolerant YOLO class ID parser."""

    def test_integer_string(self):
        """Plain integer string → int."""
        assert parse_yolo_class_id("5") == 5
        assert parse_yolo_class_id("0") == 0
        assert parse_yolo_class_id("999") == 999

    def test_integer_valued_float(self):
        """Float-formatted integer string → int (common in YOLO tooling)."""
        assert parse_yolo_class_id("5.000000") == 5
        assert parse_yolo_class_id("0.0") == 0
        assert parse_yolo_class_id("0.00") == 0
        assert parse_yolo_class_id("1.0") == 1
        assert parse_yolo_class_id("42.000000") == 42

    def test_non_integer_float_returns_none(self):
        """Non-integer float → None (e.g. '0.5' is not a valid class ID)."""
        assert parse_yolo_class_id("0.5") is None
        assert parse_yolo_class_id("1.5") is None
        assert parse_yolo_class_id("3.14") is None

    def test_negative_values_return_none(self):
        """Negative class IDs → None.  Note: -0.0 is 0.0 in IEEE 754,
        so parse_yolo_class_id('-0.0') returns 0 (valid edge case)."""
        assert parse_yolo_class_id("-1") is None
        assert parse_yolo_class_id("-5") is None
        assert parse_yolo_class_id("-3.5") is None

    def test_negative_zero_returns_zero(self):
        """-0.0 equals 0.0 in floating point — treated as valid class 0."""
        assert parse_yolo_class_id("-0.0") == 0

    def test_non_numeric_string_returns_none(self):
        """Non-numeric strings → None."""
        assert parse_yolo_class_id("abc") is None
        assert parse_yolo_class_id("") is None
        assert parse_yolo_class_id(" ") is None
        assert parse_yolo_class_id("class_5") is None

    def test_whitespace(self):
        """Whitespace-only → None."""
        assert parse_yolo_class_id("  ") is None
        assert parse_yolo_class_id("\t") is None

    def test_scientific_notation_valid(self):
        """Scientific notation for integer values → int."""
        assert parse_yolo_class_id("1e0") == 1
        assert parse_yolo_class_id("5.0e0") == 5

    def test_overflow_returns_none(self):
        """Extremely large values that overflow → None."""
        assert parse_yolo_class_id("1e999") is None
