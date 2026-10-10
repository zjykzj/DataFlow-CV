"""
Unit tests for label utility functions.
"""

from dataflow.label.utils import parse_yolo_class_id, resolve_coco_image_ids


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


class TestResolveCocoImageIds:
    """Tests for resolve_coco_image_ids — COCO image id resolution."""

    def test_digit_ids_keep_values(self):
        """Digit-string ids resolve to their int values."""
        assert resolve_coco_image_ids(["1", "2", "5"]) == {"1": 1, "2": 2, "5": 5}

    def test_zero_padded_digit_ids(self):
        """Zero-padded ids resolve to int (roundtrip-friendly)."""
        assert resolve_coco_image_ids(["01", "007"]) == {"01": 1, "007": 7}

    def test_non_digit_ids_get_counters(self):
        """Non-digit ids get dedicated counter ids."""
        assert resolve_coco_image_ids(["a", "b"]) == {"a": 1, "b": 2}

    def test_counters_skip_digit_values(self):
        """Counters never take values reserved by digit ids (even later ones)."""
        assert resolve_coco_image_ids(["a", "b", "5"]) == {"a": 1, "b": 2, "5": 5}
        # Four non-digit ids before a digit "5": counters must pass over 5
        assert resolve_coco_image_ids(["a", "b", "c", "d", "e", "5"]) == {
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 4,
            "e": 6,
            "5": 5,
        }

    def test_colliding_digit_ids(self):
        """Colliding digit ids ('01' vs '1') get unique ids."""
        resolved = resolve_coco_image_ids(["01", "1"])
        assert resolved["01"] == 1
        assert resolved["1"] != 1
        assert len(set(resolved.values())) == 2

    def test_degenerate_ids_get_counters(self):
        """Empty and negative strings are non-digit → counters."""
        resolved = resolve_coco_image_ids(["", "-1"])
        assert resolved[""] == 1
        assert resolved["-1"] == 2

    def test_uniqueness_with_mixed_inputs(self):
        """Mixed digit/non-digit/colliding inputs all stay unique ints."""
        resolved = resolve_coco_image_ids(["01", "1", "frame_a", "2"])
        values = list(resolved.values())
        assert len(set(values)) == 4
        assert all(isinstance(v, int) and v > 0 for v in values)
        assert resolved["2"] == 2
