"""Shared test fixtures for DataFlow-CV test suite."""

import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Return the test data directory (assets/test_data)."""
    return project_root / "assets" / "test_data"


@pytest.fixture(scope="session")
def test_data_det(test_data_dir):
    """Return the detection test data directory."""
    return test_data_dir / "det"


@pytest.fixture(scope="session")
def test_data_seg(test_data_dir):
    """Return the segmentation test data directory."""
    return test_data_dir / "seg"


@pytest.fixture(scope="session")
def test_data_evaluate(test_data_dir):
    """Return the evaluate test data directory."""
    return test_data_dir / "evaluate"
