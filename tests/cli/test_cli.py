#!/usr/bin/env python3
"""
Test for DataFlow CLI interface.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import dataflow
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_cli_import():
    """Test CLI import."""
    print("Testing CLI import...")

    try:
        from dataflow.cli import main
        print("  ✓ CLI module imports successfully")
        return True
    except Exception as e:
        print(f"  ✗ CLI module import failed: {e}")
        return False


def test_cli_help():
    """Test CLI help command."""
    print("Testing CLI help command...")

    try:
        import subprocess
        import dataflow

        # Run dataflow --help
        result = subprocess.run(
            [sys.executable, "-m", "dataflow.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print("  ✓ CLI help command works")
            return True
        else:
            print(f"  ✗ CLI help command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ✗ CLI help command test failed: {e}")
        return False


def test_convert_help():
    """Test convert help command."""
    print("Testing convert help command...")

    try:
        import subprocess

        # Run dataflow convert --help
        result = subprocess.run(
            [sys.executable, "-m", "dataflow.cli", "convert", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print("  ✓ Convert help command works")
            return True
        else:
            print(f"  ✗ Convert help command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ✗ Convert help command test failed: {e}")
        return False


def test_visualize_help():
    """Test visualize help command."""
    print("Testing visualize help command...")

    try:
        import subprocess

        # Run dataflow visualize --help
        result = subprocess.run(
            [sys.executable, "-m", "dataflow.cli", "visualize", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print("  ✓ Visualize help command works")
            return True
        else:
            print(f"  ✗ Visualize help command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ✗ Visualize help command test failed: {e}")
        return False


def main():
    """Run all CLI tests."""
    print("=" * 60)
    print("DataFlow CLI Tests")
    print("=" * 60)

    all_passed = True

    # Run tests
    tests = [
        test_cli_import,
        test_cli_help,
        test_convert_help,
        test_visualize_help
    ]

    for test_func in tests:
        if test_func():
            print(f"  ✓ {test_func.__name__} PASSED")
        else:
            print(f"  ✗ {test_func.__name__} FAILED")
            all_passed = False
        print()

    print("=" * 60)
    if all_passed:
        print("All CLI tests PASSED! 🎉")
    else:
        print("Some CLI tests FAILED! ❌")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)