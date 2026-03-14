#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Time    : 2026/3/8 21:20
@File    : run_tests.py
@Author  : zj
@Description: Test runner for DataFlow-CV
"""

import os
import sys
import unittest
import argparse
from pathlib import Path


def run_tests(test_pattern="test_*.py", test_dir=None, verbose=1):
    """
    Run tests using unittest discovery.

    Args:
        test_pattern: Pattern to match test files
        test_dir: Directory to search for tests (default: tests directory)
        verbose: Verbosity level (0=quiet, 1=normal, 2=verbose)

    Returns:
        bool: True if all tests passed, False otherwise
    """
    if test_dir is None:
        # Default to tests directory
        test_dir = str(Path(__file__).parent)

    # Add project root to Python path
    project_root = str(Path(__file__).parent.parent)
    sys.path.insert(0, project_root)

    print(f"Running tests from: {test_dir}")
    print(f"Test pattern: {test_pattern}")
    print(f"Verbosity: {verbose}")
    print("-" * 60)

    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = test_dir

    # Use discovery to find tests
    suite = loader.discover(start_dir=start_dir, pattern=test_pattern)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbose)
    result = runner.run(suite)

    print("-" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}:")
            print(f"    {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}:")
            print(f"    {traceback}")

    return result.wasSuccessful()


def run_specific_test(test_name, verbose=1):
    """
    Run a specific test class or method.

    Args:
        test_name: Test class or method name (e.g., "TestCocoToYoloConverter")
        verbose: Verbosity level

    Returns:
        bool: True if test passed, False otherwise
    """
    print(f"Running specific test: {test_name}")
    print("-" * 60)

    # Add project root to Python path
    project_root = str(Path(__file__).parent.parent)
    sys.path.insert(0, project_root)

    # Load all test modules from all subdirectories
    test_modules = []
    tests_root = Path(__file__).parent

    # Search for test files in all subdirectories
    for test_file in tests_root.rglob("test_*.py"):
        # Convert path to module name (e.g., "tests.convert.test_coco_to_yolo")
        rel_path = test_file.relative_to(tests_root.parent)
        module_name = ".".join(rel_path.with_suffix("").parts)

        try:
            module = __import__(module_name, fromlist=["*"])
            test_modules.append(module)
            print(f"Loaded module: {module_name}")
        except ImportError as e:
            print(f"Failed to load {module_name}: {e}")

    # Find and run the specific test
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for module in test_modules:
        try:
            # Try to load test case class
            test_case_class = getattr(module, test_name, None)
            if test_case_class and issubclass(test_case_class, unittest.TestCase):
                tests = loader.loadTestsFromTestCase(test_case_class)
                suite.addTest(tests)
                print(f"Added test class: {test_name}")
                break

            # Try to load as test method (format: TestClass.test_method)
            if "." in test_name:
                class_name, method_name = test_name.split(".", 1)
                test_case_class = getattr(module, class_name, None)
                if test_case_class and issubclass(test_case_class, unittest.TestCase):
                    test_method = getattr(test_case_class, method_name, None)
                    if test_method:
                        suite.addTest(test_case_class(method_name))
                        print(f"Added test method: {test_name}")
                        break
        except Exception as e:
            print(f"Error loading {test_name} from {module.__name__}: {e}")

    if suite.countTestCases() == 0:
        print(f"Test '{test_name}' not found!")
        return False

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbose)
    result = runner.run(suite)

    print("-" * 60)
    print(f"Test result: {'PASSED' if result.wasSuccessful() else 'FAILED'}")

    return result.wasSuccessful()


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Run DataFlow-CV tests")
    parser.add_argument(
        "--pattern",
        default="test_*.py",
        help="Test file pattern (default: test_*.py)"
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Test directory (default: tests directory)"
    )
    parser.add_argument(
        "--test",
        default=None,
        help="Run specific test class or method (e.g., TestCocoToYoloConverter)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (use -v for normal, -vv for verbose)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (minimal output)"
    )

    args = parser.parse_args()

    # Adjust verbosity
    if args.quiet:
        verbosity = 0
    elif args.verbose >= 2:
        verbosity = 2
    else:
        verbosity = 1

    # Run tests
    if args.test:
        success = run_specific_test(args.test, verbose=verbosity)
    else:
        success = run_tests(args.pattern, args.dir, verbose=verbosity)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()