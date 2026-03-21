#!/usr/bin/env python3
"""
LoggingOperations usage demo.

Demonstrates how to use the LoggingOperations class for flexible logging configuration.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataflow.util.logging_util import LoggingOperations


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("LoggingOperations Demonstration")
    print("=" * 60)

    # Create LoggingOperations instance
    log_ops = LoggingOperations()

    # 1. Basic logger with console output
    print("\n1. Basic Console Logger")
    print("-" * 40)

    logger1 = log_ops.get_logger("demo.basic", level="INFO")
    logger1.info("This is an INFO message to console")
    logger1.debug("This DEBUG message won't appear (level is INFO)")
    logger1.warning("This is a WARNING message")

    # 2. Logger with file output
    print("\n2. Logger with File Output")
    print("-" * 40)

    # Create a temporary directory for log files
    temp_dir = Path(tempfile.mkdtemp(prefix="logs_"))
    log_file = temp_dir / "application.log"

    logger2 = log_ops.get_logger("demo.file", log_file=str(log_file), level="DEBUG")
    logger2.debug("Debug message to file")
    logger2.info("Info message to file and console")
    logger2.error("Error message to file and console")

    print(f"Log file created at: {log_file}")
    print(f"Log file size: {log_file.stat().st_size} bytes")

    # 3. Logger without console output
    print("\n3. Logger without Console Output")
    print("-" * 40)

    silent_log_file = temp_dir / "silent.log"
    logger3 = log_ops.get_logger("demo.silent", log_file=str(silent_log_file), console=False)
    logger3.info("This message only goes to file, not console")

    print(f"Silent log file created: {silent_log_file.exists()}")

    # 4. Multiple independent loggers
    print("\n4. Multiple Independent Loggers")
    print("-" * 40)

    app_logger = log_ops.get_logger("demo.app", level="INFO")
    db_logger = log_ops.get_logger("demo.db", level="DEBUG")
    api_logger = log_ops.get_logger("demo.api", level="WARNING")

    app_logger.info("Application started")
    db_logger.debug("Database query executed")
    api_logger.warning("API rate limit approaching")
    api_logger.info("This won't appear (level is WARNING)")

    print("Different loggers can have different levels and handlers")

    # 5. Dynamic log level change
    print("\n5. Dynamic Log Level Change")
    print("-" * 40)

    dynamic_logger = log_ops.get_logger("demo.dynamic", level="ERROR")
    dynamic_logger.info("This won't appear (level is ERROR)")
    dynamic_logger.error("This ERROR will appear")

    # Change to DEBUG level
    log_ops.set_log_level("demo.dynamic", "DEBUG")
    dynamic_logger.debug("Now DEBUG messages appear!")
    dynamic_logger.info("INFO messages also appear now")

    # 6. Root logger configuration
    print("\n6. Root Logger Configuration")
    print("-" * 40)

    root_log_file = temp_dir / "root.log"
    log_ops.setup_root_logger(level="WARNING", log_file=str(root_log_file))

    # These will use the root logger configuration
    import logging
    logging.warning("Root logger WARNING message")
    logging.error("Root logger ERROR message")
    logging.info("This INFO won't appear (root level is WARNING)")

    print(f"Root log file created: {root_log_file.exists()}")

    # 7. Timestamped log files
    print("\n7. Timestamped Log Files")
    print("-" * 40)

    timestamped_log = log_ops.create_log_file("daily", directory=str(temp_dir))
    print(f"Generated log file path: {timestamped_log}")

    # Create logger with timestamped file
    daily_logger = log_ops.get_logger("demo.daily", log_file=timestamped_log)
    daily_logger.info("Log entry in timestamped file")

    # 8. Log message formatting
    print("\n8. Log Message Formatting")
    print("-" * 40)

    formatted_logger = log_ops.get_logger("demo.formatted")
    formatted_logger.info("Simple message")
    formatted_logger.info("Message with %s formatting", "string")
    formatted_logger.info("Message with number: %d", 42)
    formatted_logger.info("Message with multiple: %s %d", "test", 123)

    # 9. Error and exception logging
    print("\n9. Error and Exception Logging")
    print("-" * 40)

    error_logger = log_ops.get_logger("demo.error")

    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        error_logger.error("Division by zero error occurred")
        error_logger.exception("Exception details with traceback:")

    # 10. Performance consideration - cached loggers
    print("\n10. Logger Caching Demonstration")
    print("-" * 40)

    import time

    start = time.time()
    for i in range(100):
        logger = log_ops.get_logger(f"demo.performance.{i}")
    uncached_time = time.time() - start

    start = time.time()
    for i in range(100):
        logger = log_ops.get_logger("demo.cached")  # Same name
    cached_time = time.time() - start

    print(f"Time for 100 different loggers: {uncached_time:.4f}s")
    print(f"Time for 100 same logger (cached): {cached_time:.4f}s")
    print(f"Caching is {uncached_time/cached_time:.1f}x faster")

    # Display log file contents
    print("\n" + "=" * 60)
    print("Sample Log File Contents")
    print("=" * 60)

    # Read and display first few lines of main log file
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()[:10]

        print(f"\nFirst {len(lines)} lines of {log_file.name}:")
        for line in lines:
            print(f"  {line.rstrip()}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("Demonstration Completed Successfully!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("1. Flexible logger configuration per module/component")
    print("2. Support for both console and file output")
    print("3. Dynamic log level changes at runtime")
    print("4. Automatic logger caching for performance")
    print("5. Consistent log format across application")
    print("6. Easy root logger configuration")


if __name__ == "__main__":
    main()