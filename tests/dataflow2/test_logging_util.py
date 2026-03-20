import logging
import pytest
from dataflow2.util.logging_util import LoggingOperations

def test_setup_logger_verbose():
    """测试设置详细日志记录器"""
    logger = LoggingOperations.setup_logger("TestLogger", verbose=True)
    assert logger.name == "TestLogger"
    assert logger.level == logging.INFO

    # 检查是否有处理器
    assert len(logger.handlers) > 0

def test_setup_logger_not_verbose():
    """测试设置非详细日志记录器"""
    logger = LoggingOperations.setup_logger("TestLogger", verbose=False)
    assert logger.name == "TestLogger"
    assert logger.level == logging.WARNING

def test_configure_logging():
    """测试全局日志配置"""
    # 重置日志配置
    logging.root.handlers = []

    LoggingOperations.configure_logging(level=logging.DEBUG)

    # 检查根记录器级别
    assert logging.root.level == logging.DEBUG

    # 检查是否有处理器
    assert len(logging.root.handlers) > 0

def test_logger_output():
    """测试日志输出"""
    import io
    import sys

    # 捕获日志输出
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger("OutputTest")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)

    logger.info("Test message")

    output = stream.getvalue()
    assert "OutputTest - INFO - Test message" in output