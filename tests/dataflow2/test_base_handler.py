import pytest
from abc import ABC
from dataflow2.label.base import BaseAnnotationHandler

def test_base_handler_is_abstract():
    """测试BaseAnnotationHandler是抽象类"""
    assert issubclass(BaseAnnotationHandler, ABC)

    # 尝试实例化应该失败
    with pytest.raises(TypeError):
        handler = BaseAnnotationHandler()

def test_base_handler_abstract_methods():
    """测试抽象方法的存在"""
    # 检查必需的抽象方法
    required_methods = ['read', 'read_batch', 'write', 'write_batch']
    for method_name in required_methods:
        assert hasattr(BaseAnnotationHandler, method_name)
        method = getattr(BaseAnnotationHandler, method_name)
        assert getattr(method, '__isabstractmethod__', False)

def test_concrete_handler_implementation():
    """测试具体处理器实现"""
    class ConcreteHandler(BaseAnnotationHandler):
        def read(self, label_path: str, **kwargs):
            return {"test": "data"}

        def read_batch(self, **kwargs):
            return [{"test": "data"}]

        def write(self, image_annotations: dict, output_path: str, **kwargs):
            return True

        def write_batch(self, images_annotations: list, **kwargs):
            return True

    # 现在应该可以实例化
    handler = ConcreteHandler(verbose=True)
    assert handler.verbose is True
    assert handler.logger is not None
    assert handler.logger.name == "ConcreteHandler"

    # 测试方法调用
    result = handler.read("test.txt")
    assert result == {"test": "data"}

def test_validation_methods():
    """测试验证方法"""
    class TestHandler(BaseAnnotationHandler):
        def read(self, label_path: str, **kwargs):
            return {}

        def read_batch(self, **kwargs):
            return []

        def write(self, image_annotations: dict, output_path: str, **kwargs):
            return True

        def write_batch(self, images_annotations: list, **kwargs):
            return True

    handler = TestHandler()

    # 测试验证方法存在
    assert hasattr(handler, '_validate_input_path')
    assert hasattr(handler, '_validate_output_path')

    # 测试验证方法可调用
    assert callable(handler._validate_input_path)
    assert callable(handler._validate_output_path)