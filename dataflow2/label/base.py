import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class BaseAnnotationHandler(ABC):
    """标注处理器基类（抽象类）

    所有格式处理器必须继承此类并实现抽象方法。
    """

    def __init__(self, verbose: bool = False):
        """初始化标注处理器

        Args:
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器

        Returns:
            配置好的日志记录器
        """
        from dataflow2.util.logging_util import LoggingOperations
        return LoggingOperations.setup_logger(
            self.__class__.__name__,
            self.verbose
        )

    @abstractmethod
    def read(self, label_path: str, **kwargs) -> Dict[str, Any]:
        """读取单个标签文件

        Args:
            label_path: 标签文件路径
            **kwargs: 格式特定的参数（如图像路径、类别文件等）

        Returns:
            统一格式的图像标注数据字典

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        pass

    @abstractmethod
    def read_batch(self, **kwargs) -> List[Dict[str, Any]]:
        """批量读取标签文件

        Args:
            **kwargs: 格式特定的批量读取参数

        Returns:
            图像标注数据列表

        Raises:
            FileNotFoundError: 目录或文件不存在
            ValueError: 批量读取过程中发生错误
        """
        pass

    @abstractmethod
    def write(self, image_annotations: Dict[str, Any],
              output_path: str, **kwargs) -> bool:
        """写入单个标签文件

        Args:
            image_annotations: 统一格式的图像标注数据
            output_path: 输出文件路径
            **kwargs: 格式特定的写入参数

        Returns:
            是否写入成功

        Raises:
            ValueError: 数据格式错误
        """
        pass

    @abstractmethod
    def write_batch(self, images_annotations: List[Dict[str, Any]],
                   **kwargs) -> bool:
        """批量写入标签文件

        Args:
            images_annotations: 图像标注数据列表
            **kwargs: 格式特定的批量写入参数

        Returns:
            是否全部写入成功
        """
        pass

    def _validate_input_path(self, path: str) -> bool:
        """验证输入路径

        Args:
            path: 要验证的路径

        Returns:
            路径是否有效

        Raises:
            FileNotFoundError: 路径不存在
        """
        if not os.path.exists(path):
            self.logger.error(f"输入路径不存在: {path}")
            raise FileNotFoundError(f"路径不存在: {path}")
        return True

    def _validate_output_path(self, path: str) -> bool:
        """验证输出路径

        Args:
            path: 要验证的路径

        Returns:
            路径是否有效
        """
        # 对于输出路径，我们只需要确保目录存在
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        return True

    def _log_operation_start(self, operation: str, **details):
        """记录操作开始

        Args:
            operation: 操作名称
            **details: 操作详情
        """
        if self.verbose:
            details_str = ", ".join(f"{k}={v}" for k, v in details.items())
            self.logger.info(f"开始 {operation}: {details_str}")

    def _log_operation_complete(self, operation: str, result: Any, **details):
        """记录操作完成

        Args:
            operation: 操作名称
            result: 操作结果
            **details: 操作详情
        """
        if self.verbose:
            details_str = ", ".join(f"{k}={v}" for k, v in details.items())
            self.logger.info(f"完成 {operation}: {details_str}, 结果: {result}")

    def _log_warning(self, message: str, **context):
        """记录警告

        Args:
            message: 警告消息
            **context: 上下文信息
        """
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        full_message = f"{message}" + (f" ({context_str})" if context_str else "")
        self.logger.warning(full_message)

    def _log_error(self, message: str, exception: Optional[Exception] = None):
        """记录错误

        Args:
            message: 错误消息
            exception: 异常对象（可选）
        """
        if exception:
            self.logger.error(f"{message}: {exception}")
        else:
            self.logger.error(message)