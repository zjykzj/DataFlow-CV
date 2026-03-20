import logging
from typing import Optional

class LoggingOperations:
    """日志配置工具类"""

    @staticmethod
    def setup_logger(name: str, verbose: bool = False) -> logging.Logger:
        """设置和配置日志记录器

        Args:
            name: 记录器名称
            verbose: 是否输出详细信息，True为INFO级别，False为WARNING级别

        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger(name)

        # 如果记录器已经有处理器，避免重复添加
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # 设置日志级别
        if verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        # 避免日志传播到根记录器（防止重复输出）
        logger.propagate = False

        return logger

    @staticmethod
    def configure_logging(level: int = logging.WARNING,
                         format_str: Optional[str] = None) -> None:
        """全局日志配置

        Args:
            level: 日志级别，默认为WARNING
            format_str: 日志格式字符串，默认为标准格式
        """
        if format_str is None:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        # 配置根记录器
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[logging.StreamHandler()]
        )