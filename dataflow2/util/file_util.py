import os
import glob
from typing import List, Optional

class FileOperations:
    """文件操作工具类"""

    @staticmethod
    def read_file(file_path: str, encoding: str = 'utf-8') -> str:
        """读取文本文件

        Args:
            file_path: 文件路径
            encoding: 文件编码，默认为utf-8

        Returns:
            文件内容字符串

        Raises:
            FileNotFoundError: 文件不存在
            IOError: 读取文件时发生错误
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            raise IOError(f"读取文件失败: {file_path}, 错误: {e}")

    @staticmethod
    def write_file(file_path: str, content: str, encoding: str = 'utf-8') -> bool:
        """写入文本文件

        Args:
            file_path: 文件路径
            content: 要写入的内容
            encoding: 文件编码，默认为utf-8

        Returns:
            是否写入成功
        """
        try:
            # 确保目录存在
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            return True
        except Exception as e:
            return False

    @staticmethod
    def ensure_directory(dir_path: str) -> bool:
        """确保目录存在

        Args:
            dir_path: 目录路径

        Returns:
            是否创建成功或已存在
        """
        try:
            os.makedirs(dir_path, exist_ok=True)
            return True
        except Exception as e:
            return False

    @staticmethod
    def list_files(directory: str, pattern: str = "*") -> List[str]:
        """列出匹配模式的文件

        Args:
            directory: 目录路径
            pattern: 文件匹配模式，默认为"*"

        Returns:
            匹配的文件路径列表
        """
        if not os.path.exists(directory):
            return []

        try:
            search_pattern = os.path.join(directory, pattern)
            return glob.glob(search_pattern)
        except Exception:
            return []

    @staticmethod
    def batch_read(files: List[str], on_error: str = 'strict') -> List[str]:
        """批量读取文件

        Args:
            files: 文件路径列表
            on_error: 错误处理模式，'strict'表示遇到错误立即停止

        Returns:
            文件内容列表

        Raises:
            FileNotFoundError: 文件不存在（严格模式）
            IOError: 读取文件时发生错误（严格模式）
        """
        results = []
        for file_path in files:
            try:
                content = FileOperations.read_file(file_path)
                results.append(content)
            except Exception as e:
                if on_error == 'strict':
                    raise
                # 宽松模式下跳过错误文件
                continue
        return results