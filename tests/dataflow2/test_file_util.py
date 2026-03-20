import os
import tempfile
import pytest
from dataflow2.util.file_util import FileOperations

def test_read_file_success():
    """测试成功读取文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test content")
        file_path = f.name

    try:
        content = FileOperations.read_file(file_path)
        assert content == "test content"
    finally:
        os.unlink(file_path)

def test_read_file_not_found():
    """测试文件不存在"""
    with pytest.raises(FileNotFoundError):
        FileOperations.read_file("/nonexistent/path/file.txt")

def test_write_file_success():
    """测试成功写入文件"""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.txt")
        result = FileOperations.write_file(file_path, "test content")
        assert result is True
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            assert f.read() == "test content"

def test_ensure_directory():
    """测试确保目录存在"""
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = os.path.join(tmpdir, "subdir", "nested")
        result = FileOperations.ensure_directory(new_dir)
        assert result is True
        assert os.path.exists(new_dir)

def test_list_files():
    """测试列出文件"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试文件
        with open(os.path.join(tmpdir, "test1.txt"), 'w') as f:
            f.write("test1")
        with open(os.path.join(tmpdir, "test2.py"), 'w') as f:
            f.write("test2")

        txt_files = FileOperations.list_files(tmpdir, "*.txt")
        assert len(txt_files) == 1
        assert "test1.txt" in txt_files[0]

        all_files = FileOperations.list_files(tmpdir, "*")
        assert len(all_files) == 2