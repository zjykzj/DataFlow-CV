"""
Unit tests for file_util.py
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from dataflow.util.file_util import FileOperations


class TestFileOperations:
    """Test suite for FileOperations class."""

    @pytest.fixture
    def file_ops(self):
        """Create a FileOperations instance for testing."""
        return FileOperations()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_ensure_dir_creates_new(self, file_ops, temp_dir):
        """Test ensure_dir creates new directory."""
        new_dir = temp_dir / "new_subdir"
        assert not new_dir.exists()

        result = file_ops.ensure_dir(new_dir)
        assert result == new_dir
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_dir_exists(self, file_ops, temp_dir):
        """Test ensure_dir on existing directory."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()

        result = file_ops.ensure_dir(existing_dir)
        assert result == existing_dir
        assert existing_dir.exists()

    def test_safe_remove_file(self, file_ops, temp_dir):
        """Test safe_remove on a file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()

        result = file_ops.safe_remove(test_file)
        assert result is True
        assert not test_file.exists()

    def test_safe_remove_nonexistent(self, file_ops, temp_dir):
        """Test safe_remove on non-existent file."""
        non_existent = temp_dir / "ghost.txt"
        result = file_ops.safe_remove(non_existent)
        assert result is False

    def test_copy_files(self, file_ops, temp_dir):
        """Test copying files with pattern."""
        src_dir = temp_dir / "src"
        src_dir.mkdir()
        dst_dir = temp_dir / "dst"

        # Create test files
        file1 = src_dir / "test1.txt"
        file2 = src_dir / "test2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        results = file_ops.copy_files(str(src_dir / "*.txt"), dst_dir)
        assert len(results) == 2

        # Check files were copied
        copied_files = [dst_dir / "test1.txt", dst_dir / "test2.txt"]
        for cf in copied_files:
            assert cf.exists()

        # Check results contain success flags
        success_flags = [success for _, success in results]
        assert all(success_flags)

    def test_copy_files_overwrite(self, file_ops, temp_dir):
        """Test copy_files with overwrite option."""
        src_dir = temp_dir / "src"
        src_dir.mkdir()
        dst_dir = temp_dir / "dst"
        dst_dir.mkdir()

        src_file = src_dir / "test.txt"
        dst_file = dst_dir / "test.txt"
        src_file.write_text("new content")
        dst_file.write_text("old content")

        # Without overwrite, should skip
        results = file_ops.copy_files(str(src_dir / "*.txt"), dst_dir, overwrite=False)
        assert len(results) == 1
        assert results[0][1] is False  # skipped

        # With overwrite, should copy
        results = file_ops.copy_files(str(src_dir / "*.txt"), dst_dir, overwrite=True)
        assert len(results) == 1
        assert results[0][1] is True
        assert dst_file.read_text() == "new content"

    def test_find_files(self, file_ops, temp_dir):
        """Test find_files with pattern matching."""
        # Create test directory structure
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        (temp_dir / "file1.txt").write_text("")
        (temp_dir / "file2.txt").write_text("")
        (temp_dir / "file3.jpg").write_text("")
        (subdir / "file4.txt").write_text("")

        # Non-recursive
        files = file_ops.find_files(temp_dir, "*.txt", recursive=False)
        assert len(files) == 2
        filenames = [f.name for f in files]
        assert "file1.txt" in filenames
        assert "file2.txt" in filenames

        # Recursive
        files = file_ops.find_files(temp_dir, "*.txt", recursive=True)
        assert len(files) == 3  # file1, file2, file4
        assert any("file4.txt" in str(f) for f in files)

    def test_create_temp_dir(self, file_ops):
        """Test create_temp_dir creates temporary directory."""
        temp_dir = file_ops.create_temp_dir(prefix="test_")
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        assert "test_" in temp_dir.name

        # Clean up
        shutil.rmtree(temp_dir)

    def test_get_file_size(self, file_ops, temp_dir):
        """Test get_file_size returns correct size."""
        content = "Hello, World!"
        test_file = temp_dir / "test.txt"
        test_file.write_text(content)

        size = file_ops.get_file_size(test_file)
        expected = len(content.encode("utf-8"))
        assert size == expected

    def test_get_file_size_nonexistent(self, file_ops, temp_dir):
        """Test get_file_size raises FileNotFoundError for non-existent file."""
        non_existent = temp_dir / "ghost.txt"
        with pytest.raises(FileNotFoundError):
            file_ops.get_file_size(non_existent)

    def test_read_lines(self, file_ops, temp_dir):
        """Test read_lines reads all lines."""
        lines = ["Line 1", "Line 2", "Line 3"]
        test_file = temp_dir / "test.txt"
        test_file.write_text("\n".join(lines))

        read_lines = file_ops.read_lines(test_file)
        assert read_lines == lines

    def test_write_lines(self, file_ops, temp_dir):
        """Test write_lines writes lines to file."""
        lines = ["First line", "Second line", "Third line"]
        test_file = temp_dir / "test.txt"

        result = file_ops.write_lines(test_file, lines)
        assert result is True
        assert test_file.exists()

        content = test_file.read_text()
        assert content == "\n".join(lines)
