"""
File operations utility for DataFlow-CV.

Provides cross-platform file operations with proper error handling and logging.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union


class FileOperations:
    """Cross-platform file operations utility class."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def ensure_dir(self, dir_path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if it doesn't."""
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {path}")
        return path

    def safe_remove(self, file_path: Union[str, Path]) -> bool:
        """Safely remove file or directory if it exists."""
        path = Path(file_path)
        if path.exists():
            try:
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)
                self.logger.info(f"Removed: {path}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to remove {path}: {e}")
                return False
        return False

    def copy_files(
        self, src_pattern: str, dst_dir: Union[str, Path], overwrite: bool = False
    ) -> List[Tuple[Path, bool]]:
        """Batch copy files matching a pattern."""
        results = []
        dst_path = Path(dst_dir)
        self.ensure_dir(dst_path)

        # Parse source pattern, support patterns with directories (e.g., "data/*.json")
        src_path = Path(src_pattern)
        if src_path.parent != Path("."):
            # Pattern includes directory part
            search_dir = src_path.parent
            pattern = src_path.name
        else:
            # Simple pattern, search in current directory
            search_dir = Path(".")
            pattern = src_pattern

        for src_file in search_dir.glob(pattern):
            dst_file = dst_path / src_file.name
            if dst_file.exists() and not overwrite:
                self.logger.warning(f"File exists, skipping: {dst_file}")
                results.append((dst_file, False))
                continue

            try:
                shutil.copy2(src_file, dst_file)
                results.append((dst_file, True))
                self.logger.info(f"Copied: {src_file} -> {dst_file}")
            except Exception as e:
                self.logger.error(f"Failed to copy {src_file}: {e}")
                results.append((dst_file, False))

        return results

    def find_files(
        self, directory: Union[str, Path], pattern: str = "*", recursive: bool = True
    ) -> List[Path]:
        """Find files matching pattern in directory."""
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if recursive:
            return list(dir_path.rglob(pattern))
        else:
            return list(dir_path.glob(pattern))

    def create_temp_dir(self, prefix: str = "dataflow_") -> Path:
        """Create a temporary directory."""
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        self.logger.info(f"Created temp directory: {temp_dir}")
        return Path(temp_dir)

    def get_file_size(self, file_path: Union[str, Path]) -> int:
        """Get file size in bytes."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return path.stat().st_size

    def read_lines(
        self, file_path: Union[str, Path], encoding: str = "utf-8"
    ) -> List[str]:
        """Read all lines from a file."""
        path = Path(file_path)
        with open(path, "r", encoding=encoding) as f:
            return [line.strip() for line in f.readlines()]

    def write_lines(
        self, file_path: Union[str, Path], lines: List[str], encoding: str = "utf-8"
    ) -> bool:
        """Write multiple lines to a file."""
        path = Path(file_path)
        try:
            with open(path, "w", encoding=encoding) as f:
                f.write("\n".join(lines))
            self.logger.info(f"Written {len(lines)} lines to: {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to write to {path}: {e}")
            return False
