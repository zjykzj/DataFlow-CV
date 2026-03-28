"""CLI parameter models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CLIConfig:
    """CLI configuration dataclass."""

    verbose: bool = False
    log_dir: Path = Path("./logs")
    strict: bool = True
    skip_errors: bool = False


@dataclass
class VisualizeParams:
    """Visualization parameters dataclass."""

    format: str  # yolo/coco/labelme
    input_path: Path
    image_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    display: bool = False
    save: bool = True
    color_scheme: str = "random"
    thickness: int = 2


@dataclass
class ConvertParams:
    """Conversion parameters dataclass."""

    direction: str  # yolo2coco etc.
    input_path: Path
    output_path: Path
    image_dir: Optional[Path] = None
    class_file: Optional[Path] = None
    do_rle: bool = False
    category_mapping: Optional[Dict[str, Any]] = None