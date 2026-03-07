"""
Configuration management for DataFlow.
"""

from pathlib import Path
from typing import Dict, Any

# Note: cv2 is imported when needed in visualization modules
# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    # Visualization settings
    "visualization": {
        "bbox_color": (0, 255, 0),  # Green (BGR format)
        "bbox_thickness": 2,
        "text_color": (255, 255, 255),  # White (BGR)
        "text_background": (0, 0, 0),  # Black (BGR)
        "text_scale": 0.5,
        "text_thickness": 1,
    },
    # Conversion settings
    "conversion": {
        "default_class_names": [],  # Will be loaded from file if needed
        "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    },
    # Path settings
    "paths": {
        "default_save_dir": str(Path.home() / ".dataflow"),
        "class_names_file": "classes.txt",
    },
    # Batch visualization settings
    "batch": {
        "navigation_keys": {
            "previous": ["left", "a"],
            "next": ["right", "d"],
            "quit": ["q", "escape"]
        },
        "instructions_text": "← previous | → next | q quit",
        "auto_advance_delay": 0,  # ms, 0 for manual navigation
    },
}

def get_config() -> Dict[str, Any]:
    """
    Get configuration.

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    return DEFAULT_CONFIG.copy()