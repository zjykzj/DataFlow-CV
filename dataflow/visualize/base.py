"""
Base visualizer class for DataFlow.
"""

import os
# Set Qt environment variables before importing cv2 to suppress warnings
os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts'
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.*=false'
# Suppress specific font warnings
os.environ['QT_QPA_NO_FONTDATABASE'] = '1'
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any
from pathlib import Path


class BaseVisualizer:
    """Base class for all visualizers."""

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Load image from file.

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array (BGR format)
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return img

    @staticmethod
    def draw_bbox(image: np.ndarray, bbox: List[float], color: Tuple[int, int, int],
                  thickness: int = 2, label: str = None) -> np.ndarray:
        """
        Draw bounding box on image.

        Args:
            image: Input image
            bbox: [x1, y1, x2, y2] (top-left and bottom-right corners)
            color: BGR color tuple
            thickness: Line thickness
            label: Optional label text

        Returns:
            Image with bounding box drawn
        """
        if len(bbox) != 4:
            raise ValueError(f"Invalid bbox format: {bbox}")

        x1, y1, x2, y2 = bbox
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Draw label if provided
        if label:
            # Text settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1  # Filled
            )

            # Draw text
            cv2.putText(
                image,
                label,
                (x1, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness
            )

        return image

    @staticmethod
    def get_color(class_idx: int, total_classes: int = 20) -> Tuple[int, int, int]:
        """
        Get a distinct color for a class index.

        Args:
            class_idx: Class index
            total_classes: Total number of classes (for color distribution)

        Returns:
            BGR color tuple
        """
        # Generate distinct colors using HSV color wheel
        hue = (class_idx * 180) % 180  # Use 180 degree range for better distinction
        saturation = 255
        value = 255

        # Convert HSV to BGR
        hsv_color = np.uint8([[[hue, saturation, value]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

        return tuple(map(int, bgr_color))

    @staticmethod
    def show_image(image: np.ndarray, window_name: str = "Visualization") -> None:
        """
        Display image in a window.

        Args:
            image: Image to display
            window_name: Window title
        """
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def save_image(image: np.ndarray, save_path: str) -> None:
        """
        Save image to file.

        Args:
            image: Image to save
            save_path: Path to save image
        """
        from pathlib import Path
        save_path_obj = Path(save_path)

        # Ensure parent directory exists
        output_dir = save_path_obj.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure file has a valid image extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        original_suffix = save_path_obj.suffix.lower()
        if original_suffix not in valid_extensions:
            # Add .jpg extension if no valid extension
            save_path_obj = save_path_obj.with_suffix('.jpg')
            save_path = str(save_path_obj)

        try:
            success = cv2.imwrite(save_path, image)
            if not success:
                # Try with different extensions if .jpg fails
                for ext in ['.png', '.jpeg', '.bmp']:
                    if ext != save_path_obj.suffix.lower():
                        alt_path = save_path_obj.with_suffix(ext)
                        success = cv2.imwrite(str(alt_path), image)
                        if success:
                            raise ValueError(f"Failed to save image to {save_path}, but saved to {alt_path}")
                raise ValueError(f"Failed to save image to {save_path} - cv2.imwrite returned False")
        except Exception as e:
            # Re-raise with more context
            raise ValueError(f"Failed to save image to {save_path}: {e}")

    @staticmethod
    def show_batch_navigation(
        image,
        window_name,
        current_idx,
        total,
        instructions="← previous | → next | q quit | Enter next"
    ):
        """
        Display image with navigation controls and wait for key press.

        Args:
            image: Image to display
            window_name: Window title
            current_idx: Current index (0-based)
            total: Total number of images
            instructions: Navigation instructions

        Returns:
            Key pressed ('left', 'right', 'q', or 'other')
        """
        # Close any existing windows to avoid accumulation
        cv2.destroyAllWindows()

        # Add progress and instructions to window title
        progress = f" ({current_idx + 1}/{total})"
        full_title = f"{window_name}{progress}"

        # Create a copy of the image to add instructions overlay
        display_image = image.copy()

        # Display instructions on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        color = (255, 255, 255)  # White
        bg_color = (0, 0, 0)     # Black

        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(
            instructions, font, font_scale, thickness
        )

        # Draw background rectangle
        cv2.rectangle(
            display_image,
            (10, display_image.shape[0] - text_height - 20),
            (10 + text_width + 10, display_image.shape[0] - 10),
            bg_color,
            -1
        )

        # Draw text
        cv2.putText(
            display_image,
            instructions,
            (15, display_image.shape[0] - 15),
            font,
            font_scale,
            color,
            thickness
        )

        # Show image
        cv2.imshow(full_title, display_image)
        cv2.waitKey(1)  # Allow window to update

        # Wait for key press
        while True:
            key = cv2.waitKey(0)
            key_code = key & 0xFF

            # Map key codes to actions
            if key_code == ord('q'):
                cv2.destroyAllWindows()
                return 'q'
            elif key_code == ord('a'):  # 'a' for left
                cv2.destroyAllWindows()
                return 'left'
            elif key_code == ord('d'):  # 'd' for right
                cv2.destroyAllWindows()
                return 'right'
            elif key_code == 13 or key_code == 10:  # Enter key
                cv2.destroyAllWindows()
                return 'right'
            elif key_code == 32:  # Space key
                cv2.destroyAllWindows()
                return 'right'
            elif key_code == 0:  # Extended key (arrows)
                ext_code = (key >> 8) & 0xFF
                if ext_code == 81:  # Left arrow
                    cv2.destroyAllWindows()
                    return 'left'
                elif ext_code == 83:  # Right arrow
                    cv2.destroyAllWindows()
                    return 'right'
            # Unknown key, return 'other'
            cv2.destroyAllWindows()
            return 'other'