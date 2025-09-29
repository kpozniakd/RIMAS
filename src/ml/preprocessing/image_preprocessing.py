import sys
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))


class ImageProcessor:
    """Handles image processing operations."""

    @staticmethod
    def apply_morphological_operations(
        binary_image: np.ndarray, kernel_size: Tuple[int, int] = (2, 2)
    ) -> np.ndarray:
        """Apply morphological operations to clean up binary image."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        morph = cv2.morphologyEx(
            binary_image, cv2.MORPH_CLOSE, kernel=kernel, iterations=2
        )
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel=kernel, iterations=1)
        return morph

    @staticmethod
    def preprocess_image(
        image_path: str, target_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess raw image."""
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Resize image
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale if needed
        if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
            grey = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        else:
            grey = resized_image

        # Denoise image
        denoised_image = cv2.fastNlMeansDenoising(
            grey, h=20, templateWindowSize=9, searchWindowSize=21
        )

        # Apply adaptive threshold
        binary_image = cv2.adaptiveThreshold(
            denoised_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15,
        )

        # Apply morphological operations
        binary_image = ImageProcessor.apply_morphological_operations(binary_image)

        return binary_image, grey

    @staticmethod
    def visualize_image(image: np.ndarray, window_name: str = "Image") -> None:
        """Visualize image in a window."""
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
