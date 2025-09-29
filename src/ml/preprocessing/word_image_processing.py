from typing import Tuple
from pathlib import Path
import numpy as np
import cv2


def load_gray(path: Path) -> np.ndarray:
    """Load image as grayscale."""
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def binarize_invert(gray: np.ndarray) -> np.ndarray:
    """Binarize grayscale image and invert it."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def dilate(bw: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
    """Dilate binary mask with given kernel size."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.dilate(bw, kernel, iterations=1)


def crop_to_content(img: np.ndarray) -> np.ndarray:
    """Crop image to the bounding box of non-white pixels."""
    mask = img < 255
    if not np.any(mask):
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1])
    left = np.argmax(cols)
    right = len(cols) - np.argmax(cols[::-1])
    return img[top:bottom, left:right]
