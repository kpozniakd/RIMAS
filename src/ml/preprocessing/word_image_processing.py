from typing import Tuple
from pathlib import Path
import numpy as np
import cv2


# refactor from class
def process_line(self, img_path: Path, label: str, out_dir: Path):
    pass


def load_gray(self, path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def prepare_masks(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    bw = self._binarize_invert(gray)
    dil = self._dilate(bw, self.kernel)
    cleaned = (dil > 0).astype(np.uint8)
    return dil, cleaned


def find_word_contours(self, dilated: np.ndarray, img_shape: Tuple[int, int]):
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img_shape[0] * img_shape[1]
    min_area = img_area * self.min_area_ratio
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def binarize_invert(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def dilate(bw: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.dilate(bw, kernel, iterations=1)


def crop_to_content(img: np.ndarray) -> np.ndarray:
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
