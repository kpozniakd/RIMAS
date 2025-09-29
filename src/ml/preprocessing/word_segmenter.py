from __future__ import annotations
from typing import Tuple, List
from pathlib import Path
import cv2
import numpy as np
from .word_image_processing import binarize_invert, dilate, crop_to_content, load_gray
import matplotlib.pyplot as plt
from core.config.config import Config
import logging


logger = logging.getLogger(__name__)


class WordSegmenter:
    def __init__(
        self,
        kernel: Tuple[int, int] = Config.DEFAULT_KERNEL,
        min_area_ratio: float = Config.DEFAULT_MIN_AREA_RATIO,
    ) -> None:
        self.kernel = kernel
        self.min_area_ratio = min_area_ratio

    def process_line(
        self, img_path: Path, line_text: str
    ) -> List[Tuple[np.ndarray, str]]:
        gray = load_gray(img_path)
        if gray is None:
            logger.warning("Can't read image: %s", img_path)
            return []
        pairs = self.segment_word_crops(
            gray, line_text, kernel=self.kernel, min_area_ratio=self.min_area_ratio
        )
        if not pairs:
            logger.info("Skip: no contours or token mismatch — %s", img_path)
        return pairs

    def prepare_masks(
        self, gray: np.ndarray, kernel: Tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create dilated and cleaned masks from a grayscale image."""
        bw = binarize_invert(gray)
        dil = dilate(bw, kernel)
        cleaned = (dil > 0).astype(np.uint8)
        return dil, cleaned

    def find_word_contours(
        self,
        dilated: np.ndarray,
        img_shape: Tuple[int, int],
        min_area_ratio: float = Config.DEFAULT_MIN_AREA_RATIO,
    ) -> List[np.ndarray]:
        """Find contours above the minimum area ratio."""
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        img_area = img_shape[0] * img_shape[1]
        min_area = img_area * min_area_ratio
        return [c for c in contours if cv2.contourArea(c) >= min_area]

    def sort_left_to_right(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Sort contours from left to right."""
        return sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    def segment_word_crops(
        self,
        gray: np.ndarray,
        line_text: str,
        *,
        kernel: Tuple[int, int] = Config.DEFAULT_KERNEL,
        min_area_ratio: float = Config.DEFAULT_MIN_AREA_RATIO,
    ) -> List[tuple[np.ndarray, str]]:
        """Segment a text line into word crops paired with tokens."""
        dil, cleaned = self.prepare_masks(gray, kernel)
        contours = self.find_word_contours(dil, gray.shape, min_area_ratio)
        if not contours:
            return []
        contours = self.sort_left_to_right(contours)

        tokens = line_text.split()
        if len(tokens) != len(contours):
            return []

        pairs: List[tuple[np.ndarray, str]] = []
        for cnt, label in zip(contours, tokens):
            x, y, w, h = cv2.boundingRect(cnt)
            crop = gray[y : y + h, x : x + w].copy()
            crop_mask = cleaned[y : y + h, x : x + w]
            crop[crop_mask == 0] = 255
            crop = crop_to_content(crop)
            pairs.append((crop, label))
        return pairs

    def visualize_line_by_path(
        self,
        row,
        img_root: Path,
        kernel: Tuple[int, int] = Config.DEFAULT_KERNEL,
        min_area_ratio: float = Config.DEFAULT_MIN_AREA_RATIO,
    ):
        """Visualize segmentation results for a single dataset row."""
        rel = row["Filenames"]
        text = row.get("Contents", "")
        img_path = (img_root / rel) if img_root else Path(rel)

        gray = load_gray(img_path)
        if gray is None:
            print(f"[WARN] Can't read image: {img_path}")
            return

        dil, cleaned = self.prepare_masks(gray, kernel)
        contours = self.sort_left_to_right(
            self.find_word_contours(dil, gray.shape, min_area_ratio)
        )
        tokens = (text or "").split()
        matched = len(tokens) == len(contours)

        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if i < len(tokens):
                cv2.putText(
                    vis,
                    tokens[i],
                    (x, max(0, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        plt.figure(figsize=(12, 4))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f"Words (green) | tokens (red) | matched={matched}")
        plt.axis("off")
        plt.show()
