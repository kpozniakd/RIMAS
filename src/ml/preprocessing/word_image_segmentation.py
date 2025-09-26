from __future__ import annotations
from typing import Tuple, List
from pathlib import Path
import cv2
import numpy as np
from .word_image_processing import binarize_invert, dilate, crop_to_content, load_gray
import matplotlib.pyplot as plt
from core.config.config import Config


def prepare_masks(
    gray: np.ndarray, kernel: Tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    bw = binarize_invert(gray)
    dil = dilate(bw, kernel)
    cleaned = (dil > 0).astype(np.uint8)
    return dil, cleaned


def find_word_contours(
    dilated: np.ndarray,
    img_shape: Tuple[int, int],
    min_area_ratio: float = Config.DEFAULT_MIN_AREA_RATIO,
) -> List[np.ndarray]:
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img_shape[0] * img_shape[1]
    min_area = img_area * min_area_ratio
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def sort_left_to_right(contours: List[np.ndarray]) -> List[np.ndarray]:
    return sorted(contours, key=lambda c: cv2.boundingRect(c)[0])


def segment_word_crops(
    gray: np.ndarray,
    line_text: str,
    *,
    kernel: Tuple[int, int] = Config.DEFAULT_KERNEL,
    min_area_ratio: float = Config.DEFAULT_MIN_AREA_RATIO,
) -> List[tuple[np.ndarray, str]]:
    dil, cleaned = prepare_masks(gray, kernel)
    contours = find_word_contours(dil, gray.shape, min_area_ratio)
    if not contours:
        return []
    contours = sort_left_to_right(contours)

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
    row,
    img_root: Path,
    *,
    kernel: Tuple[int, int] = Config.DEFAULT_KERNEL,
    min_area_ratio: float = Config.DEFAULT_MIN_AREA_RATIO,
):
    rel = row["Filenames"]
    text = row.get("Contents", "")
    img_path = (img_root / rel) if img_root else Path(rel)

    gray = load_gray(img_path)
    if gray is None:
        print(f"[WARN] Can't read image: {img_path}")
        return

    dil, cleaned = prepare_masks(gray, kernel)
    contours = sort_left_to_right(find_word_contours(dil, gray.shape, min_area_ratio))
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
