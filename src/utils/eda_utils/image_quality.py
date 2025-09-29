import numpy as np
import cv2
import os
from PIL import Image
import pandas as pd


def is_blank(name: str, img_dir: str, BLANK_STD_THRESHOLD: int) -> bool:
    """Check if image is mostly blank (low standard deviation)."""
    try:
        img = Image.open(os.path.join(img_dir, name)).convert("L")
        return np.array(img).std() < BLANK_STD_THRESHOLD
    except:
        return False


def is_sharp(name: str, img_dir: str, SHARP_LAPLACIAN_THRESHOLD: int) -> bool:
    """Check if image is sharp based on Laplacian variance."""
    try:
        img = cv2.imread(os.path.join(img_dir, name), cv2.IMREAD_GRAYSCALE)
        return cv2.Laplacian(img, cv2.CV_64F).var() > SHARP_LAPLACIAN_THRESHOLD
    except:
        return False


def is_noisy(name: str, img_dir: str, NOISE_STD_MEAN_THRESHOLD: int) -> bool:
    """Estimate image noise by std/mean ratio after blur."""
    try:
        img = cv2.imread(os.path.join(img_dir, name), cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return (
            (np.std(img) / np.mean(img)) > NOISE_STD_MEAN_THRESHOLD
            if np.mean(img) > 0
            else False
        )
    except:
        return False


def is_low_contrast(name: str, img_dir: str, CONTRAST_DIFF_THRESHOLD: int) -> bool:
    """Check if image has low contrast based on intensity range."""
    try:
        img = Image.open(os.path.join(img_dir, name)).convert("L")
        img_np = np.array(img)
        return img_np.max() - img_np.min() < CONTRAST_DIFF_THRESHOLD
    except:
        return False


def check_image_quality(
    df: pd.DataFrame,
    img_dir: str,
    BLANK_STD_THRESHOLD: int,
    SHARP_LAPLACIAN_THRESHOLD: int,
    NOISE_STD_MEAN_THRESHOLD: int,
    CONTRAST_DIFF_THRESHOLD: int,
) -> None:
    """Flag images that are blank, blurry, noisy or low-contrast."""
    df["is_blank"] = df["Filenames"].apply(
        lambda f: is_blank(f, img_dir, BLANK_STD_THRESHOLD)
    )
    df["is_sharp"] = df["Filenames"].apply(
        lambda f: is_sharp(f, img_dir, SHARP_LAPLACIAN_THRESHOLD)
    )
    df["is_noisy"] = df["Filenames"].apply(
        lambda f: is_noisy(f, img_dir, NOISE_STD_MEAN_THRESHOLD)
    )
    df["is_low_contrast"] = df["Filenames"].apply(
        lambda f: is_low_contrast(f, img_dir, CONTRAST_DIFF_THRESHOLD)
    )

    print(f"Blank images: {df['is_blank'].sum()}")
    print(f"Blurry images: {(~df['is_sharp']).sum()}")
    print(f"Noisy images: {df['is_noisy'].sum()}")
    print(f"Low contrast images: {df['is_low_contrast'].sum()}")
