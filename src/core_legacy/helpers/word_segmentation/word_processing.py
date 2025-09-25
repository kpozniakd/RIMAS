from PIL import Image
import numpy as np
import pandas as pd
import os
import re
import json
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Callable, Any, Union


def crop_to_content(img):
    """
    Crops away border areas where all pixels are white (255);
    expects a single-channel (grayscale) image.
    """
    mask = img < 255
    if not np.any(mask):
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top, bottom = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    left, right = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    return img[top:bottom, left:right]


def resize_to_height(arr, target_height=64, resample=Image.BILINEAR):
    """
    Resizes a 2-D (H×W) to the given height while preserving aspect ratio; returns a NumPy array.
    """
    img = Image.fromarray(np.uint8(arr))
    h, w = img.size[1], img.size[0]

    ratio = target_height / h
    target_width = int(round(w * ratio))

    img = img.resize((target_width, target_height), resample=resample)
    return np.array(img)


def crop_word_to_content(words_df):
    """
    For each row in words_df, crops the 'image' to content and normalizes its height; returns a copy.
    """
    words_df_normalized = words_df.copy(deep=True)

    words_df_normalized["image"] = words_df_normalized["image"].apply(crop_to_content)
    words_df_normalized["image"] = words_df_normalized["image"].apply(resize_to_height)
    return words_df_normalized


# Mapping for filesystem-safe names of special symbols
SYMBOL_NAMES = {
    "?": "questionmark",
    '"': "doublequote",
    "/": "forwardslash",
    ":": "colon",
}

SAFE_EXT = ".png"
JSON_NAME = "words.json"


def only_symbols(s: str) -> bool:
    """Return True if s is non-empty and all chars are in SYMBOL_NAMES."""
    return len(s) > 0 and all(ch in SYMBOL_NAMES for ch in s)


def symbols_to_names_joined(s: str) -> str:
    """Convert symbol chars to names joined with underscores."""
    parts = [SYMBOL_NAMES[ch] for ch in s if ch in SYMBOL_NAMES]
    return "_".join(parts) if parts else "word"


def sanitize_base_name(label: str) -> str:
    """
    Make a safe lowercase base name for filenames.
    - If label has only mapped symbols -> use their names.
    - Else remove mapped symbols, then keep [a-z0-9_-], replace others with '_'.
    """
    lbl = str(label).lower()

    if only_symbols(lbl):
        return symbols_to_names_joined(lbl)

    cleaned = "".join(ch for ch in lbl if ch not in SYMBOL_NAMES)
    base = re.sub(r"[^a-z0-9_\-]+", "_", cleaned).strip("_")
    return base or "word"


def to_pil(img_like: Union[np.ndarray, Image.Image]) -> Image.Image:
    """Convert a numpy array or PIL Image to a PIL Image (uint8)."""
    if isinstance(img_like, Image.Image):
        return img_like
    arr = np.asarray(img_like, dtype=np.uint8)
    return Image.fromarray(arr)


def save_to_folder(
    word_df: pd.DataFrame, output_path: str | os.PathLike
) -> Tuple[str, str]:
    """
    Save images and write words.json manifest.
    """
    words_root = Path(output_path) / "words"
    images_dir = words_root / "word"
    images_dir.mkdir(parents=True, exist_ok=True)

    counters = defaultdict(int)
    manifest: dict[str, list[dict[str, str]]] = {"words": []}

    for row in word_df.itertuples(index=False):
        label = getattr(row, "label")
        image_obj = getattr(row, "image")

        base = sanitize_base_name(label)
        idx = counters[base]
        counters[base] += 1

        file_name = f"{base}_{idx:05d}{SAFE_EXT}"
        full_path = images_dir / file_name

        try:
            img = to_pil(image_obj)
            img.save(full_path)
        except Exception as e:
            print(f"Failed to save '{file_name}': {e}")
            continue

        manifest["words"].append(
            {
                "filename": str(full_path),
                "label": str(label),
            }
        )

    json_path = words_root / JSON_NAME
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done! Images: {images_dir}\nJson: {json_path}")
