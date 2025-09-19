from PIL import Image
import numpy as np


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
